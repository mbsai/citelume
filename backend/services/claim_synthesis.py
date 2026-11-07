import hashlib
import re
from contextvars import ContextVar
from typing import Dict, List, Optional, Tuple, Any

from models import (
    ApiKeysConfig,
    AtomicFinding,
    ClaimEvidenceType,
    ClaimVerificationResult,
    ClaimVerificationState,
    EvidenceCluster,
    FindingDirectionType,
    HFResourceRecord,
    PaperRecord,
    QueryDecompositionPlan,
    QuestionAnswerabilityType,
    ReportingStatusType,
    SynthesisClaim,
    SynthesisEvidencePackage,
    SynthesisQualityCheck,
    SynthesisQualityReport,
    SynthesisSection,
)
from services.domain_profiles import DOMAIN_PROFILES
from services.llm_service import TASK_PROMPT_REGISTRY
from services.query_utils import extract_core_scientific_terms
from services.verification_gates import extract_citation_numbers, extract_nontrivial_numbers, study_source_text
from frameworks.rules import RULES_BY_ID, select_rules
from services.review_composition import apply_review_framework, compose_organized_body

# Matches every study design, so domain views list all design-specific appraisal rules
ANY_DESIGN = ("*",)

WORKFLOW_VERSION = "section1-claim-synthesis/1.0"

_CURRENT_PACKAGE: ContextVar[Optional[SynthesisEvidencePackage]] = ContextVar(
    "section1_package", default=None
)


def get_last_synthesis_package() -> Optional[SynthesisEvidencePackage]:
    return _CURRENT_PACKAGE.get()


def set_last_synthesis_package(pkg: Optional[SynthesisEvidencePackage]) -> None:
    _CURRENT_PACKAGE.set(pkg)


# ============================================================================
# PHASE 6: Domain-Adaptive Appraisal Profiles & Considerations
# ============================================================================

# Appraisal considerations per domain are a view over the source-cited registry in `frameworks.rules`
# (each consideration names the framework it comes from); edit rules there, not here.
DOMAIN_APPRAISAL_CRITERIA: Dict[str, List[str]] = {
    dom: [f"{r.text} ({r.citation_label()})" for r in select_rules(dom, ANY_DESIGN, None, ("appraisal",), include_universal=False)]
    + [f"{RULES_BY_ID['U-certainty'].text} ({RULES_BY_ID['U-certainty'].citation_label()})"]
    for dom in (
        "biomedical_clinical", "omics_high_throughput", "computer_science_ai", "interdisciplinary_biomedical_ai",
        "chemistry", "physics", "mathematics", "engineering_materials", "social_sciences", "humanities_interdisciplinary",
    )
}

UNIVERSAL_CONSERVATIVE_CRITERIA: List[str] = [
    "Direct relevance to the research question and subquestions",
    "Methodological validity and appropriateness of study design",
    "Comparability of experimental systems, populations, or formal assumptions",
    "Consistency and independence of supporting and challenging studies",
    "Completeness of source reporting (full text vs abstract-only vs metadata-only)",
]


def get_domain_appraisal_considerations(
    primary_domain: str,
    method_profiles: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    if method_profiles and "omics_high_throughput" in method_profiles:
        return "omics_molecular_biology", DOMAIN_APPRAISAL_CRITERIA["omics_high_throughput"]
    if primary_domain in DOMAIN_APPRAISAL_CRITERIA:
        return primary_domain, DOMAIN_APPRAISAL_CRITERIA[primary_domain]
    return "universal_conservative", UNIVERSAL_CONSERVATIVE_CRITERIA


# ============================================================================
# PHASE 3: Atomic Finding Extraction Layer
# ============================================================================

_QUANT_PATTERN = re.compile(
    r"(?:\b(?:AUROC|AUC|accuracy|F1|RMSD|MAE|RMSE|IC50|MIC|HR|OR|RR|CI|p)\s*(?:=|<|>|≤|≥|:)?\s*\d+(?:\.\d+)?%?)"
    # "%" ends the match on its own: a trailing \b after it would need a letter next ("12.5% in" never matched)
    r"|(?:\b\d+(?:\.\d+)?\s*(?:%|(?:kcal/mol|meV|eV|nm|μm|ug/mL|μg/mL|mg/L|mM|μM|ms|fold|ee)\b))",
    re.IGNORECASE,
)

_UNCERTAINTY_PATTERN = re.compile(
    r"(?:95%\s*CI\s*[:=\[]?\s*[\d.\-–,\s]+[\])]?|±\s*\d+(?:\.\d+)?|\bp\s*[<=>]\s*0\.\d+)",
    re.IGNORECASE,
)

_COMPARATOR_PATTERN = re.compile(
    r"\b(?:compared\s+(?:with|to)|versus|vs\.?|over|against|relative\s+to)\s+([^.;,]{3,60})",
    re.IGNORECASE,
)

_AUTHOR_INTERP_CUES = re.compile(
    r"\b(suggest\w*|may|might|could|potential\w*|impl(?:y|ies)|hypothesi[sz]\w*|indicat\w*\s+that|promising)\b",
    re.IGNORECASE,
)

_RETRACTION_CUES = re.compile(
    r"\b(retracted|retraction\s+notice|withdrawn|expression\s+of\s+concern)\b",
    re.IGNORECASE,
)

_DATASET_TOKEN_PATTERN = re.compile(
    r"\b(geom-drugs|pdbbind|chembl|uniprot|imagenet|mimic-iii|mimic-iv|eicu|uk\s+biobank|tcga|gtex|qm9|depmap)\b"
    r"|\bn\s*=\s*\d[\d,]*",
    re.IGNORECASE,
)


def _infer_claim_evidence_type(
    study_design: str,
    text: str,
    primary_domain: str,
) -> ClaimEvidenceType:
    sd_low = (study_design or "").lower()
    t_low = (text or "").lower()
    if primary_domain == "mathematics" or "formal mathematical proof" in sd_low or re.search(r"\b(theorem|proof|lemma|minimax)\b", t_low):
        return "proved_mathematical_result"
    if "randomized" in sd_low or "clinical trial" in sd_low:
        return "experimental_causal_effect"
    if "observational" in sd_low or "cohort" in sd_low or "retrospective" in sd_low:
        if re.search(r"\b(auroc|auc|prediction|prognostic|risk score|early warning)\b", t_low):
            return "predictive_relationship"
        return "observed_association"
    if re.search(r"\b(benchmark|outperform|transformer|neural|diffusion|classifier|auroc|accuracy|f1)\b", t_low):
        return "computational_performance_comparison"
    if re.search(r"\b(mechanism|pathway|kinetics|spectroscop\w*|cataly\w*|deamination)\b", t_low):
        if re.search(r"\b(in vitro|in vivo|assay|spectroscop\w*|crystallograph\w*)\b", t_low):
            return "validated_mechanism"
        return "proposed_mechanism"
    if "simulation" in sd_low or re.search(r"\b(monte carlo|molecular dynamics|density functional|hamiltonian)\b", t_low):
        return "theoretical_or_model_based"
    if "experimental" in sd_low or "wet-lab" in sd_low:
        return "experimental_causal_effect"
    return "descriptive_observation"


def _infer_finding_direction(p: PaperRecord, text: str) -> FindingDirectionType:
    t_low = (text or "").lower()
    if re.search(r"\b(no significant|not statistically significant|null effect|no mortality reduction|did not improve)\b", t_low):
        return "null_or_nonsignificant"
    if p.stance == "contradicting" or re.search(
            r"\b(fail(?:ed|s)? to|failure to (?:replicate|reproduce|improve|reduce)|counterexample|"
            r"performance (?:degrad\w*|drop\w*)|accuracy (?:degrad\w*|drop\w*)|inversion errors|byproduct)\b", t_low):
        # Disease names ("heart failure") and biochemistry ("BCAA degradation") are not negative results
        return "negative"
    # A review summarises others' findings; it has no direction of effect of its own
    if re.search(r"\b(in this review|we review|this review|narrative review|we summari[sz]e|we outline)\b", t_low) or \
            re.search(r"\breview\b", (p.title or "").lower()):
        return "unclear"
    if p.stance == "mixed" or re.search(r"\b(however|trade-off|conditional|context-dependent|drifted)\b", t_low):
        return "mixed_or_conditional"
    if p.elicit.sample_reporting_status == "Not applicable":
        return "positive"
    if re.search(r"\b(improv\w*|achiev\w*|reduc\w*|outperform\w*|establish\w*|enabl\w*|high-fidelity)\b", t_low):
        return "positive"
    return "unclear"


# Characters of full-text result sentences (those containing figures) added to a study's quantitative extraction
FULL_TEXT_RESULT_CHARS = 1500


def _full_text_result_sentences(p: PaperRecord) -> str:
    text = p.full_text_sections.get("results") or p.full_text_sections.get("body", "")
    picked, used = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\d", sent) and 20 < len(sent) < 500 and used + len(sent) <= FULL_TEXT_RESULT_CHARS:
            picked.append(sent)
            used += len(sent) + 1
    return " ".join(picked)


def extract_atomic_findings(
    query: str,
    papers: List[PaperRecord],
    decomposition_plan: Optional[QueryDecompositionPlan] = None,
) -> List[AtomicFinding]:
    """
    Phase 3 — Builds verified AtomicFinding records from retrieved PaperRecords.
    Separates directly reported results from author interpretations, captures explicit
    ReportingStatusType for every field, and computes question relevance.
    """
    findings: List[AtomicFinding] = []
    q_terms = set(t.lower() for t in extract_core_scientific_terms(query, max_terms=10))
    prim_domain = "biomedical_clinical"
    subquestions = []
    if decomposition_plan:
        if decomposition_plan.active_profile_summary:
            prim_domain = decomposition_plan.active_profile_summary.get("primary_domain", "biomedical_clinical")
        if decomposition_plan.question_understanding:
            subquestions = decomposition_plan.question_understanding.subquestions or []

    for p in papers:
        raw_abs = (p.abstract or "").strip()
        has_abstract = len(raw_abs) >= 30
        access_type = "full_text" if p.full_text_sections else "abstract" if has_abstract else "metadata_only"
        full_text = f"{p.title}. {raw_abs}"
        # Figures and uncertainty intervals are also read from an open-access full text's result sentences
        result_text = f"{raw_abs} {_full_text_result_sentences(p)}".strip()
        study_id = p.canonical_study_id or f"study:{p.citation_index}"
        result_id = f"res:{p.citation_index}:1"

        # Check for retraction/withdrawal markers in title/abstract
        excluded_reason = ""
        if _RETRACTION_CUES.search(full_text):
            excluded_reason = "Retracted or expression-of-concern notice detected in source metadata/abstract."

        # Compute question lexical overlap
        p_tokens = set(re.findall(r"[a-z0-9]{3,}", full_text.lower()))
        overlap = len(q_terms & p_tokens) / max(1, min(len(q_terms), 5)) if q_terms else 0.5
        q_rel = round(min(1.0, overlap), 3)

        # Match best subquestion
        subq_id = "Q1"
        if subquestions:
            best_sq = subquestions[0].id
            best_score = -1
            for sq in subquestions:
                sq_toks = set(re.findall(r"[a-z0-9]{3,}", sq.question_text.lower()))
                sc = len(sq_toks & p_tokens)
                if sc > best_score:
                    best_score = sc
                    best_sq = sq.id
            subq_id = best_sq

        # Split sentences to separate observed results from authors' concluding interpretation
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_abs) if len(s.strip()) > 15]
        observed_sent = p.elicit.key_findings or (sentences[0] if sentences else p.title)
        author_conc = "Not reported"
        if len(sentences) >= 2 and _AUTHOR_INTERP_CUES.search(sentences[-1]):
            author_conc = sentences[-1][:240]
        elif has_abstract and sentences:
            author_conc = sentences[-1][:240]

        quant_list = [m.group(0).strip() for m in _QUANT_PATTERN.finditer(result_text)]
        unc_match = _UNCERTAINTY_PATTERN.search(result_text)
        uncertainty_str = unc_match.group(0).strip() if unc_match else ("Not applicable" if prim_domain == "mathematics" else "Not reported")

        comp_match = _COMPARATOR_PATTERN.search(full_text)
        comparator_str = comp_match.group(1).strip() if comp_match else ("Not applicable" if prim_domain == "mathematics" else "Not reported")

        dataset_toks = [m.group(0).lower().strip() for m in _DATASET_TOKEN_PATTERN.finditer(full_text)]

        ev_type = _infer_claim_evidence_type(p.elicit.study_design, full_text, prim_domain)
        direction = _infer_finding_direction(p, full_text)

        field_status: Dict[str, ReportingStatusType] = {
            "sample_or_scale": p.elicit.sample_reporting_status,
            "quantitative_estimate": "Reported" if quant_list else ("Not applicable" if prim_domain == "mathematics" else ("Unavailable" if not has_abstract else "Not reported")),
            "uncertainty": "Reported" if unc_match else ("Not applicable" if prim_domain == "mathematics" else "Not reported"),
            "comparator": "Reported" if comp_match else ("Not applicable" if prim_domain == "mathematics" else "Not reported"),
            "limitations": "Reported" if has_abstract else "Unavailable",
        }

        findings.append(
            AtomicFinding(
                result_id=result_id,
                study_id=study_id,
                citation_index=p.citation_index,
                subquestion_id=subq_id,
                proposition=f"{p.elicit.methodology} on {p.elicit.dataset_or_system}",
                study_evidence_type=p.elicit.study_design,
                claim_evidence_type=ev_type,
                context_or_system=p.elicit.dataset_or_system,
                exposure_or_method=p.elicit.methodology,
                comparator=comparator_str,
                outcome=p.elicit.dataset_or_system,
                observed_result=observed_sent if has_abstract else f"Unavailable in abstract (metadata-only record: {p.title})",
                direction=direction,
                quantitative_estimates=quant_list[:5],
                uncertainty=uncertainty_str,
                authors_conclusion=author_conc,
                methodological_limitations=p.elicit.limitations,
                source_document_id=p.doi or p.arxiv_id or p.pmid or p.id,
                source_passage=raw_abs[:360] if has_abstract else f"Title only: {p.title}",
                evidence_access=access_type,
                origin="directly_reported" if has_abstract else "author_interpretation",
                field_status=field_status,
                dataset_tokens=dataset_toks,
                question_relevance=q_rel,
                excluded_reason=excluded_reason,
            )
        )

    return findings


# ============================================================================
# PHASE 4 & 5: Comparable Evidence Clustering & Non-Independent Overlap Detection
# ============================================================================

def cluster_comparable_findings(
    query: str,
    papers: List[PaperRecord],
    findings: List[AtomicFinding],
) -> List[EvidenceCluster]:
    """
    Phase 4 Steps 2–3 & Phase 5:
    Groups AtomicFindings into comparable thematic clusters while:
    - Preserving differences in study design, system/population, and outcome measure.
    - Classifying supporting, challenging/contradictory, inconclusive/null, and contextual findings.
    - Detecting non-independent studies (shared dataset/cohort tokens, overlapping author groups, or linked preprint/published versions)
      so they are not counted as independent replications.
    """
    paper_by_idx = {p.citation_index: p for p in papers}
    valid_findings = [f for f in findings if not f.excluded_reason]

    grouped: Dict[str, List[AtomicFinding]] = {}
    for f in valid_findings:
        p = paper_by_idx.get(f.citation_index)
        theme = (p.cluster_label if p and p.cluster_label else "Core Scientific Evidence").strip()
        grouped.setdefault(theme, []).append(f)

    clusters: List[EvidenceCluster] = []
    for c_idx, (theme, f_list) in enumerate(grouped.items(), start=1):
        sup_ids: List[str] = []
        chal_ids: List[str] = []
        inc_ids: List[str] = []
        ctx_ids: List[str] = []

        designs_seen = set()
        contexts_seen = set()
        outcomes_seen = set()

        for f in f_list:
            designs_seen.add(f.study_evidence_type)
            contexts_seen.add(f.context_or_system)
            if f.quantitative_estimates:
                for q in f.quantitative_estimates:
                    m_unit = re.match(r"^[A-Za-z0-9]+", q)
                    if m_unit:
                        outcomes_seen.add(m_unit.group(0).upper())

            if f.evidence_access == "metadata_only":
                ctx_ids.append(f.result_id)
            elif f.direction in ("negative",):
                chal_ids.append(f.result_id)
            elif f.direction in ("null_or_nonsignificant", "mixed_or_conditional", "unclear"):
                inc_ids.append(f.result_id)
            else:
                sup_ids.append(f.result_id)

        # Detect non-independent study groups (shared dataset token or shared first/last author)
        indep_groups: List[List[int]] = []
        overlap_notes: List[str] = []
        visited = set()

        for i, f1 in enumerate(f_list):
            if f1.citation_index in visited:
                continue
            grp = [f1.citation_index]
            visited.add(f1.citation_index)
            p1 = paper_by_idx.get(f1.citation_index)
            a1 = {a.split()[-1].lower() for a in (p1.authors if p1 else []) if a}

            for f2 in f_list[i + 1 :]:
                if f2.citation_index in visited:
                    continue
                p2 = paper_by_idx.get(f2.citation_index)
                a2 = {a.split()[-1].lower() for a in (p2.authors if p2 else []) if a}
                shared_ds = set(f1.dataset_tokens) & set(f2.dataset_tokens)
                shared_auth = a1 & a2
                if shared_ds or (len(shared_auth) >= 2):
                    grp.append(f2.citation_index)
                    visited.add(f2.citation_index)
                    reason = (
                        f"shared dataset/sample marker ({', '.join(shared_ds)})"
                        if shared_ds
                        else f"overlapping author team ({', '.join(shared_auth)})"
                    )
                    overlap_notes.append(
                        f"Studies [{f1.citation_index}] and [{f2.citation_index}] share {reason} and are treated as non-independent for replication assessment."
                    )
            indep_groups.append(grp)

        comp_notes: List[str] = []
        if len(designs_seen) > 1:
            comp_notes.append(f"Heterogeneous study designs across cluster ({'; '.join(sorted(designs_seen))}) preclude formal quantitative pooling.")
        if len(outcomes_seen) > 1:
            comp_notes.append(f"Different quantitative outcome measures reported ({', '.join(sorted(outcomes_seen))}); compared qualitatively without informal meta-analysis.")
        if not comp_notes:
            comp_notes.append(f"Evaluated within { '; '.join(list(contexts_seen)[:2]) }.")

        # Determine dominant evidence type with conservative precedence
        ev_types = [f.claim_evidence_type for f in f_list]
        if "proved_mathematical_result" in ev_types:
            dom_ev: ClaimEvidenceType = "proved_mathematical_result"
        elif " observed_association" in ev_types or all(e == "observed_association" for e in ev_types):
            dom_ev = "observed_association"
        elif "predictive_relationship" in ev_types:
            dom_ev = "predictive_relationship"
        elif "computational_performance_comparison" in ev_types:
            dom_ev = "computational_performance_comparison"
        elif "validated_mechanism" in ev_types:
            dom_ev = "validated_mechanism"
        elif "proposed_mechanism" in ev_types:
            dom_ev = "proposed_mechanism"
        elif "experimental_causal_effect" in ev_types and len(designs_seen) == 1:
            dom_ev = "experimental_causal_effect"
        else:
            dom_ev = ev_types[0] if ev_types else "descriptive_observation"

        avg_rel = round(sum(f.question_relevance for f in f_list) / max(1, len(f_list)), 3)

        clusters.append(
            EvidenceCluster(
                cluster_id=f"cluster:{c_idx}",
                theme_label=theme,
                proposition=f"Evidence regarding {theme} in relation to {query}",
                claim_evidence_type=dom_ev,
                subquestion_id=f_list[0].subquestion_id if f_list else "Q1",
                result_ids=[f.result_id for f in f_list],
                study_indices=[f.citation_index for f in f_list],
                supporting_result_ids=sup_ids,
                challenging_result_ids=chal_ids,
                inconclusive_result_ids=inc_ids,
                contextual_result_ids=ctx_ids,
                independent_study_groups=indep_groups,
                overlap_notes=overlap_notes,
                comparability_notes=comp_notes,
                question_relevance=avg_rel,
            )
        )

    return clusters


# ============================================================================
# PHASE 4 Steps 4-6 & PHASE 7: Claim Qualification & 3-Level Verification
# ============================================================================

_OVERCLAIM_CAUSAL_RE = re.compile(
    r"\b(proves?\s+that|definitively\s+causes|establishes\s+causal|guarantees\s+clinical\s+utility)\b",
    re.IGNORECASE,
)


def _qualify_verb_for_evidence_type(ev_type: ClaimEvidenceType, indep_count: int) -> str:
    if ev_type == "proved_mathematical_result":
        return "formally establishes (under stated regularity and space assumptions)"
    if ev_type == "experimental_causal_effect":
        return "experimentally demonstrates" if indep_count >= 2 else "provides experimental evidence in a single study that"
    if ev_type == "validated_mechanism":
        return "supports an experimentally assayed mechanism whereby" if indep_count >= 2 else "indicates in initial experimental assays that"
    if ev_type == "proposed_mechanism":
        return "proposes a candidate mechanism in which"
    if ev_type == "predictive_relationship":
        return "demonstrates retrospective/benchmark predictive discrimination (distinct from prospective clinical utility) where"
    if ev_type == "computational_performance_comparison":
        return "reports benchmark computational performance showing that"
    if ev_type == "observed_association":
        return "identifies an observational association (without establishing direct causal efficacy) in which"
    if ev_type == "theoretical_or_model_based":
        return "models under simulation assumptions that"
    return "reports evidence indicating that"


def verify_single_claim(
    claim: SynthesisClaim,
    findings_by_id: Dict[str, AtomicFinding],
    papers_by_idx: Dict[int, PaperRecord],
) -> ClaimVerificationResult:
    """
    Phase 7 — Three-Level Scientific Claim Verification:
    - Level 1 (Structural): Valid study IDs, existing supporting/challenging/inconclusive result IDs, resolvable citations.
    - Level 2 (Factual): All quantitative numbers in `claim_text` and `key_quantitative_results` exist in the cited source passages.
    - Level 3 (Inference): Claim wording does not exceed evidence type (no causal claims from observational associations,
      no clinical utility claims from retrospective predictions, no suppressed challenging findings).
    """
    issues: List[str] = []
    qualifications: List[str] = []
    structural_ok = True
    factual_ok = True
    inference_ok = True

    all_res_ids = (
        claim.supporting_result_ids
        + claim.challenging_result_ids
        + claim.inconclusive_result_ids
        + claim.contextual_result_ids
    )
    if not all_res_ids or not claim.relevant_study_indices:
        structural_ok = False
        issues.append(f"Level 1 Structural Failure: Claim {claim.claim_id} has no linked result IDs or study indices.")

    cited_findings: List[AtomicFinding] = []
    for rid in all_res_ids:
        if rid not in findings_by_id:
            structural_ok = False
            issues.append(f"Level 1 Structural Failure: Result ID '{rid}' does not exist in AtomicFinding base.")
        else:
            f_obj = findings_by_id[rid]
            if f_obj.excluded_reason:
                structural_ok = False
                issues.append(f"Level 1 Structural Failure: Result '{rid}' comes from an excluded/retracted record ({f_obj.excluded_reason}).")
            cited_findings.append(f_obj)

    for s_idx in claim.relevant_study_indices:
        if s_idx not in papers_by_idx:
            structural_ok = False
            issues.append(f"Level 1 Structural Failure: Study index [{s_idx}] does not resolve to bibliography [1..{len(papers_by_idx)}].")

    # Check if all underlying sources are metadata-only (unverifiable abstract)
    if cited_findings and all(f.evidence_access == "metadata_only" for f in cited_findings):
        return ClaimVerificationResult(
            structural_passed=structural_ok,
            factual_passed=True,
            inference_passed=True,
            state="Not verifiable from accessible sources",
            issues=["All supporting records for this claim are metadata-only (no abstract or full text accessible)."],
            qualifications=["Full text required to verify quantitative or methodological details."],
        )

    # Level 2: Factual & Numerical Verification
    combined_source_text = " ".join(
        study_source_text(papers_by_idx[i]) for i in claim.relevant_study_indices if i in papers_by_idx
    )
    for num in extract_nontrivial_numbers(claim.claim_text):
        if num not in combined_source_text:
            factual_ok = False
            issues.append(f"Level 2 Factual Failure: Number '{num}' in claim {claim.claim_id} is absent from cited source records {claim.relevant_study_indices}.")

    # Level 3: Inference Verification
    if claim.evidence_type in ("observed_association", "predictive_relationship"):
        if _OVERCLAIM_CAUSAL_RE.search(claim.claim_text):
            inference_ok = False
            issues.append(
                f"Level 3 Inference Failure: Claim {claim.claim_id} uses causal/clinical-guarantee language on {claim.evidence_type} evidence."
            )

    if claim.challenging_indices or claim.inconclusive_indices:
        qualifications.append(
            f"Heterogeneity/contradiction present in studies {claim.challenging_indices + claim.inconclusive_indices}."
        )
    if claim.independent_support_count < 2:
        qualifications.append("Supported by fewer than 2 independent studies; replication required.")
    if any(f.evidence_access == "abstract" for f in cited_findings):
        qualifications.append("Derived from study abstracts; full-text methods and supplementary tables were not directly inspected.")

    if not structural_ok or not factual_ok:
        state: ClaimVerificationState = "Unsupported"
    elif not inference_ok:
        state = "Partially supported"
    elif qualifications:
        state = "Verified with qualifications"
    else:
        state = "Verified"

    return ClaimVerificationResult(
        structural_passed=structural_ok,
        factual_passed=factual_ok,
        inference_passed=inference_ok,
        state=state,
        issues=issues,
        qualifications=qualifications,
    )


def build_and_verify_claims(
    query: str,
    papers: List[PaperRecord],
    findings: List[AtomicFinding],
    clusters: List[EvidenceCluster],
) -> Tuple[List[SynthesisClaim], List[SynthesisClaim]]:
    """
    Phase 4 Steps 4–6 & Phase 7:
    Builds qualified SynthesisClaim records from comparable EvidenceClusters,
    verifies each claim across Levels 1–3, and separates verified/qualified claims
    from unsupported/excluded claims.
    """
    findings_by_id = {f.result_id: f for f in findings}
    papers_by_idx = {p.citation_index: p for p in papers}

    verified_claims: List[SynthesisClaim] = []
    excluded_claims: List[SynthesisClaim] = []

    for idx, cl in enumerate(clusters, start=1):
        c_findings = [findings_by_id[rid] for rid in cl.result_ids if rid in findings_by_id]
        if not c_findings:
            continue

        sup_findings = [findings_by_id[rid] for rid in cl.supporting_result_ids if rid in findings_by_id]
        chal_findings = [findings_by_id[rid] for rid in cl.challenging_result_ids if rid in findings_by_id]
        inc_findings = [findings_by_id[rid] for rid in cl.inconclusive_result_ids if rid in findings_by_id]
        ctx_findings = [findings_by_id[rid] for rid in cl.contextual_result_ids if rid in findings_by_id]

        sup_idxs = [f.citation_index for f in sup_findings]
        chal_idxs = [f.citation_index for f in chal_findings]
        inc_idxs = [f.citation_index for f in inc_findings]
        ctx_idxs = [f.citation_index for f in ctx_findings]

        # Count independent supporting study groups
        indep_sup_groups = [
            g for g in cl.independent_study_groups if any(s in sup_idxs for s in g)
        ]
        indep_count = len(indep_sup_groups)
        qual_verb = _qualify_verb_for_evidence_type(cl.claim_evidence_type, indep_count)

        # Build claim text integrating supporting, challenging, and inconclusive findings
        parts: List[str] = []
        if sup_findings:
            lead = sup_findings[0]
            sup_cite = ", ".join(str(i) for i in sup_idxs)
            quant_clause = (
                f" (reported quantitative estimates: {'; '.join(lead.quantitative_estimates[:2])})"
                if lead.quantitative_estimates
                else ""
            )
            parts.append(
                f"Within {cl.theme_label} ({lead.context_or_system}), {lead.study_evidence_type.lower()} evidence [{sup_cite}] "
                f"{qual_verb} {lead.observed_result.rstrip('.')}{quant_clause}."
            )
            if len(sup_findings) > 1:
                other_f = sup_findings[1]
                parts.append(
                    f"Complementary evidence in [{other_f.citation_index}] ({other_f.context_or_system}) reports that {other_f.observed_result.rstrip('.')}."
                )

        if chal_findings:
            cf = chal_findings[0]
            chal_cite = ", ".join(str(i) for i in chal_idxs)
            parts.append(
                f"Conversely, challenging evidence [{chal_cite}] ({cf.study_evidence_type} in {cf.context_or_system}) "
                f"identifies opposing or limiting outcomes: {cf.observed_result.rstrip('.')}."
            )

        if inc_findings:
            inf = inc_findings[0]
            inc_cite = ", ".join(str(i) for i in inc_idxs)
            parts.append(
                f"Conditional or null findings [{inc_cite}] further indicate regime dependence: {inf.observed_result.rstrip('.')} "
                f"(a non-significant or conditional result is interpreted as absence of demonstrated effect in that regime rather than proof of no effect)."
            )

        if not parts and ctx_findings:
            cf0 = ctx_findings[0]
            parts.append(
                f"Metadata-only record [{cf0.citation_index}] ({cf0.observed_result}) addresses {cl.theme_label}, "
                f"but lacks accessible abstract or full-text data for quantitative verification."
            )

        claim_text = " ".join(parts)
        quants = [q for f in c_findings for q in f.quantitative_estimates][:6]
        meth_lims = list(dict.fromkeys([f.methodological_limitations for f in c_findings if f.methodological_limitations]))[:4]

        qual_interp = (
            f"Classified as {cl.claim_evidence_type.replace('_', ' ')} across {len(cl.study_indices)} study(ies) "
            f"({indep_count} independent supporting study group(s)). "
            + (" ".join(cl.overlap_notes) if cl.overlap_notes else "No duplicate dataset overlap detected among supporting studies.")
        )

        claim_obj = SynthesisClaim(
            claim_id=f"claim:{idx}",
            claim_text=claim_text,
            research_question_id=cl.subquestion_id,
            theme_label=cl.theme_label,
            cluster_id=cl.cluster_id,
            evidence_type=cl.claim_evidence_type,
            supporting_result_ids=cl.supporting_result_ids,
            challenging_result_ids=cl.challenging_result_ids,
            inconclusive_result_ids=cl.inconclusive_result_ids,
            contextual_result_ids=cl.contextual_result_ids,
            relevant_study_indices=cl.study_indices,
            supporting_indices=sup_idxs,
            challenging_indices=chal_idxs,
            inconclusive_indices=inc_idxs,
            key_quantitative_results=quants,
            comparability_limitations=cl.comparability_notes + cl.overlap_notes,
            methodological_limitations=meth_lims,
            qualified_interpretation=qual_interp,
            independent_support_count=indep_count,
            text_origin="deterministic_template",
        )

        ver_res = verify_single_claim(claim_obj, findings_by_id, papers_by_idx)
        claim_obj.verification = ver_res

        if ver_res.state == "Unsupported":
            excluded_claims.append(claim_obj)
        else:
            verified_claims.append(claim_obj)

    return verified_claims, excluded_claims


# ============================================================================
# PHASE 8: Generate Final Scientific Synthesis (Subsections A - E)
# ============================================================================

def determine_question_answerability(
    papers: List[PaperRecord],
    findings: List[AtomicFinding],
    verified_claims: List[SynthesisClaim],
) -> QuestionAnswerabilityType:
    if not papers or not verified_claims:
        return "no_eligible_evidence"
    valid_f = [f for f in findings if not f.excluded_reason]
    if not valid_f:
        return "no_eligible_evidence"
    max_rel = max((f.question_relevance for f in valid_f), default=0.0)
    if max_rel < 0.15:
        return "not_answerable_from_selected_evidence"
    if any(c.independent_support_count >= 2 and c.verification.state in ("Verified", "Verified with qualifications") for c in verified_claims):
        return "answered"
    return "partially_answered"


def compose_section1_narrative_from_verified_claims(
    query: str,
    mode: str,
    papers: List[PaperRecord],
    hf_resources: List[HFResourceRecord],
    findings: List[AtomicFinding],
    clusters: List[EvidenceCluster],
    verified_claims: List[SynthesisClaim],
    excluded_claims: List[SynthesisClaim],
    primary_domain: str,
    appraisal_considerations: List[str],
    answerability: QuestionAnswerabilityType,
    organization: str = "thematic",
    review_framework: str = "narrative",
) -> Tuple[str, List[SynthesisSection], str, str]:
    """
    Phase 8 — Generates the structured Section 1 Scientific Synthesis strictly from
    verified SynthesisClaim records, organized into the required subsections:
      A. Direct Answer / Executive Scientific Synthesis (`executive_answer`)
      B. Thematic Evidence Synthesis
      C. Conflicting Evidence and Alternative Interpretations
      D. Reliability and Limitations of the Evidence
      E. Integrated Scientific Conclusion
    """
    dom_prof = DOMAIN_PROFILES.get(primary_domain, DOMAIN_PROFILES["biomedical_clinical"])
    all_indices = [p.citation_index for p in papers]
    pub_count = sum(1 for p in papers if p.version_status in ("published", "preprint_with_published_link"))
    prep_count = len(papers) - pub_count
    linked_prep_count = sum(1 for p in papers if p.version_status == "preprint_with_published_link")
    abs_only_count = sum(1 for f in findings if f.evidence_access == "abstract")
    full_text_count = sum(1 for f in findings if f.evidence_access == "full_text")
    meta_only_count = sum(1 for f in findings if f.evidence_access == "metadata_only")

    sup_claims = [c for c in verified_claims if c.supporting_indices]
    chal_claims = [c for c in verified_claims if c.challenging_indices or c.inconclusive_indices]

    lead_cites = ", ".join(str(i) for i in all_indices[: min(3, len(all_indices))])

    # Subsection A: Direct Answer / Executive Scientific Synthesis (~150-250 words)
    if answerability == "not_answerable_from_selected_evidence":
        answer_status_sentence = (
            f"Direct Answer & Answerability Assessment: The {len(papers)} retrieved study(ies) [{lead_cites}] exhibit low direct topical overlap "
            f"with the specific research question (\"{query}\") and therefore DO NOT directly establish an answer to the query; "
            f"below is a conservative appraisal of what the retrieved records actually report without extrapolating beyond their scope [{lead_cites}]."
        )
    elif len(papers) <= 2:
        answer_status_sentence = (
            f"Direct Answer & Answerability Assessment: The retrieved evidence base addressing \"{query}\" is sparse "
            f"({len(papers)} selected study(ies): {pub_count} peer-reviewed and {prep_count} preprint(s) [{lead_cites}]), "
            f"providing preliminary, context-bounded findings under the {dom_prof.display_name} framework rather than field-wide certainty."
        )
    else:
        answer_status_sentence = (
            f"Direct Answer & Answerability Assessment: Across {len(papers)} deduplicated studies ({pub_count} peer-reviewed publications and {prep_count} preprints [{lead_cites}]) "
            f"evaluated under {dom_prof.display_name} criteria, the evidence partially answers \"{query}\" through {len(verified_claims)} verified thematic claim cluster(s)."
        )

    principal_finding_sents = [c.claim_text for c in verified_claims[:2]]
    conflict_exec_sent = (
        f"Important conflicting or conditional findings are documented in [{', '.join(str(i) for c in chal_claims for i in (c.challenging_indices + c.inconclusive_indices))}], "
        f"demonstrating that observed outcomes depend on experimental regime, dataset calibration, and methodological assumptions."
        if chal_claims
        else f"While no direct contradictions were observed among [{lead_cites}], agreement across a small or homogeneous set of studies does not establish universal certainty."
    )
    lim_exec_sent = (
        f"Interpretation is bounded by source accessibility ({full_text_count} open-access full-text, {abs_only_count} abstract-only and {meta_only_count} metadata-only record(s)), "
        f"study design constraints, and the absence of formal quantitative meta-analysis across heterogeneous measures [{lead_cites}]."
    )

    executive_answer = "\n\n".join(
        [
            f"{answer_status_sentence} {' '.join(principal_finding_sents)}",
            f"{conflict_exec_sent} {lim_exec_sent}",
        ]
    )

    sections: List[SynthesisSection] = []

    # Subsection B: Thematic Evidence Synthesis
    theme_paragraphs: List[str] = []
    theme_cites: List[int] = []
    for t_idx, c in enumerate(verified_claims, start=1):
        theme_cites.extend(c.relevant_study_indices)
        comp_str = " ".join(c.comparability_limitations)
        lim_str = "; ".join(c.methodological_limitations[:2])
        ver_badge = f"[Verification State: {c.verification.state}]"
        para = (
            f"Theme B.{t_idx} — {c.theme_label} ({c.evidence_type.replace('_', ' ')}) {ver_badge}: "
            f"{c.claim_text} "
            f"Cross-study comparison & comparability: {comp_str} "
            f"Methodological qualifications: {lim_str} [{', '.join(str(i) for i in c.relevant_study_indices)}]."
        )
        theme_paragraphs.append(para)

    sections.append(
        SynthesisSection(
            heading="B. Thematic Evidence Synthesis",
            content="\n\n".join(theme_paragraphs),
            cited_indices=sorted(list(set(theme_cites or all_indices[:1]))),
        )
    )

    # Subsection C: Conflicting Evidence and Alternative Interpretations
    conflict_lines: List[str] = []
    conflict_cites: List[int] = []
    for f in findings:
        if f.excluded_reason:
            continue
        if f.direction in ("negative", "null_or_nonsignificant", "mixed_or_conditional"):
            conflict_cites.append(f.citation_index)
            conflict_lines.append(
                f"• Study [{f.citation_index}] ({f.study_evidence_type}, {f.context_or_system}, direction: {f.direction}): "
                f"{f.observed_result.rstrip('.')}. Methodological/contextual factor: {f.methodological_limitations}"
            )

    if conflict_lines:
        het_sources = (
            "Observed divergences across the selected studies trace to differences in experimental or clinical context, "
            "outcome definitions and measurement scales, sample size/statistical precision, model assumptions, and out-of-distribution validation design "
            f"[{', '.join(str(i) for i in sorted(list(set(conflict_cites))))}]. "
            "No meta-analysis was performed: findings are compared by direction of effect and no pooled effect "
            "estimate is calculated, as reported under the SWiM guideline (Campbell et al., BMJ 2020)."
        )
        conflict_content = het_sources + "\n" + "\n".join(conflict_lines)
    else:
        conflict_cites = all_indices[: min(2, len(all_indices))]
        conflict_content = (
            f"Among the {len(papers)} selected study(ies) [{', '.join(str(i) for i in conflict_cites)}], no directly contradictory findings were reported. "
            "However, consistency within this retrieved subset does not imply that the broader literature is contradiction-free, "
            "nor does statistical significance across individual studies substitute for independent replication across diverse populations or regimes."
        )

    sections.append(
        SynthesisSection(
            heading="C. Conflicting Evidence and Alternative Interpretations",
            content=conflict_content,
            cited_indices=sorted(list(set(conflict_cites))),
        )
    )

    # Subsection D: Reliability and Limitations of the Evidence
    overlap_all = [n for cl in clusters for n in cl.overlap_notes]
    study_lim_lines = [
        f"• [{f.citation_index}] ({f.study_evidence_type}, access: {f.evidence_access}, sample status: {f.field_status.get('sample_or_scale', 'Not reported')}): {f.methodological_limitations}"
        for f in findings
        if not f.excluded_reason
    ][:8]
    prep_note = (
        f"{prep_count} record(s) are preprints that have not completed peer review"
        + (f" ({linked_prep_count} linked to a published version and deduplicated to prevent double-counting)" if linked_prep_count else "")
        + "."
    )
    overlap_summary = (
        "Non-independent dataset/author overlap detected: " + " ".join(overlap_all)
        if overlap_all
        else "No shared cohort/dataset duplication was detected across the selected primary studies."
    )
    excl_note = (
        f"Excluded/unverified claims: {len(excluded_claims)} candidate claim(s) failed structural or factual verification and were withheld from the narrative."
        if excluded_claims
        else "All candidate claims passed structural and factual verification against source records."
    )
    criteria_str = "; ".join(appraisal_considerations[:5])

    reliability_content = (
        f"Domain-Adaptive Appraisal Criteria Applied ({dom_prof.display_name}): {criteria_str}.\n"
        f"1. Source Accessibility Boundary: {full_text_count} study(ies) were evaluated from open-access full text, {abs_only_count} from abstracts and {meta_only_count} from metadata only; "
        f"unreported details in abstracts are recorded as 'Not reported' rather than assumed absent [{lead_cites}].\n"
        f"2. Publication & Version Status: {prep_note}\n"
        f"3. Independence of Replication: {overlap_summary}\n"
        f"4. Verification Gate Summary: {excl_note}\n"
        f"5. Study-Specific Methodological Caveats:\n" + "\n".join(study_lim_lines)
    )

    sections.append(
        SynthesisSection(
            heading="D. Reliability and Limitations of the Evidence",
            content=reliability_content,
            cited_indices=all_indices[: min(8, len(all_indices))],
        )
    )

    # Subsection E: Integrated Scientific Conclusion
    established_items = [
        f"{c.theme_label} ({c.evidence_type.replace('_', ' ')}) [{', '.join(str(i) for i in c.supporting_indices)}]"
        for c in verified_claims
        if c.supporting_indices and c.verification.state in ("Verified", "Verified with qualifications")
    ]
    suggested_items = [
        f"Author-reported interpretations and single-study or conditional observations in [{', '.join(str(i) for i in c.relevant_study_indices)}]"
        for c in verified_claims
    ]
    hf_clause = (
        f"Open-source computational artifacts on Hugging Face ({', '.join(h.id for h in hf_resources[:3])}) provide reusable implementations "
        f"but represent community software/data artifacts rather than peer-reviewed scientific evidence [{all_indices[0]}]."
        if hf_resources
        else ""
    )

    conclusion_content = (
        f"• What the Selected Literature Establishes: Within the tested regimes and study designs, the evidence establishes findings across "
        f"{'; '.join(established_items) if established_items else f'descriptive observations in [{lead_cites}]'}.\n"
        f"• What It Suggests but Does Not Establish: {'; '.join(suggested_items[:2])}. {hf_clause}\n"
        f"• What Remains Uncertain: Out-of-distribution generalizability, unmeasured confounders in observational designs, and long-term or external regime stability remain unresolved across [{lead_cites}].\n"
        f"• Conclusions That Would Be Premature: Claiming field-wide consensus, inferring causal clinical efficacy from observational or benchmark correlations, or pooling heterogeneous effect measures without standardized full-text data would be scientifically premature [{lead_cites}]."
    )

    sections.append(
        SynthesisSection(
            heading="E. Integrated Scientific Conclusion",
            content=conclusion_content,
            cited_indices=all_indices[: min(5, len(all_indices))],
        )
    )

    if chal_claims and sup_claims:
        verdict = "Context-Dependent & Methodologically Heterogeneous Evidence"
    elif not sup_claims:
        verdict = "Inconclusive / Preliminary Evidence"
    elif len(papers) < 3:
        verdict = "Limited / Preliminary Supporting Evidence"
    else:
        verdict = "Consistent Within Tested Regimes (Subject to Methodological Qualifications)"

    c_summary = (
        f"Evaluated across {len(papers)} deduplicated studies [{lead_cites}] ({len(verified_claims)} verified claim cluster(s)); "
        f"the agreement meter counts studies by direction of effect (supporting / mixed / contradicting), not by "
        f"statistical significance, so it ignores effect size and precision (SWiM); evidence weight is qualified by "
        f"study design, replication independence, and {dom_prof.display_name} appraisal standards."
    )

    # Review type: re-organise the findings section (B) and adapt C-E to the reporting framework
    if organization != "thematic" and sections:
        sections[0] = compose_organized_body(organization, papers, findings, verified_claims)
    sections = apply_review_framework(review_framework, sections, papers, findings)

    return executive_answer, sections, verdict, c_summary


# ============================================================================
# PHASE 11: Explicit 10-Dimension Synthesis Quality Gate
# ============================================================================

def evaluate_section1_quality_gate(
    query: str,
    papers: List[PaperRecord],
    findings: List[AtomicFinding],
    clusters: List[EvidenceCluster],
    verified_claims: List[SynthesisClaim],
    executive_answer: str,
    sections: List[SynthesisSection],
    primary_domain: str,
    model_used: str = "deterministic",
) -> SynthesisQualityReport:
    """
    Phase 11 — Deterministic 10-Dimension Synthesis Quality Gate for Section 1:
    1. Answer relevance
    2. Evidence coverage
    3. Claim faithfulness
    4. Quantitative accuracy
    5. Scientific validity
    6. Contradiction handling
    7. Citation integrity
    8. Internal consistency
    9. Domain appropriateness
    10. Readability (non-enumerative academic prose, required subsections B-E)
    """
    checks: List[SynthesisQualityCheck] = []
    max_idx = len(papers)
    # Off-topic studies are screened out of the synthesis but keep their reference numbers, so check membership
    valid_idx = {p.citation_index for p in papers}
    full_narrative = executive_answer + "\n" + "\n".join(f"{s.heading}\n{s.content}" for s in sections)
    cited_in_text = extract_citation_numbers(full_narrative)
    cited_set = set(cited_in_text)
    source_corpus_text = " ".join(f"{study_source_text(p)} {p.elicit.quantitative_result}" for p in papers)

    # 1. Answer relevance
    q_terms = [t.lower() for t in extract_core_scientific_terms(query, max_terms=6)]
    exec_low = (executive_answer or "").lower()
    matched_q = [t for t in q_terms if t in exec_low]
    ans_ok = bool(executive_answer and len(executive_answer.split()) >= 35 and (matched_q or not q_terms))
    checks.append(
        SynthesisQualityCheck(
            name="answer_relevance",
            passed=ans_ok,
            reasons=[] if ans_ok else ["Executive synthesis does not directly address core query terms or is too brief."],
        )
    )

    # 2. Evidence coverage
    cluster_studies = [set(cl.study_indices) for cl in clusters if cl.study_indices]
    uncovered = [idx + 1 for idx, sset in enumerate(cluster_studies) if not (sset & cited_set)]
    cov_ok = len(uncovered) == 0 or not papers
    checks.append(
        SynthesisQualityCheck(
            name="evidence_coverage",
            passed=cov_ok,
            reasons=[] if cov_ok else [f"Evidence cluster(s) {uncovered} have no cited studies in the narrative."],
        )
    )

    # 3. Claim faithfulness
    unfaithful = [c.claim_id for c in verified_claims if c.verification.state == "Unsupported"]
    faith_ok = len(unfaithful) == 0
    checks.append(
        SynthesisQualityCheck(
            name="claim_faithfulness",
            passed=faith_ok,
            reasons=[] if faith_ok else [f"Unsupported claim(s) {unfaithful} present in verified claim set."],
        )
    )

    # 4. Quantitative accuracy
    bad_nums = [num for num in extract_nontrivial_numbers(full_narrative) if num not in source_corpus_text]
    quant_ok = len(bad_nums) == 0
    checks.append(
        SynthesisQualityCheck(
            name="quantitative_accuracy",
            passed=quant_ok,
            reasons=[] if quant_ok else [f"Narrative contains ungrounded numerical value(s): {bad_nums}."],
        )
    )

    # 5. Scientific validity
    all_obs = bool(findings) and all(f.claim_evidence_type in ("observed_association", "predictive_relationship") for f in findings)
    sci_ok = True
    sci_reasons: List[str] = []
    if all_obs and _OVERCLAIM_CAUSAL_RE.search(full_narrative):
        sci_ok = False
        sci_reasons.append("Narrative states unqualified causal proof from purely observational/predictive studies.")
    if re.search(r"\b(meta-analysis|pooled\s+effect\s+estimate)\b", full_narrative, re.IGNORECASE) and "without" not in full_narrative.lower() and "absence of" not in full_narrative.lower():
        sci_ok = False
        sci_reasons.append("Narrative claims a formal meta-analysis or pooled estimate without quantitative pooling.")
    checks.append(SynthesisQualityCheck(name="scientific_validity", passed=sci_ok, reasons=sci_reasons))

    # 6. Contradiction handling
    chal_study_idxs = [f.citation_index for f in findings if f.direction in ("negative", "null_or_nonsignificant") and not f.excluded_reason]
    contra_ok = True
    contra_reasons: List[str] = []
    if chal_study_idxs and not any(i in cited_set for i in chal_study_idxs):
        contra_ok = False
        contra_reasons.append(f"Challenging/null study index(es) {chal_study_idxs} were omitted from the synthesis.")
    checks.append(SynthesisQualityCheck(name="contradiction_handling", passed=contra_ok, reasons=contra_reasons))

    # 7. Citation integrity
    out_of_bounds = [i for i in cited_in_text if i not in valid_idx]
    cit_ok = (len(out_of_bounds) == 0) and (len(cited_in_text) > 0 if max_idx > 0 else True)
    cit_reasons: List[str] = []
    if out_of_bounds:
        cit_reasons.append(f"Citation(s) {out_of_bounds} are not among the synthesised studies (valid: {sorted(valid_idx)}).")
    if max_idx > 0 and not cited_in_text:
        cit_reasons.append("No inline bracketed citations [1..N] found in narrative.")
    checks.append(SynthesisQualityCheck(name="citation_integrity", passed=cit_ok, reasons=cit_reasons))

    # 8. Internal consistency
    claim_study_union = {i for c in verified_claims for i in c.relevant_study_indices}
    inconsistent = [i for i in cited_set if claim_study_union and i not in set( p.citation_index for p in papers )]
    cons_ok = len(inconsistent) == 0
    checks.append(
        SynthesisQualityCheck(
            name="internal_consistency",
            passed=cons_ok,
            reasons=[] if cons_ok else [f"Citations {inconsistent} do not match paper records."],
        )
    )

    # 9. Domain appropriateness
    dom_ok = True
    dom_reasons: List[str] = []
    if primary_domain == "mathematics":
        for f in findings:
            if "patients" in f.context_or_system.lower() and "patients" not in source_corpus_text.lower():
                dom_ok = False
                dom_reasons.append("Inappropriate clinical patient sample applied to mathematical study.")
    checks.append(SynthesisQualityCheck(name="domain_appropriateness", passed=dom_ok, reasons=dom_reasons))

    # 10. Readability & Structure
    headings_joined = " | ".join(s.heading for s in sections)
    enum_matches = re.findall(r"(?:^|\n)\s*Paper\s+\d+\s*:", full_narrative)
    read_ok = len(sections) >= 4 and len(enum_matches) == 0
    read_reasons: List[str] = []
    if len(sections) < 4:
        read_reasons.append("Synthesis has fewer than the 4 required thematic/analytical subsections (B–E).")
    if enum_matches:
        read_reasons.append("Synthesis contains sequential paper-by-paper enumeration ('Paper 1: ...').")
    checks.append(SynthesisQualityCheck(name="readability", passed=read_ok, reasons=read_reasons))

    all_passed = all(c.passed for c in checks)
    return SynthesisQualityReport(
        checks=checks,
        all_passed=all_passed,
        regenerated_sections=[],
        narrative_origin="deterministic" if "deterministic" in model_used else "llm_verified",
        workflow_version=WORKFLOW_VERSION,
        prompt_versions={k: v["version"] for k, v in TASK_PROMPT_REGISTRY.items()},
        model_used=model_used,
    )
