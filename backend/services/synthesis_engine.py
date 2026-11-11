import asyncio
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable, Awaitable
from models import (
    PaperRecord,
    HFResourceRecord,
    ApiKeysConfig,
    BibliographyEntry,
    ConsensusBreakdown,
    SynthesisSection,
    ElicitExtraction,
    GapOpportunity,
    KnowledgeGapMatrix,
    StudyEntity,
    ResultEntity,
    ClaimEntity,
    GapEntity,
    RelationshipEntity,
    SharedEvidenceBase,
    LiteratureGraph,
    QueryDecompositionPlan,
    GapCategoryType,
    TaskExecutionRecord,
    TaskCStudyBatchOutput,
    TaskDEFGClaimsOutput,
    TaskHIGapOutput,
    TaskJNarrativeSynthesisOutput,
    TaskJSectionOutput,
    SynthesisEvidencePackage,
    StructuredEvidenceMatrix,
)
from services.domain_profiles import DOMAIN_PROFILES, compose_profiles
from services.gap_matrix_engine import (
    build_universal_knowledge_gap_matrix,
    clear_gap_verification_cache,
)
from services.llm_service import (
    call_llm,
    extract_json_block,
    resolve_model_capabilities,
    execute_validated_llm_task,
)
from services.query_utils import build_concise_search_phrase, extract_core_scientific_terms
from services.verification_gates import (
    _index_label,
    extract_citation_numbers,
    normalize_citations,
    study_source_text,
    drop_unsupported_numbers,
    repair_study_extraction,
    strip_citations_not_in,
    verify_study_extraction_against_source,
    verify_claims_against_sources,
    extract_nontrivial_numbers,
)
from frameworks.appraisal_routes import detect_biomedical_design
from frameworks.review_types import resolve_review_type
from frameworks.rules import render_rules_for_prompt, select_rules
from services.extraction_patterns import study_sample_size
from services.review_composition import finalize_llm_sections, organized_groups
from services.claim_synthesis import (
    WORKFLOW_VERSION,
    get_domain_appraisal_considerations,
    extract_atomic_findings,
    cluster_comparable_findings,
    build_and_verify_claims,
    determine_question_answerability,
    compose_section1_narrative_from_verified_claims,
    evaluate_section1_quality_gate,
    get_last_synthesis_package,
    set_last_synthesis_package,
)


# Section 12: Source-level study extraction cache & validated claim cache
# Keyed by study, question, domain and model: stance and relevance depend on the question asked
_STUDY_EXTRACTION_CACHE: Dict[str, Tuple[ElicitExtraction, str, str, str]] = {}
# Upper bound on concurrent Task C extraction calls per search
# Study extraction runs as many small calls side by side: a call that returns 4 studies takes a third of the time
# of one that returns 12, and a failed call costs 4 studies' worth of work, not 12. A local model runs one call at
# a time on one GPU, so it gets slightly larger batches (fewer repeated instructions to read).
TASK_C_MAX_CONCURRENT_BATCHES = 5
TASK_C_STUDIES_PER_CALL = {"hosted": 4, "local": 6}
# Wall-clock budgets (seconds) for one step on a hosted model, retries included. When a budget runs out only that
# piece falls back to rule-based output. Local models are not time-limited: they are slow but keep writing.
STEP_BUDGET_S = {"extraction": 90.0, "claims": 100.0, "gaps": 110.0, "answer": 100.0, "section": 80.0}


def step_budget(config: ApiKeysConfig, step: str) -> Optional[float]:
    return None if config.llm_provider in ("ollama", "custom") else STEP_BUDGET_S[step]
# Token budget for the methodology rules block in synthesis prompts (keeps small-model prompts small)
PROMPT_RULES_TOKEN_BUDGET = 250
# Below this many on-topic studies the synthesis keeps all retrieved studies (flagged) rather than too few
MIN_SYNTHESIS_PAPERS = 3
# Quality checks whose failure means the model's text cannot be trusted (wrong citations, numbers not in the
# sources, causal overclaims); other checks only add notes to a kept answer
HARD_QUALITY_CHECKS = {"citation_integrity", "quantitative_accuracy", "scientific_validity"}
# Deep Research reviews: minimum length (a review, not a summary) and the answer budget that allows it.
# ≈1.35 tokens per word plus JSON overhead; the budget is capped by each model's output and context limits.
DEEP_REVIEW_MIN_WORDS = 2500
DEEP_SECTION_MIN_WORDS = 350
DEEP_LENGTH_TOLERANCE = 0.85
SHORT_TEXT_KEPT_NOTE = "Text kept at shorter than the target length: all other checks passed."


def _deep_section_target_words(study_count: int) -> int:
    """Target length for a Deep section: a section resting on one or two studies cannot honestly fill the full length."""
    return min(DEEP_SECTION_MIN_WORDS, 150 + 75 * max(study_count, 1))


def _keep_if_only_short(rec: TaskExecutionRecord) -> bool:
    """
    True (and marks the record as a success) when the model's last valid reply failed only the length check: its
    own text is still better than the rule-based fallback, so this is not a fallback at all.
    """
    if not (rec.last_candidate and rec.validation_errors):
        return False
    problems = rec.validation_errors[-1].split(": ", 1)[-1].split("; ")
    if not all(e.startswith(("The section is", "The executive answer is")) for e in problems):
        return False
    rec.degraded_to_deterministic = False
    rec.validation_errors.append(SHORT_TEXT_KEPT_NOTE)
    return True
DEEP_REVIEW_MAX_TOKENS = 12000
# Deep Research executive answer, written alongside the separately written sections
DEEP_EXEC_MIN_WORDS = 250
DEEP_EXEC_MAX_TOKENS = 1600
# Standard synthesis, written as separate short calls: target words per piece
STANDARD_EXEC_WORDS = 150
STANDARD_BODY_WORDS = 230
STANDARD_SECTION_WORDS = 140
STANDARD_EXEC_MAX_TOKENS = 900
# Answer tokens budgeted per study (9 short fields) plus fixed JSON overhead; measured ≈150-200 per study
TASK_C_TOKENS_PER_STUDY = 220
TASK_C_OUTPUT_OVERHEAD = 200
# Abstract characters sent per study, chosen by _key_result_sentences rather than a blind cut
TASK_C_ABSTRACT_CHARS = 480
# Characters of open-access full text sent per study, by section (≈300 tokens in all; keeps small models in budget)
FULL_TEXT_EXCERPT_CHARS = {"methods": 380, "results": 520, "limitations": 260}

_RESULT_SENTENCE_CUES = re.compile(
    r"\d|\b(result|found|show|demonstrat|improv|reduc|increas|outperform|achiev|significant|associat|"
    r"conclud|suggest|compared|accuracy|effect)",
    re.IGNORECASE,
)


def _full_text_excerpt(p: PaperRecord) -> str:
    """Key sentences from the open-access full text's methods, results and limitations, within a fixed budget."""
    if not p.full_text_sections:
        return ""
    parts = []
    for section, budget in FULL_TEXT_EXCERPT_CHARS.items():
        text = p.full_text_sections.get(section) or ""
        if section == "results" and not text:
            text = p.full_text_sections.get("body", "")  # papers without a Results heading (common outside medicine)
        excerpt = _key_result_sentences(text, budget)
        if excerpt:
            parts.append(f"{section.capitalize()}: {excerpt}")
    return f"\n    Full text ({p.full_text_source}) - " + " | ".join(parts) if parts else ""


def _key_result_sentences(text: Optional[str], max_chars: int) -> str:
    """
    Compresses an abstract to fit `max_chars` while keeping its most informative sentences: the opening
    (context/aim) plus sentences carrying results or numbers, in original order. Cheaper and more reliable
    than truncation, which often cuts off the result sentence at the end of an abstract.
    """
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return text[:max_chars]
    ranked = sorted(
        range(1, len(sentences)),
        key=lambda i: (-len(_RESULT_SENTENCE_CUES.findall(sentences[i])), i),
    )
    keep = {0}
    used = len(sentences[0])
    for i in ranked:
        if used + len(sentences[i]) + 1 <= max_chars:
            keep.add(i)
            used += len(sentences[i]) + 1
    return " ".join(sentences[i] for i in sorted(keep))[:max_chars]


@dataclass
class _SynthesisScope:
    validated_claims: List[ClaimEntity] = field(default_factory=list)
    task_audit_trail: List[TaskExecutionRecord] = field(default_factory=list)


# Per-search claim & audit state. Each /api/search request begins its own scope so concurrent searches never
# share results; asyncio tasks spawned within a search inherit (and mutate) the same scope object.
_SEARCH_SCOPE: ContextVar[Optional[_SynthesisScope]] = ContextVar("synthesis_search_scope", default=None)
# Fallback for direct callers (tests, scripts) that never begin a scope
_DEFAULT_SCOPE = _SynthesisScope()


def begin_synthesis_scope() -> None:
    _SEARCH_SCOPE.set(_SynthesisScope())


def _scope() -> _SynthesisScope:
    return _SEARCH_SCOPE.get() or _DEFAULT_SCOPE


def _scoped_claims() -> List[ClaimEntity]:
    return _scope().validated_claims


def _scoped_audit_trail() -> List[TaskExecutionRecord]:
    return _scope().task_audit_trail


def clear_synthesis_caches() -> None:
    _STUDY_EXTRACTION_CACHE.clear()
    _scoped_claims().clear()
    _scoped_audit_trail().clear()
    clear_gap_verification_cache()
    set_last_synthesis_package(None)


def get_last_task_audit_trail() -> List[TaskExecutionRecord]:
    return list(_scoped_audit_trail())


def generate_bibliography(papers: List[PaperRecord]) -> List[BibliographyEntry]:
    """
    Phase 9 — Bibliographic Identity & Integrity:
    Generates complete, formatted APA, IEEE, and BibTeX bibliography entries
    for every paper [1]..[N] using ONLY retrieved bibliographic metadata (never LLM guesses),
    preserving canonical identifiers, evidence access level, preprint/version linkage,
    and retraction/correction screening status.
    """
    entries: List[BibliographyEntry] = []
    for p in papers:
        idx = p.citation_index
        if not p.authors:
            authors_str = "Anonymous"
            first_author_key = "anon"
        elif len(p.authors) == 1:
            authors_str = p.authors[0]
            first_author_key = re.sub(r"[^a-zA-Z]", "", p.authors[0].split()[-1]).lower() or "author"
        elif len(p.authors) <= 4:
            authors_str = ", ".join(p.authors[:-1]) + f", & {p.authors[-1]}"
            first_author_key = re.sub(r"[^a-zA-Z]", "", p.authors[0].split()[-1]).lower() or "author"
        else:
            authors_str = ", ".join(p.authors[:3]) + ", et al."
            first_author_key = re.sub(r"[^a-zA-Z]", "", p.authors[0].split()[-1]).lower() or "author"

        year_str = str(p.year) if p.year else "Year not reported"
        venue = p.venue or p.source or "Source venue not reported"

        id_parts = []
        canonical_id = p.id
        if p.doi:
            id_parts.append(f"DOI: {p.doi}")
            canonical_id = f"doi:{p.doi}"
        if p.pmid:
            id_parts.append(f"PMID: {p.pmid}")
            if canonical_id == p.id:
                canonical_id = f"pmid:{p.pmid}"
        if p.arxiv_id:
            id_parts.append(f"arXiv:{p.arxiv_id}")
            if canonical_id == p.id:
                canonical_id = f"arxiv:{p.arxiv_id}"
        if p.s2_id:
            id_parts.append(f"S2:{p.s2_id[:12]}")
        identifier_str = " | ".join(id_parts) if id_parts else f"Source: {p.source}"

        url = p.url or (f"https://doi.org/{p.doi}" if p.doi else "")

        has_abs = bool((p.abstract or "").strip() and len((p.abstract or "").strip()) >= 30)
        ev_access = (f"Full text ({p.full_text_source})" if p.full_text_source
                     else "Abstract" if has_abs else "Metadata only")

        if p.version_status == "preprint_with_published_link":
            pub_type = "Peer-reviewed publication (linked preprint deduplicated)"
            ver_note = f"Merged preprint and peer-reviewed records across {', '.join(p.all_sources or [p.source])} as a single study."
        elif p.version_status == "preprint":
            pub_type = "Preprint (not peer-reviewed)"
            ver_note = "Preprint version; verify against subsequent peer-reviewed publication if available."
        else:
            pub_type = "Peer-reviewed / archival publication"
            ver_note = ""

        full_t = f"{p.title} {p.abstract or ''}"
        if re.search(r"\b(retracted|retraction\s+notice|withdrawn|expression\s+of\s+concern)\b", full_t, re.IGNORECASE):
            retract_note = "Flagged: Retraction or expression-of-concern notice detected in source metadata/abstract"
        else:
            retract_note = "Screened title/abstract (no retraction notice detected; external Crossref/RetractionWatch feed not queried)"

        apa = f"[{idx}] {authors_str} ({year_str}). {p.title}. {venue}. [{pub_type}; Access: {ev_access}]. {identifier_str}. {url}".strip()
        ieee = f"[{idx}] {authors_str}, \"{p.title},\" {venue}, {year_str}. [{identifier_str}; {pub_type}]. Available: {url}".strip()

        bib_key = f"{first_author_key}{p.year or 'nd'}_{idx}"
        bib_authors = " and ".join(p.authors[:6]) if p.authors else "Anonymous"
        bibtex_lines = [
            f"@article{{{bib_key},",
            f"  title = {{{p.title}}},",
            f"  author = {{{bib_authors}}},",
            f"  year = {{{p.year or 'n.d.'}}},",
            f"  journal = {{{venue}}},",
            f"  note = {{{pub_type}; Evidence access: {ev_access}}},",
        ]
        if p.doi:
            bibtex_lines.append(f"  doi = {{{p.doi}}},")
        if p.url:
            bibtex_lines.append(f"  url = {{{p.url}}}")
        bibtex_lines.append("}")
        bibtex = "\n".join(bibtex_lines)

        entries.append(
            BibliographyEntry(
                index=idx,
                paper_id=p.id,
                title=p.title,
                authors_str=authors_str,
                year_str=year_str,
                venue=venue,
                source=", ".join(p.all_sources) if p.all_sources else p.source,
                identifier_str=identifier_str,
                url=url,
                apa_formatted=apa,
                ieee_formatted=ieee,
                bibtex=bibtex,
                stable_key=p.canonical_study_id or bib_key,
                canonical_identifier=canonical_id,
                evidence_access=ev_access,
                publication_type=pub_type,
                linked_version_note=ver_note,
                retraction_check=retract_note,
                cited_in_synthesis=False,
            )
        )
    return entries


def mark_cited_bibliography_entries(
    bibliography: List[BibliographyEntry],
    executive_answer: str,
    sections: List[SynthesisSection],
) -> None:
    full_text = (executive_answer or "") + "\n" + "\n".join(s.content for s in sections)
    cited_set = set(extract_citation_numbers(full_text))
    for s in sections:
        cited_set.update(s.cited_indices or [])
    for b in bibliography:
        b.cited_in_synthesis = b.index in cited_set


async def build_standard_search_queries(query: str, config: ApiKeysConfig) -> List[str]:
    concise = build_concise_search_phrase(query, max_terms=4)
    core_terms = extract_core_scientific_terms(query, max_terms=6)
    queries = [concise]
    if len(core_terms) >= 3:
        broader = " ".join(core_terms[:3])
        if broader != concise:
            queries.append(broader)
    return queries[:2]


async def decompose_query_for_deep_research(query: str, config: ApiKeysConfig) -> List[str]:
    clean_q = build_concise_search_phrase(query, max_terms=3)
    return [
        clean_q,
        f"{clean_q} benchmark validation",
        f"{clean_q} limitations generalization",
        f"{clean_q} novel methods",
    ]


def _extract_domain_adaptive_study_fields(
    p: PaperRecord,
    hf_resources: List[HFResourceRecord],
    primary_domain: str = "biomedical_clinical",
    sample_policy: str = "optional",
) -> None:
    """
    Phase 7 & 8 Section 2: Domain-adaptive, provenance-aware structured extraction.
    - Distinguishes `abstract_derived` vs `metadata_only`.
    - Distinguishes `Reported`, `Not reported`, and `Not applicable` (e.g., for mathematical proofs).
    - Preserves original numerical measurements and units when present in the abstract; never invents them.
    """
    has_abstract = bool((p.abstract or "").strip() and len((p.abstract or "").strip()) >= 30)
    provenance = "abstract_derived" if has_abstract else "metadata_only"
    text = f"{p.title}. {p.abstract or ''}"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p.abstract or "") if len(s.strip()) > 20]

    # 1. Domain-aware Methodology & Study Design detection
    method_keywords = [
        ("Formal Mathematical Proof / Analysis", r"\b(theorem|proof|lemma|conjecture|minimax|banach|hilbert|cohomology|ergodic|pde|ode)\b"),
        ("Quantum / Physical Simulation", r"\b(quantum|hamiltonian|monte carlo|tensor network|dft|density functional|molecular dynamics|lattice|navier-stokes)\b"),
        ("Transformer / Foundation Model", r"\b(transformer|bert|gpt|llm|language model|esm|attention|pretrained|foundation model)\b"),
        ("Graph / Geometric Neural Network", r"\b(graph neural|gnn|message passing|equivariant|egnn|gcn)\b"),
        ("Diffusion / Generative Architecture", r"\b(diffusion|generative|vae|gan|flow matching|autoregressive|de novo)\b"),
        ("Randomized / Clinical Trial", r"\b(randomized|clinical trial|phase i|phase ii|phase iii|placebo|double-blind)\b"),
        ("Observational Cohort / Epidemiological", r"\b(cohort|retrospective|prospective|observational|case-control|hazard ratio|odds ratio)\b"),
        ("Experimental Wet-Lab / Chemical Assay", r"\b(in vitro|in vivo|assay|sequencing|crispr|pcr|synthesis|catalysis|spectroscopy|mic|ic50)\b"),
        ("Systematic Review / Meta-Analysis", r"\b(systematic review|meta-analysis|survey|overview|benchmark)\b"),
        ("Statistical / Ensemble ML", r"\b(random forest|xgboost|regression|bayesian|clustering|support vector)\b"),
    ]
    detected_methods = []
    for label, pat in method_keywords:
        if re.search(pat, text, flags=re.IGNORECASE):
            detected_methods.append(label)
    methodology = ", ".join(detected_methods[:2]) if detected_methods else f"{p.cluster_label} Study"
    study_design = detected_methods[0] if detected_methods else "Empirical / Analytical Study"

    # 2. Sample size / Scale / Regime extraction with explicit ReportingStatusType
    is_math_or_theory = (
        primary_domain == "mathematics"
        or sample_policy == "not_applicable"
        or "Formal Mathematical Proof" in study_design
    )
    found_sample = study_sample_size(p)
    if is_math_or_theory and not found_sample:
        sample_or_scale = "Not applicable (theoretical/mathematical formulation)"
        sample_status = "Not applicable"
    elif found_sample:
        sample_or_scale = found_sample
        sample_status = "Reported"
    elif not has_abstract and not p.full_text_sections:
        sample_or_scale = "Unavailable (metadata only)"
        sample_status = "Unavailable"
    else:
        sample_or_scale = "Not reported in full text" if p.full_text_sections else "Not reported in abstract"
        sample_status = "Not reported"

    # 3. Dataset / System / Mathematical Space detection
    if primary_domain == "mathematics":
        space_match = re.search(
            r"\b(hilbert|banach|sobolev|riemannian|manifold|convex|non-convex|stochastic|euclidean|operator|lattice|graph)[^.;,]{0,45}",
            text,
            flags=re.IGNORECASE,
        )
        dataset_base = space_match.group(0).strip() if space_match else "Formal mathematical setting"
    else:
        dataset_match = re.search(
            r"\b(benchmark|dataset|cohort|database|patients|samples|proteins|molecules|sequences|cells|pdb|uniprot|chembl|imagenet|pubmed|quantum|catalyst|electrolyte)\b[^.;,]{0,55}",
            text,
            flags=re.IGNORECASE,
        )
        dataset_base = dataset_match.group(0).strip() if dataset_match else f"{p.venue or p.source} study system"

    if sample_status == "Reported" and sample_or_scale not in dataset_base:
        dataset_str = f"{dataset_base} ({sample_or_scale})"
    elif sample_status == "Not applicable":
        dataset_str = f"{dataset_base} [Sample size: N/A]"
    else:
        dataset_str = dataset_base

    # 4. Quantitative Result & Key Findings (preserving exact numbers/units from abstract)
    quant_match = re.search(
        r"([^.;]*(?:\b(?:auroc|auc|accuracy|f1|rmsd|mae|rmse|ic50|mic|hr|or|rr|p\s*[<=>])\b[^.;]*|\b\d+(?:\.\d+)?\s*(?:%|kcal/mol|mev|nm|μm|ug/ml|mm|ms|fold)\b[^.;]*))",
        p.abstract or "",
        flags=re.IGNORECASE,
    )
    quant_str = quant_match.group(0).strip()[:160] if quant_match else ("Not reported in abstract" if has_abstract else "Unavailable (metadata only)")

    finding_sent = ""
    for s in sentences:
        if re.search(r"\b(show|demonstrate|achieve|outperform|improve|reveal|find|indicate|prove|establish|accuracy|auc|theorem|bound)\b", s, re.IGNORECASE):
            finding_sent = s
            break
    if not finding_sent and sentences:
        finding_sent = sentences[-1]
    if not finding_sent:
        finding_sent = f"[Metadata-only record] {p.title}"

    # 5. Limitations & Provenance note
    limit_sent = ""
    for s in sentences:
        if re.search(r"\b(however|limit|challenge|remain|future|despite|bottleneck|lack|require|assumption|restrict)\b", s, re.IGNORECASE):
            limit_sent = s
            break
    if not limit_sent:
        if not has_abstract:
            limit_sent = "Abstract unavailable from source metadata; full-text inspection required to assess methodological limitations."
        elif is_math_or_theory:
            limit_sent = "Subject to stated regularity, dimensionality, or asymptotic assumptions in the formal derivation."
        else:
            limit_sent = "Specific sample-size power or out-of-distribution generalization limits not detailed in abstract."

    # 6. Code or HF link
    code_link = ""
    if p.arxiv_id:
        for hf in hf_resources:
            if p.arxiv_id in hf.linked_arxiv_ids:
                code_link = f"HF: {hf.id}"
                break
    if not code_link:
        gh_match = re.search(r"https?://github\.com/[^\s)\]]+", p.abstract or "")
        if gh_match:
            code_link = gh_match.group(0)
        elif p.arxiv_id:
            code_link = f"arXiv:{p.arxiv_id}"
        elif p.doi:
            code_link = f"DOI:{p.doi}"
        else:
            code_link = p.source

    p.elicit = ElicitExtraction(
        methodology=methodology[:140],
        dataset_or_system=dataset_str[:150],
        key_findings=finding_sent[:250],
        limitations=limit_sent[:220],
        code_or_hf_link=code_link[:120],
        study_design=study_design[:100],
        sample_or_scale=sample_or_scale[:100],
        sample_reporting_status=sample_status,
        quantitative_result=quant_str,
        provenance_type=provenance,
        evidence_UNCERTAINTY_note=f"Extracted from {provenance.replace('_', ' ')}.",
    )

    # Stance classification
    if re.search(r"\b(contradict|fail|ineffective|no significant|refute|overestimate|negative|counterexample|suboptimal)\b", text, re.IGNORECASE):
        p.stance = "contradicting"
        p.stance_reason = "Reports negative/null findings, counterexamples, or challenges prevailing assumptions."
    elif re.search(r"\b(however|mixed|trade-off|conditional|context-dependent|partial)\b", text, re.IGNORECASE):
        p.stance = "mixed"
        p.stance_reason = "Reports conditional efficacy or regime-dependent trade-offs."
    else:
        p.stance = "supporting"
        p.stance_reason = "Provides empirical, computational, or theoretical evidence addressing the query."
    if p.relevance == "off_topic":
        p.stance = "neutral"
        p.stance_reason = f"Not rated: study does not address the question ({p.relevance_note})."


def _ensure_inline_citations(text: str, max_idx: int) -> str:
    if not text or max_idx < 1:
        return text
    if not re.search(r"\[\d+(?:,\s*\d+)*\]", text):
        ref_tag = f"[1, {min(2, max_idx)}]" if max_idx > 1 else "[1]"
        return f"{text.rstrip()} {ref_tag}"
    return text


async def _run_task_c_batched_study_extraction(
    query: str,
    papers: List[PaperRecord],
    hf_resources: List[HFResourceRecord],
    prim_domain: str,
    sample_policy: str,
    config: ApiKeysConfig,
    audit_trail: List[TaskExecutionRecord],
    descriptive_question: bool = False,
) -> None:
    """
    Section 4 & 9 — Task C: Extract factual findings from studies in context-window-sized batches.
    - First computes deterministic source-grounded extraction (`abstract_derived` / `metadata_only`) on all papers.
    - Uses `_STUDY_EXTRACTION_CACHE` so unchanged studies are not re-extracted across runs.
    - Executes LLM extraction in batches of `cap.recommended_study_batch_size`, validating every item
      with `verify_study_extraction_against_source` (numerical integrity & domain sample rules).
    """
    cap = resolve_model_capabilities(config)
    # Batch size is bounded by the context window (recommended size) and by how many study answers fit the output limit
    per_call = TASK_C_STUDIES_PER_CALL["local" if config.llm_provider in ("ollama", "custom") else "hosted"]
    batch_size = max(
        1,
        min(per_call, cap.recommended_study_batch_size,
            (cap.max_output_tokens - TASK_C_OUTPUT_OVERHEAD) // TASK_C_TOKENS_PER_STUDY),
    )

    uncached_papers: List[PaperRecord] = []
    for p in papers:
        _extract_domain_adaptive_study_fields(p, hf_resources, primary_domain=prim_domain, sample_policy=sample_policy)
        ckey = f"{p.canonical_study_id or p.id}:{query.strip().lower()}:{prim_domain}:{cap.provider}:{cap.model_id}"
        if ckey in _STUDY_EXTRACTION_CACHE:
            cached_elicit, cached_stance, cached_relevance, cached_note = _STUDY_EXTRACTION_CACHE[ckey]
            if p.relevance != "off_topic":
                p.relevance, p.relevance_note = cached_relevance, cached_note
            p.elicit = cached_elicit.model_copy(deep=True)
            p.stance = cached_stance
        else:
            uncached_papers.append(p)

    if not uncached_papers:
        audit_trail.append(
            TaskExecutionRecord(
                task_id="Task_C",
                task_name="Task C: Batched Study & Result Extraction (Cache Hit)",
                model_used=f"{cap.provider}:{cap.model_id}",
                attempts_used=0,
                validation_passed=True,
                verification_status="passed_validation",
                degraded_to_deterministic=False,
            )
        )
        return

    async def _extract_batch(batch: List[PaperRecord]) -> List[TaskExecutionRecord]:
        batch_map = {p.citation_index: p for p in batch}
        batch_lines = []
        for p in batch:
            abs_text = _key_result_sentences(p.abstract, TASK_C_ABSTRACT_CHARS) or "Metadata only (no abstract provided by source)"
            batch_lines.append(
                f"[{p.citation_index}] Title: {p.title} ({p.year or 'n.d.'}, {p.source}, Status: {p.version_status})\n"
                f"    Deterministic Baseline Design: {p.elicit.study_design} | Sample/Scale: {p.elicit.sample_or_scale} ({p.elicit.sample_reporting_status})\n"
                f"    Abstract: {abs_text}"
                + _full_text_excerpt(p)
            )

        sys_prompt = (
            "You are executing Task C (Factual Study & Result Extraction) of a quality-gated scientific research engine.\n"
            f"Active Domain: {prim_domain} | Sample Policy: {sample_policy}\n"
            "STRICT SOURCE-FACT RULES:\n"
            "1. Extract study design, methodology, dataset/system, sample/scale, sample_reporting_status, quantitative_result, key_findings, and limitations ONLY from the provided study records.\n"
            "2. NEVER invent decimal numbers, percentages, p-values, or sample sizes not present in the study's Title or Abstract.\n"
            "3. Set sample_reporting_status to 'Reported', 'Not reported', 'Not applicable' (for pure mathematical proofs/theory), or 'Unavailable' (if metadata-only).\n"
            "4. Set addresses_question: 'yes' if the study is about the query's topic AND any place or population it names; "
            "'partly' if it is related but only indirectly (e.g. a neighbouring population or a broader topic); 'no' if it is "
            "about a different topic, place or population. Give a short relevance_reason.\n"
            + (
                "5. This is a descriptive question (what is known / the state of a topic), so there is no hypothesis to support "
                "or contradict: use stance 'supporting' when the study provides relevant data, 'contradicting' only when its data "
                "conflict with what other studies report on the same measure, and 'neutral' otherwise.\n"
                if descriptive_question else
                "5. stance is relative to the query: 'supporting' if the findings support it, 'contradicting' if they go "
                "against it, 'mixed' if conditional, 'neutral' if the study does not take a position.\n"
            )
            + "Return ONLY valid JSON matching:\n"
            '{"study_extractions": [{"citation_index": 1, "addresses_question": "yes|partly|no", "relevance_reason": "...", "stance": "supporting|mixed|contradicting|neutral", "study_design": "...", "methodology": "...", "dataset_or_system": "...", "sample_or_scale": "...", "sample_reporting_status": "Reported|Not reported|Not applicable|Unavailable", "quantitative_result": "...", "key_findings": "...", "limitations": "..."}]}'
        )
        user_prompt = f"Query: {query}\nExtract factual study fields for batch [{batch[0].citation_index}..{batch[-1].citation_index}]:\n" + "\n".join(batch_lines)

        def _validate_batch(out_obj: TaskCStudyBatchOutput) -> List[str]:
            errs: List[str] = []
            if not out_obj.study_extractions:
                return ["study_extractions list cannot be empty."]
            for item in out_obj.study_extractions:
                if item.citation_index not in batch_map:
                    errs.append(f"citation_index {item.citation_index} is not in the requested batch {list(batch_map.keys())}.")
                    continue
                errs.extend(verify_study_extraction_against_source(item, batch_map[item.citation_index], prim_domain))
            return errs

        def _repair_batch(out_obj: TaskCStudyBatchOutput, _errors: List[str]) -> Optional[TaskCStudyBatchOutput]:
            # Remove the unsupported figure (or reset the unsupported status) for the affected study only
            fixed = out_obj.model_copy(deep=True)
            fixed.study_extractions = [it for it in fixed.study_extractions if it.citation_index in batch_map]
            for it in fixed.study_extractions:
                repair_study_extraction(it, batch_map[it.citation_index], prim_domain)
            return fixed if fixed.study_extractions else None

        validated_batch, rec = await execute_validated_llm_task(
            task_id="Task_C",
            task_name=f"Task C: Study Extraction Batch [{batch[0].citation_index}..{batch[-1].citation_index}]",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            response_model=TaskCStudyBatchOutput,
            config=config,
            custom_validator=_validate_batch,
            max_retries=1,
            temperature=0.1,
            max_tokens=TASK_C_TOKENS_PER_STUDY * len(batch) + TASK_C_OUTPUT_OVERHEAD,
            grow_on_truncation=False,  # a cut-off batch is split in half below instead
            deadline_s=step_budget(config, "extraction"),
            repair=_repair_batch,
        )
        if rec.output_truncated and len(batch) > 1:
            # The answer did not fit the model's output limit: split the batch rather than resend the same request
            half = len(batch) // 2
            halves = await asyncio.gather(_extract_batch(batch[:half]), _extract_batch(batch[half:]))
            return halves[0] + halves[1]

        if validated_batch and not rec.degraded_to_deterministic:
            for item in validated_batch.study_extractions:
                if item.citation_index in batch_map:
                    tp = batch_map[item.citation_index]
                    if item.stance in ("supporting", "mixed", "contradicting", "neutral"):
                        tp.stance = item.stance
                    # Constraint mismatches found during ranking stay off-topic; otherwise take the model's screening
                    if tp.relevance != "off_topic":
                        tp.relevance = {"yes": "addresses", "partly": "partly", "no": "off_topic"}[item.addresses_question]
                        tp.relevance_note = (item.relevance_reason or "")[:200]
                    if tp.relevance == "off_topic":
                        tp.stance = "neutral"
                    if item.study_design:
                        tp.elicit.study_design = item.study_design[:100]
                    if item.methodology:
                        tp.elicit.methodology = item.methodology[:140]
                    if item.dataset_or_system:
                        tp.elicit.dataset_or_system = item.dataset_or_system[:150]
                    if item.sample_or_scale:
                        tp.elicit.sample_or_scale = item.sample_or_scale[:100]
                    tp.elicit.sample_reporting_status = item.sample_reporting_status
                    if item.quantitative_result:
                        tp.elicit.quantitative_result = item.quantitative_result[:160]
                    if item.key_findings:
                        tp.elicit.key_findings = item.key_findings[:250]
                    if item.limitations:
                        tp.elicit.limitations = item.limitations[:220]

        for p in batch:
            ckey = f"{p.canonical_study_id or p.id}:{query.strip().lower()}:{prim_domain}:{cap.provider}:{cap.model_id}"
            _STUDY_EXTRACTION_CACHE[ckey] = (p.elicit.model_copy(deep=True), p.stance, p.relevance, p.relevance_note)
        return [rec]

    # Batches cover disjoint papers, so run them concurrently (bounded to stay under free-tier rate limits)
    batches = [uncached_papers[i : i + batch_size] for i in range(0, len(uncached_papers), batch_size)]
    semaphore = asyncio.Semaphore(TASK_C_MAX_CONCURRENT_BATCHES)

    async def _bounded(batch: List[PaperRecord]) -> List[TaskExecutionRecord]:
        async with semaphore:
            return await _extract_batch(batch)

    for batch_records in await asyncio.gather(*[_bounded(b) for b in batches]):
        audit_trail.extend(batch_records)


async def _run_tasks_defg_claim_formulation(
    query: str,
    papers: List[PaperRecord],
    prim_domain: str,
    decomposition_plan: Optional[QueryDecompositionPlan],
    config: ApiKeysConfig,
    audit_trail: List[TaskExecutionRecord],
) -> List[ClaimEntity]:
    """
    Section 4 — Bounded Reasoning Tasks D, E, F, G:
    - Task D: Compare results across studies
    - Task E: Formulate scientific claims
    - Task F: Identify supporting and opposing evidence
    - Task G: Evaluate claim limitations
    """
    study_digest = "\n".join(
        f"[{p.citation_index}] ({p.elicit.study_design}, Stance: {p.stance}, Provenance: {p.elicit.provenance_type}): "
        f"Finding: {p.elicit.key_findings} | Quant: {p.elicit.quantitative_result} | System: {p.elicit.dataset_or_system} | Limits: {p.elicit.limitations}"
        for p in papers[:40]
    )
    sys_prompt = (
        "You are executing Tasks D, E, F, and G (Cross-Study Comparison, Scientific Claim Formulation, Supporting/Opposing Evidence Weighing, and Limitation Appraisal).\n"
        f"Active Domain: {prim_domain}\n"
        "STRICT SCIENTIFIC RULES:\n"
        "1. Group comparable study results and formulate 2 to 5 structured scientific claims.\n"
        "2. Every claim MUST list valid supporting_indices and (where applicable) opposing_indices from [1..N].\n"
        "3. INFERENCE VALIDITY: Never assign claim_category='causal_intervention' or claim definitive causation ('proves that X causes Y') if supporting studies are purely Observational Cohort / Epidemiological.\n"
        "4. NUMERICAL INTEGRITY: Never include decimal numbers or percentages in a claim proposition unless they appear verbatim in the cited studies.\n"
        "Return ONLY valid JSON matching:\n"
        '{"comparable_groups_summary": "...", "claims": [{"claim_id": "claim:1", "proposition": "...", "claim_category": "association|causal_intervention|predictive_performance|mechanistic|theoretical|methodological_comparison", "supporting_indices": [1, 2], "opposing_indices": [], "applicability_scope": "...", "methodological_caveats": "..."}]}'
    )
    user_prompt = (
        f"Query: {query}\nValidated Study Extractions (cite only these study numbers: "
        f"{_index_label(sorted(p.citation_index for p in papers))}):\n{study_digest}"
    )

    def _validate_claims_out(out_obj: TaskDEFGClaimsOutput) -> List[str]:
        if not out_obj.claims:
            return ["At least 1 scientific claim must be formulated in 'claims'."]
        return verify_claims_against_sources(out_obj.claims, papers)

    validated_defg, rec = await execute_validated_llm_task(
        task_id="Task_DEFG",
        task_name="Tasks D-G: Result Comparison, Claim Formulation & Limitation Appraisal",
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        response_model=TaskDEFGClaimsOutput,
        config=config,
        custom_validator=_validate_claims_out,
        max_retries=2,
        temperature=0.1,
        max_tokens=2200,
        deadline_s=step_budget(config, "claims"),
    )
    audit_trail.append(rec)

    if validated_defg and not rec.degraded_to_deterministic:
        claims_out: List[ClaimEntity] = []
        for idx, c in enumerate(validated_defg.claims, start=1):
            claims_out.append(
                ClaimEntity(
                    claim_id=c.claim_id or f"claim:{idx}",
                    proposition=c.proposition,
                    claim_category=c.claim_category,
                    supporting_indices=c.supporting_indices,
                    opposing_indices=c.opposing_indices,
                    applicability_scope=c.applicability_scope,
                    methodological_caveats=c.methodological_caveats,
                    provenance="abstract_derived",
                    verification_state=rec.verification_status,
                )
            )
        return claims_out

    # Graceful deterministic fallback for Tasks D-G
    clusters: Dict[str, List[PaperRecord]] = {}
    for p in papers:
        clusters.setdefault(p.cluster_label, []).append(p)

    prim_intent = "methodological_comparison"
    if decomposition_plan and decomposition_plan.question_understanding:
        ui = decomposition_plan.question_understanding.primary_intent
        if ui == "causal_intervention":
            prim_intent = "causal_intervention"
        elif ui == "associational":
            prim_intent = "association"
        elif ui == "predictive":
            prim_intent = "predictive_performance"
        elif ui == "explanatory_mechanistic":
            prim_intent = "mechanistic"
        elif ui == "theoretical":
            prim_intent = "theoretical"

    det_claims: List[ClaimEntity] = []
    for c_idx, (cname, cpapers) in enumerate(clusters.items(), start=1):
        sup_idxs = [p.citation_index for p in cpapers if p.stance == "supporting"]
        opp_idxs = [p.citation_index for p in cpapers if p.stance in ("contradicting", "mixed")]
        rep = cpapers[0]
        # Respect Inference Validity on observational studies
        eff_cat = prim_intent
        if eff_cat == "causal_intervention" and all("observational" in p.elicit.study_design.lower() for p in cpapers):
            eff_cat = "association"
        det_claims.append(
            ClaimEntity(
                claim_id=f"claim:{c_idx}",
                proposition=f"In {cname}, {rep.elicit.study_design} investigations report: {rep.elicit.key_findings}",
                claim_category=eff_cat,
                supporting_indices=sup_idxs or [rep.citation_index],
                opposing_indices=opp_idxs,
                applicability_scope=rep.elicit.dataset_or_system,
                methodological_caveats=rep.elicit.limitations,
                provenance=rep.elicit.provenance_type,
                verification_state="passed_with_disclosed_limitations",
            )
        )
    return det_claims


# Purpose of sections C-E per review framework (body subsections B.n follow the chosen organisation)
_DEEP_SECTION_PURPOSES = {
    "narrative": (
        "Explain where studies disagree or report conditional results and why (population, organ, model system, "
        "methods, outcome measures); weigh the contradicting evidence and alternative interpretations.",
        "Assess how reliable the evidence is: design-specific limitations, model-system translatability (animal, "
        "organoid, human), sample sizes, replication, and what the evidence cannot show.",
    ),
    "systematic_prisma": (
        "Analyse heterogeneity and conflicting evidence: where and why studies disagree (population, organ, model "
        "system, methods, measures), weighing the contradicting studies.",
        "Assess risk of bias and certainty of the evidence: design-specific bias concerns and how certain the main "
        "findings are given risk of bias, inconsistency, indirectness, imprecision and publication bias.",
    ),
    "scoping": (
        "Characterise the evidence base: which study designs, model systems, populations, organs and years are "
        "covered, and how unevenly.",
        "Identify evidence gaps: questions, organs, model systems or methods that the retrieved studies do not cover "
        "or cover only weakly, and why they matter.",
    ),
}
_DEEP_CONCLUSION_PURPOSE = (
    "Conclude: answer each part of the research question directly, separate what is established from what is only "
    "suggested or uncertain, and set out implications for future research and study design."
)
# Concurrent section-writing calls per review (local models are serialised separately by the LLM router)
DEEP_SECTION_CONCURRENCY = 4


def _deep_digest_line(p: PaperRecord) -> str:
    who = p.authors[0] if p.authors else p.source
    return (
        f"[{p.citation_index}] {p.title} ({who}, {p.year or 'n.d.'}) - Design: {p.elicit.study_design} | "
        f"System: {p.elicit.dataset_or_system} | Finding: {p.elicit.key_findings} | Quant: {p.elicit.quantitative_result}"
        f" | Limits: {p.elicit.limitations}"
    )


async def _write_deep_review_sections(
    query: str,
    synth_papers: List[PaperRecord],
    validated_claims: List[ClaimEntity],
    synth_claims: List[Any],
    det_sections: List[SynthesisSection],
    organization: str,
    review_framework: str,
    rules_block: str,
    corpus_full_text: str,
    contradicting_idxs: List[int],
    config: ApiKeysConfig,
    standard: bool = False,
) -> Tuple[List[SynthesisSection], List[TaskExecutionRecord], List[str]]:
    """
    `standard=True` writes the four sections of a Standard synthesis (one body section, then C, D and E) at Standard
    length. Writing them as separate short calls side by side takes about a third of the time of one long call,
    and a section that fails falls back alone.

    Writes a Deep Research review one section at a time: body subsections B.1..B.n (per theme, period, method or
    theory), then C, D and E for the review framework. Small, focused calls reliably reach the review's length on
    large and small models alike; a section that fails keeps its rule-based text while the others stay AI-written.
    Returns the sections, one audit record per section, and the headings that fell back to rule-based text.
    """
    framework_def, organization_def = resolve_review_type(review_framework, organization)
    allowed = {p.citation_index for p in synth_papers}
    allowed_label = _index_label(sorted(allowed))
    by_idx = {p.citation_index: p for p in synth_papers}

    def claim_lines(indices: Optional[set]) -> str:
        lines = [
            f"- ({c.claim_category}) {c.proposition} | Supporting: {c.supporting_indices} | Opposing: {c.opposing_indices}"
            for c in validated_claims
            if indices is None or set(c.supporting_indices + c.opposing_indices) & indices
        ]
        return "\n".join(lines) or "- (no verified claims for this part)"

    # Units: (heading, purpose, studies, rule-based fallback text, must cite contradicting studies)
    units: List[Tuple[str, str, List[PaperRecord], str, bool]] = []
    body_fallback = (det_sections[0].content.split("\n\n") if det_sections else [])
    if standard:
        units.append((
            det_sections[0].heading if det_sections else organization_def.body_heading,
            f"{organization_def.prompt_instruction} Synthesise what the studies found across the main themes, compare "
            "their designs and results, and relate them to the research question.",
            synth_papers[:30], det_sections[0].content if det_sections else "", False,
        ))
    for i, (label, ps) in enumerate([] if standard else organized_groups(organization, synth_papers, synth_claims), 1):
        fallback = body_fallback[i - 1] if i - 1 < len(body_fallback) else (det_sections[0].content if det_sections else "")
        units.append((
            f"B.{i} {label}",
            f"{organization_def.prompt_instruction} This subsection covers: {label}. Synthesise what these studies "
            "found, compare their methods, model systems and results, and relate them to the research question.",
            ps, fallback, False,
        ))
    c_purpose, d_purpose = _DEEP_SECTION_PURPOSES.get(framework_def.framework_id, _DEEP_SECTION_PURPOSES["narrative"])
    disagreeing = [p for p in synth_papers if p.stance in ("contradicting", "mixed")]
    others = [p for p in synth_papers if p not in disagreeing]
    for j, (purpose, ps, must_cite) in enumerate([
        (c_purpose, (disagreeing + others)[:25], bool(contradicting_idxs) and framework_def.framework_id != "scoping"),
        (d_purpose, synth_papers[:30], False),
        (_DEEP_CONCLUSION_PURPOSE, synth_papers[:30], False),
    ]):
        fallback = det_sections[j + 1].content if j + 1 < len(det_sections) else ""
        units.append((framework_def.section_plan[j], purpose, ps, fallback, must_cite))

    semaphore = asyncio.Semaphore(DEEP_SECTION_CONCURRENCY)

    async def write(unit) -> Tuple[SynthesisSection, TaskExecutionRecord, bool]:
        heading, purpose, ps, fallback, must_cite = unit
        unit_idx = {p.citation_index for p in ps}
        target_words = _deep_section_target_words(len(ps))
        if standard:
            target_words = min(target_words, STANDARD_BODY_WORDS if heading == units[0][0] else STANDARD_SECTION_WORDS)
        # A Standard section is short by design: a firm upper length keeps each call quick
        upper_words = target_words + (90 if standard else 250)
        proposes_heading = heading.startswith("B.") and not standard
        sys_prompt = (
            f"You are writing ONE section of a {'literature synthesis' if standard else 'Deep Research academic literature review'}.\n"
            f"Section: {heading}\nPurpose: {purpose}\n"
            f"Write {target_words}-{upper_words} words of continuous academic prose (no bullet "
            "lists, no heading), synthesising across studies rather than summarising them one by one.\n"
            "RULES:\n"
            f"1. Cite every factual statement with bracketed study numbers such as [3] or [2, 5], using only: {allowed_label}.\n"
            "2. Never invent numbers, percentages or sample sizes that are not in the provided records.\n"
            "3. Do not claim causation from observational or correlational evidence.\n"
            f"Methodology rules:\n{rules_block}\n"
            + ('Return ONLY valid JSON: {"heading": "short descriptive subsection title (max 10 words)", "content": "..."}'
               if proposes_heading else 'Return ONLY valid JSON: {"content": "..."}')
        )
        user_prompt = (
            f"Research question: {query}\n\nVerified claims:\n{claim_lines(None if len(ps) >= len(synth_papers) else unit_idx)}"
            "\n\nStudies for this section:\n" + "\n".join(_deep_digest_line(p) for p in ps)
        )

        def validate(out: TaskJSectionOutput) -> List[str]:
            errs: List[str] = []
            text = out.content or ""
            words = len(text.split())
            if words < target_words * DEEP_LENGTH_TOLERANCE:
                # No semicolons: the audit record joins errors with "; " and the length-only check splits on it
                errs.append(f"The section is {words} words - write at least {target_words} words, keeping "
                            "what you wrote and adding study-by-study comparison.")
            cited = extract_citation_numbers(text)
            if not cited:
                errs.append("The section must cite studies with bracketed numbers such as [3] or [2, 5].")
            bad = sorted({c for c in cited if c not in allowed})
            if bad:
                errs.append(f"Cites study numbers {bad} that are not among the provided studies; use only {allowed_label}.")
            for num in extract_nontrivial_numbers(text):
                if num not in corpus_full_text:
                    errs.append(f"Contains the number '{num}', which is not in the retrieved study records.")
            if must_cite and not any(ci in cited for ci in contradicting_idxs):
                errs.append(f"Must cite and weigh the contradicting studies {contradicting_idxs}.")
            return errs

        def repair(out: TaskJSectionOutput, _errors: List[str]) -> Optional[TaskJSectionOutput]:
            # Drop the sentence with an unsupported figure, or the citation of a study that was not provided
            text = strip_citations_not_in(out.content or "", allowed)
            text = drop_unsupported_numbers(text, corpus_full_text, by_clause=False)
            return out.model_copy(update={"content": text}) if text and text != out.content else None

        async with semaphore:
            out, rec = await execute_validated_llm_task(
                task_id="Task_J",
                task_name=f"Task J: Section {heading}",
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                response_model=TaskJSectionOutput,
                config=config,
                custom_validator=validate,
                max_retries=1,
                temperature=0.2,
                max_tokens=int(upper_words * 1.6) + 220 if standard else int(max(target_words, 150) * 1.8) + 500,
                deadline_s=step_budget(config, "section"),
                repair=repair,
            )
        if out is None and _keep_if_only_short(rec):
            out = TaskJSectionOutput(**rec.last_candidate)
        if out is not None and not rec.degraded_to_deterministic or (out is not None and rec.last_candidate):
            content = normalize_citations(out.content)
            used_fallback = False
            proposed = re.sub(r"[\[\]\d,;]+$", "", (out.heading or "").strip()).strip(" .:-")
            if proposes_heading and proposed and len(proposed.split()) <= 14:
                heading = f"{heading.split()[0]} {proposed}"
        else:
            content, used_fallback = fallback, True
        cited = sorted({n for n in extract_citation_numbers(content) if n in by_idx})
        return SynthesisSection(heading=heading, content=content, cited_indices=cited), rec, used_fallback

    results = await asyncio.gather(*[write(u) for u in units])
    sections = [r[0] for r in results]
    records = [r[1] for r in results]
    fallbacks = [r[0].heading for r in results if r[2]]
    return sections, records, fallbacks


async def generate_synthesis_and_extractions(
    query: str,
    mode: str,
    papers: List[PaperRecord],
    hf_resources: List[HFResourceRecord],
    bibliography: List[BibliographyEntry],
    config: ApiKeysConfig,
    decomposition_plan: Optional[QueryDecompositionPlan] = None,
    on_claim_evidence_ready: Optional[Callable[[SynthesisEvidencePackage], Awaitable[None]]] = None,
    review_framework: str = "narrative",
    organization: str = "thematic",
) -> Tuple[str, List[SynthesisSection], ConsensusBreakdown, List[str]]:
    """
    `on_claim_evidence_ready`, when given, is started as a concurrent task as soon as the study findings,
    clusters and verified claims exist (right after Task C). Downstream work that only needs those (evidence
    table, gap matrix, Tasks H-I) then overlaps the Tasks D-G and J LLM calls; it is awaited before returning.

    Phase 8 & Sections 3-9:
    Executes bounded scientific tasks:
    - Task C: Context-window-batched study & result extraction
    - Tasks D-G: Cross-study result comparison, scientific claim formulation, supporting/opposing weighing, and limitation appraisal
    - Task J: Hierarchical narrative synthesis strictly from validated study results and claims
    """
    warnings: List[str] = []
    audit_trail: List[TaskExecutionRecord] = []

    prim_domain = "biomedical_clinical"
    sample_policy = "optional"
    if decomposition_plan and decomposition_plan.active_profile_summary:
        prim_domain = decomposition_plan.active_profile_summary.get("primary_domain", "biomedical_clinical")
        sample_policy = decomposition_plan.active_profile_summary.get("sample_reporting_policy", "optional")

    if not papers:
        _scoped_claims().clear()
        _scoped_audit_trail().clear()
        empty_pkg = SynthesisEvidencePackage(
            workflow_version=WORKFLOW_VERSION,
            domain_appraisal_profile=prim_domain,
            question_answerability="no_eligible_evidence",
            disclosures=["No eligible studies were returned from the enabled sources."],
        )
        set_last_synthesis_package(empty_pkg)
        empty_consensus = ConsensusBreakdown(
            supporting_pct=0,
            mixed_pct=0,
            contradicting_pct=0,
            total_evaluated=0,
            consensus_verdict="Insufficient Evidence",
            consensus_summary="No matching papers were returned from the enabled sources.",
        )
        return (
            "No eligible studies were found for this query across the enabled databases. Try broadening your search constraints.",
            [],
            empty_consensus,
            warnings,
        )

    # 1. Execute Task C: Batched Study & Result Extraction
    await _run_task_c_batched_study_extraction(
        query=query,
        papers=papers,
        hf_resources=hf_resources,
        prim_domain=prim_domain,
        sample_policy=sample_policy,
        config=config,
        audit_trail=audit_trail,
        descriptive_question=bool(
            decomposition_plan and decomposition_plan.question_understanding
            and decomposition_plan.question_understanding.primary_intent == "descriptive"
        ),
    )

    # Studies screened as not addressing the question (constraint mismatch or the model's judgement) keep their
    # reference numbers but are left out of claims, the narrative, the agreement meter and the contradiction check
    synth_papers = [p for p in papers if p.relevance != "off_topic"]
    off_topic = [p for p in papers if p.relevance == "off_topic"]
    if len(synth_papers) < MIN_SYNTHESIS_PAPERS:
        synth_papers = list(papers)
        if off_topic:
            warnings.append(
                f"Relevance screening flagged {len(off_topic)} of {len(papers)} studies as not addressing the question, "
                "but too few remained to synthesise, so all studies were kept; treat flagged studies with caution."
            )
    elif off_topic:
        warnings.append(
            f"Relevance screening: {len(off_topic)} of {len(papers)} retrieved studies do not address the question "
            f"and were left out of the synthesis (references {', '.join(str(p.citation_index) for p in off_topic)})."
        )

    # 1b. Section 1 Phases 3, 4, 5, 6, 7: Atomic Findings -> Comparable Clusters -> Verified Claims
    meth_profiles = (
        list(decomposition_plan.question_understanding.active_methodological_profiles)
        if decomposition_plan and decomposition_plan.question_understanding
        else []
    )
    appraisal_profile_id, appraisal_considerations = get_domain_appraisal_considerations(
        prim_domain, meth_profiles
    )
    atomic_findings = extract_atomic_findings(query, synth_papers, decomposition_plan)
    # Studies whose open-access full text was used are reported as full-text evidence, not abstract-only
    full_text_idx = {p.citation_index for p in synth_papers if p.full_text_sections}
    for f in atomic_findings:
        if f.citation_index in full_text_idx:
            f.evidence_access = "full_text"
    evidence_clusters = cluster_comparable_findings(query, synth_papers, atomic_findings)
    verified_synth_claims, excluded_synth_claims = build_and_verify_claims(
        query, synth_papers, atomic_findings, evidence_clusters
    )
    answerability = determine_question_answerability(synth_papers, atomic_findings, verified_synth_claims)

    evidence_task: Optional[asyncio.Task] = None
    handed_off_records = 0
    if on_claim_evidence_ready is not None:
        # Publish Task C records now; the concurrent task appends its own (e.g. Task H-I) to the same scoped trail
        _scoped_audit_trail().clear()
        _scoped_audit_trail().extend(audit_trail)
        handed_off_records = len(audit_trail)
        evidence_task = asyncio.create_task(
            on_claim_evidence_ready(
                SynthesisEvidencePackage(
                    workflow_version=WORKFLOW_VERSION,
                    domain_appraisal_profile=appraisal_profile_id,
                    appraisal_considerations=appraisal_considerations,
                    question_answerability=answerability,
                    findings=atomic_findings,
                    clusters=evidence_clusters,
                    claims=verified_synth_claims,
                    excluded_claims=excluded_synth_claims,
                )
            )
        )

    # Pre-build verified Section 1 narrative (Subsections A-E) from verified claim records
    det_exec, det_sections, det_verdict, det_summary = compose_section1_narrative_from_verified_claims(
        query=query,
        mode=mode,
        papers=synth_papers,
        hf_resources=hf_resources,
        findings=atomic_findings,
        clusters=evidence_clusters,
        verified_claims=verified_synth_claims,
        excluded_claims=excluded_synth_claims,
        primary_domain=prim_domain,
        appraisal_considerations=appraisal_considerations,
        answerability=answerability,
        organization=organization,
        review_framework=review_framework,
    )

    # 2. Execute Tasks D, E, F, G: Result Comparison & Scientific Claim Formulation
    validated_claims = await _run_tasks_defg_claim_formulation(
        query=query,
        papers=synth_papers,
        prim_domain=prim_domain,
        decomposition_plan=decomposition_plan,
        config=config,
        audit_trail=audit_trail,
    )
    _scoped_claims().clear()
    _scoped_claims().extend(validated_claims)

    # 3. Execute Task J: Hierarchical Narrative Synthesis from Validated Claims & Study Results
    claims_digest = "\n".join(
        f"- [{c.claim_id}] ({c.claim_category}) {c.proposition} | Supporting: {c.supporting_indices} | Opposing: {c.opposing_indices} | Scope: {c.applicability_scope} | Caveats: {c.methodological_caveats}"
        for c in validated_claims
    )
    paper_digest_lines = [
        f"[{p.citation_index}] {p.title} ({p.authors[0] if p.authors else p.source}, {p.year or 'n.d.'}, Status: {p.version_status}, Stance: {p.stance}) — "
        f"Design: {p.elicit.study_design} | Sample: {p.elicit.sample_or_scale} ({p.elicit.sample_reporting_status}) | Finding: {p.elicit.key_findings} | Quant: {p.elicit.quantitative_result}"
        for p in synth_papers[:60]
    ]
    hf_digest_lines = [
        f"- {hf.id} ({hf.resource_type}, pipeline: {hf.pipeline_tag}, downloads: {hf.downloads}) [Community artifact]"
        for hf in hf_resources[:8]
    ]
    # Only the source-cited rules for this domain, the study designs present and the review type (token-capped)
    study_designs = sorted(
        {d for d in (detect_biomedical_design(p.abstract or "") for p in synth_papers) if d}
        | set(meth_profiles)
    )
    rules_block = render_rules_for_prompt(
        select_rules(prim_domain, study_designs, review_framework, ("synthesis", "reporting")),
        max_tokens=PROMPT_RULES_TOKEN_BUDGET,
    )

    framework_def, organization_def = resolve_review_type(review_framework, organization)
    section_spec = (
        f"Write it as a {framework_def.label.lower()}. Organize the narrative into these subsections: "
        f"{organization_def.body_heading}, " + ", ".join(framework_def.section_plan) + ". "
        f"{organization_def.prompt_instruction}"
    )
    # With a model available, the sections are written as separate calls alongside this one (both modes), so this
    # call writes only the executive answer and the agreement verdict
    llm_available = any(r.model_used != "deterministic_offline_engine" for r in audit_trail)
    sectioned = llm_available
    if sectioned and mode != "deep_research":
        section_spec = (
            "The sections of this synthesis are written separately. Write ONLY the executive answer "
            f"({STANDARD_EXEC_WORDS}-{STANDARD_EXEC_WORDS + 100} words: a direct, cited answer to the research question), "
            "consensus_verdict and consensus_summary, and return 'sections' as an empty list."
        )
    if mode == "deep_research":
        # Deep Research sections are written one at a time by _write_deep_review_sections; this call writes only the
        # executive answer and the agreement verdict
        section_spec = (
            "This is a Deep Research review whose sections are written separately. Write ONLY the executive answer "
            f"({DEEP_EXEC_MIN_WORDS}-{DEEP_EXEC_MIN_WORDS + 200} words: a direct, cited answer to every part of the "
            "research question), consensus_verdict and consensus_summary, and return 'sections' as an empty list."
        )

    sys_prompt_j = (
        "You are executing Task J (Hierarchical Narrative Synthesis from Validated Records) of a quality-gated scientific engine.\n"
        "NON-NEGOTIABLE SCIENTIFIC RULES:\n"
        "1. Synthesize strictly from the provided Validated Scientific Claims and Validated Study Results—do NOT write a paper-by-paper list.\n"
        "2. Every factual statement MUST carry inline bracketed citations like [1], [2, 4], using only these study numbers: "
        + _index_label(sorted(p.citation_index for p in synth_papers)) + ".\n"
        "3. Never invent numbers, percentages, or sample sizes not present in the provided records.\n"
        "4. If any contradicting studies exist in the corpus, explicitly cite and weigh them.\n"
        f"{section_spec}\n"
        "Return ONLY valid JSON matching:\n"
        '{"executive_answer": "...", "consensus_verdict": "...", "consensus_summary": "...", "sections": [{"heading": "...", "content": "...", "cited_indices": [1, 2]}]}'
    )
    user_prompt_j = (
        f"Research Query: {query}\nActive Domain: {prim_domain}\nDomain Rules:\n{rules_block}\n\n"
        f"Validated Scientific Claims (Tasks D-G):\n{claims_digest}\n\n"
        f"Validated Study Results (Task C):\n" + "\n".join(paper_digest_lines) + "\n\n"
        f"Hugging Face Artifacts:\n" + ("\n".join(hf_digest_lines) if hf_digest_lines else "None")
    )

    max_idx = len(papers)
    corpus_full_text = " ".join(f"{study_source_text(p)} {p.elicit.quantitative_result}" for p in papers)
    contradicting_idxs = [p.citation_index for p in synth_papers if p.stance == "contradicting"]

    def _validate_task_j(out_obj: TaskJNarrativeSynthesisOutput) -> List[str]:
        errs: List[str] = []
        if not out_obj.sections and not sectioned:
            errs.append("sections list cannot be empty.")
        combined_text = out_obj.executive_answer + "\n" + "\n".join(s.content for s in out_obj.sections)
        cited = extract_citation_numbers(combined_text)
        if not cited:
            errs.append("Synthesis text must include inline bracketed citations like [1].")
        allowed = {p.citation_index for p in synth_papers}
        out_of_bounds = [c for c in cited if c not in allowed]
        if out_of_bounds:
            errs.append(f"Synthesis cites study numbers {sorted(set(out_of_bounds))} that are not among the provided "
                        f"studies; use only {_index_label(sorted(allowed))}.")
        for num in extract_nontrivial_numbers(combined_text):
            if num not in corpus_full_text:
                errs.append(f"Synthesis contains ungrounded number '{num}' not present in the retrieved study records.")
        if mode == "deep_research":
            # Contradicting studies are required in section C, which is written and checked separately
            words = len(out_obj.executive_answer.split())
            if words < DEEP_EXEC_MIN_WORDS * DEEP_LENGTH_TOLERANCE:
                errs.append(f"The executive answer is {words} words - write at least {DEEP_EXEC_MIN_WORDS} words that "
                            "answer every part of the research question with citations.")
        elif not sectioned and contradicting_idxs and not any(ci in cited for ci in contradicting_idxs):
            errs.append(f"Synthesis omitted contradicting study index(es) {contradicting_idxs}; opposing evidence must be cited.")
        return errs

    def _repair_task_j(out_obj: TaskJNarrativeSynthesisOutput, _errors: List[str]) -> Optional[TaskJNarrativeSynthesisOutput]:
        # Drop the sentence with an unsupported figure, or the citation of a study that was not provided
        allowed = {p.citation_index for p in synth_papers}
        clean = lambda t: drop_unsupported_numbers(strip_citations_not_in(t or "", allowed), corpus_full_text, by_clause=False)
        fixed = out_obj.model_copy(deep=True)
        fixed.executive_answer = clean(fixed.executive_answer)
        for sec in fixed.sections:
            sec.content = clean(sec.content)
        return fixed if fixed.executive_answer else None

    j_call = execute_validated_llm_task(
        task_id="Task_J",
        task_name="Task J: Hierarchical Narrative Synthesis from Validated Claims"
        + (" (executive answer)" if mode == "deep_research" else ""),
        system_prompt=sys_prompt_j,
        user_prompt=user_prompt_j,
        response_model=TaskJNarrativeSynthesisOutput,
        config=config,
        custom_validator=_validate_task_j,
        max_retries=1 if sectioned else 2,
        temperature=0.15,
        max_tokens=DEEP_EXEC_MAX_TOKENS if mode == "deep_research" else (STANDARD_EXEC_MAX_TOKENS if sectioned else 2800),
        deadline_s=step_budget(config, "answer"),
        repair=_repair_task_j,
    )
    deep_sections: Optional[List[SynthesisSection]] = None
    deep_fallbacks: List[str] = []
    if sectioned:
        (validated_j, rec_j), (deep_sections, deep_records, deep_fallbacks) = await asyncio.gather(
            j_call,
            _write_deep_review_sections(
                query, synth_papers, validated_claims, verified_synth_claims, det_sections, organization,
                review_framework, rules_block, corpus_full_text, contradicting_idxs, config,
                standard=mode != "deep_research",
            ),
        )
        audit_trail.append(rec_j)
        audit_trail.extend(deep_records)
        if deep_fallbacks:
            warnings.append(
                f"{len(deep_fallbacks)} of {len(deep_sections)} sections fell back to rule-based text "
                f"({'; '.join(deep_fallbacks)})."
            )
    else:
        validated_j, rec_j = await j_call
        audit_trail.append(rec_j)

    if validated_j is None and _keep_if_only_short(rec_j):
        validated_j = TaskJNarrativeSynthesisOutput(**rec_j.last_candidate)

    if evidence_task is None:
        _scoped_audit_trail().clear()
        _scoped_audit_trail().extend(audit_trail)
    else:
        _scoped_audit_trail().extend(audit_trail[handed_off_records:])

    # Sections whose separate call failed were written from the verified claim records instead
    regenerated_secs: List[str] = list(deep_fallbacks)

    if validated_j and not rec_j.degraded_to_deterministic:
        exec_answer = _ensure_inline_citations(normalize_citations(validated_j.executive_answer), len(papers))
        if len(exec_answer.split()) < 35:
            exec_answer = f"{exec_answer}\n\n{det_exec}"
            regenerated_secs.append("A. Direct Answer / Executive Scientific Synthesis")

        sections_out: List[SynthesisSection] = []
        for sec in ([] if deep_sections else validated_j.sections):
            content = _ensure_inline_citations(normalize_citations(sec.content), len(papers))
            cited = [int(i) for i in sec.cited_indices if 1 <= int(i) <= len(papers)]
            if not cited:
                cited = [n for n in extract_citation_numbers(content) if 1 <= n <= len(papers)]
            sections_out.append(
                SynthesisSection(
                    heading=sec.heading or "Analysis",
                    content=content,
                    cited_indices=sorted(list(set(cited))),
                )
            )

        if deep_sections:
            sections_out = list(deep_sections)
        # Ensure all 4 analytical dimensions (B-E) are represented; selectively append missing subsections from verified claims
        if len(sections_out) < 4:
            existing_headings_low = " ".join(s.heading.lower() for s in sections_out)
            for det_s in det_sections:
                key_word = det_s.heading.split()[1].lower() if len(det_s.heading.split()) > 1 else det_s.heading.lower()
                if key_word not in existing_headings_low and len(sections_out) < 4:
                    sections_out.append(det_s)
                    regenerated_secs.append(det_s.heading)

        sections_out = finalize_llm_sections(review_framework, sections_out, synth_papers, atomic_findings)
        verdict = validated_j.consensus_verdict or det_verdict
        c_summary = _ensure_inline_citations(
            validated_j.consensus_summary or det_summary,
            len(papers),
        )
        if rec_j.attempts_used > 1:
            warnings.append(
                f"Quality Gate Notice: Task J passed validation after {rec_j.attempts_used - 1} corrective retry(ies) on {rec_j.model_used}."
            )
    else:
        if rec_j.model_used == "deterministic_offline_engine":
            warnings.append(
                "Operating in deterministic claim-centered synthesis mode (no external LLM key configured). "
                "All claims, reporting statuses (Reported / Not reported / Not applicable), and citations are derived directly from retrieved study records."
            )
        else:
            reason_str = "; ".join(rec_j.validation_errors[-1:]) if rec_j.validation_errors else "unreachable endpoint"
            warnings.append(
                f"Quality Gate Degradation Disclosure: Model ({rec_j.model_used}) did not satisfy Task J verification after bounded retries ({reason_str}) — "
                "degraded gracefully to deterministic source-grounded claim synthesis."
            )

        exec_answer, sections_out, verdict, c_summary = det_exec, det_sections, det_verdict, det_summary
        if deep_sections:
            # The executive answer fell back, but the separately written sections can still be used
            sections_out = finalize_llm_sections(review_framework, list(deep_sections), synth_papers, atomic_findings)

    # Phase 11: Run Explicit 10-Dimension Section 1 Synthesis Quality Gate
    s1_quality = evaluate_section1_quality_gate(
        query=query,
        papers=synth_papers,
        findings=atomic_findings,
        clusters=evidence_clusters,
        verified_claims=verified_synth_claims,
        executive_answer=exec_answer,
        sections=sections_out,
        primary_domain=prim_domain,
        model_used=rec_j.model_used,
    )
    failed = [c for c in s1_quality.checks if not c.passed]
    hard_failed = [c for c in failed if c.name in HARD_QUALITY_CHECKS]
    llm_narrative = exec_answer is not det_exec
    if llm_narrative and failed and not hard_failed:
        # Softer issues (wording, coverage, structure) do not justify discarding a verified model answer: keep it
        # and say what the checks noticed
        s1_quality.narrative_origin = "llm_partially_verified"
        warnings.append("Quality notes on the synthesis: " + " ".join(r for c in failed for r in c.reasons[:1]))
    elif not s1_quality.all_passed:
        if llm_narrative:
            warnings.append(
                "The model-written synthesis was replaced by the rule-based synthesis because it failed an integrity "
                "check: " + " ".join(r for c in hard_failed for r in c.reasons[:1]))
        # Bounded section-level regeneration from verified claims
        exec_answer, sections_out, verdict, c_summary = det_exec, det_sections, det_verdict, det_summary
        regenerated_secs.append("Full narrative regenerated from verified claim records after quality gate check")
        s1_quality = evaluate_section1_quality_gate(
            query=query,
            papers=synth_papers,
            findings=atomic_findings,
            clusters=evidence_clusters,
            verified_claims=verified_synth_claims,
            executive_answer=exec_answer,
            sections=sections_out,
            primary_domain=prim_domain,
            model_used=rec_j.model_used,
        )
    s1_quality.regenerated_sections = regenerated_secs

    # Mark cited bibliography entries (Phase 9)
    if bibliography:
        mark_cited_bibliography_entries(bibliography, exec_answer, sections_out)

    # Save Section 1 SynthesisEvidencePackage
    disclosures: List[str] = []
    if any(f.excluded_reason for f in atomic_findings):
        disclosures.append(
            f"{sum(1 for f in atomic_findings if f.excluded_reason)} record(s) flagged with retraction/concern notices and excluded from supporting synthesis."
        )
    if answerability == "not_answerable_from_selected_evidence":
        disclosures.append("Selected evidence has low direct topical overlap with the user question; synthesis reports what retrieved studies establish without extrapolating.")

    pkg = SynthesisEvidencePackage(
        workflow_version=WORKFLOW_VERSION,
        domain_appraisal_profile=appraisal_profile_id,
        appraisal_considerations=appraisal_considerations,
        question_answerability=answerability,
        findings=atomic_findings,
        clusters=evidence_clusters,
        claims=verified_synth_claims,
        excluded_claims=excluded_synth_claims,
        quality=s1_quality,
        disclosures=disclosures,
    )
    set_last_synthesis_package(pkg)

    # Agreement meter: on-topic studies with a direction of effect only
    rated = [p for p in synth_papers if p.stance in ("supporting", "mixed", "contradicting")]
    total_eval = len(rated)
    sup_count = sum(1 for p in rated if p.stance == "supporting")
    mix_count = sum(1 for p in rated if p.stance == "mixed")
    con_count = sum(1 for p in rated if p.stance == "contradicting")

    sup_pct = round((sup_count / total_eval) * 100) if total_eval else 0
    mix_pct = round((mix_count / total_eval) * 100) if total_eval else 0
    con_pct = max(0, 100 - sup_pct - mix_pct) if total_eval else 0

    consensus = ConsensusBreakdown(
        supporting_pct=sup_pct,
        mixed_pct=mix_pct,
        contradicting_pct=con_pct,
        total_evaluated=total_eval,
        consensus_verdict=verdict,
        consensus_summary=c_summary,
    )

    if evidence_task is not None:
        await evidence_task
    return exec_answer, sections_out, consensus, warnings


def _build_deterministic_synthesis(
    query: str,
    mode: str,
    papers: List[PaperRecord],
    hf_resources: List[HFResourceRecord],
    prim_domain: str = "biomedical_clinical",
) -> Tuple[str, List[SynthesisSection], str, str]:
    """
    Constructs a claim-level, domain-qualified synthesis directly from the retrieved
    evidence base, distinguishing study designs, reporting statuses, and opposing evidence.
    """
    top_p = papers[: min(6, len(papers))]
    top_refs = ", ".join(f"[{p.citation_index}]" for p in top_p[:3])

    sources_used = sorted(list({s for p in papers for s in (p.all_sources or [p.source]) if s}))
    years = [p.year for p in papers if p.year]
    yr_range = f"{min(years)}–{max(years)}" if years else "recent years"

    pub_count = sum(1 for p in papers if p.version_status in ("published", "preprint_with_published_link"))
    prep_count = len(papers) - pub_count

    p1 = papers[0]
    p2 = papers[1] if len(papers) > 1 else papers[0]
    p3 = papers[2] if len(papers) > 2 else papers[0]

    domain_prof = DOMAIN_PROFILES.get(prim_domain, DOMAIN_PROFILES["biomedical_clinical"])

    exec_para_1 = (
        f"Across {len(papers)} unique studies ({pub_count} peer-reviewed publications and {prep_count} preprints, {yr_range}) "
        f"retrieved from {', '.join(sources_used)} under the {domain_prof.display_name} profile, "
        f"the evidence base addressing \"{query}\" centers on {p1.elicit.study_design} and {p2.elicit.study_design} {top_refs}. "
        f"Primary investigations ({p1.authors[0] if p1.authors else p1.source} et al., {p1.year or 'n.d.'} [{p1.citation_index}]) "
        f"demonstrate that {p1.elicit.key_findings}"
    )
    exec_para_2 = (
        f"Cross-study comparison with {p2.authors[0] if p2.authors else p2.source} et al. ({p2.year or 'n.d.'}) [{p2.citation_index}] "
        f"and {p3.authors[0] if p3.authors else p3.source} et al. ({p3.year or 'n.d.'}) [{p3.citation_index}] indicates that outcomes "
        f"are conditioned on {p2.elicit.dataset_or_system} and methodological assumptions [{p2.citation_index}, {p3.citation_index}]. "
        f"In accordance with {domain_prof.display_name} appraisal standards, abstract-derived findings should be interpreted "
        f"within their stated regime ({p1.elicit.sample_reporting_status.lower()} sample/scale metadata) without extrapolating beyond validated conditions [{p1.citation_index}, {p3.citation_index}]."
    )
    exec_answer = f"{exec_para_1}\n\n{exec_para_2}"

    clusters: Dict[str, List[PaperRecord]] = {}
    for p in papers:
        clusters.setdefault(p.cluster_label, []).append(p)

    sections: List[SynthesisSection] = []

    if mode == "deep_research":
        sec1_papers = sorted(papers, key=lambda x: (x.citation_count, x.relevance_score), reverse=True)[:5]
        sec1_lines = [
            f"• [{p.citation_index}] ({p.elicit.study_design}, {p.version_status}): {p.elicit.key_findings} (System/Scale: {p.elicit.dataset_or_system})."
            for p in sec1_papers
        ]
        sections.append(
            SynthesisSection(
                heading="1. Core Scientific Propositions & Foundational Evidence",
                content=(
                    f"Evaluating the primary propositions across foundational and high-relevance studies "
                    f"[{', '.join(str(p.citation_index) for p in sec1_papers)}] reveals the following verified findings:\n"
                    + "\n".join(sec1_lines)
                ),
                cited_indices=[p.citation_index for p in sec1_papers],
            )
        )

        tax_lines = []
        all_tax_indices = []
        for cname, cpapers in clusters.items():
            c_idxs = [p.citation_index for p in cpapers]
            all_tax_indices.extend(c_idxs)
            methods_set = list(dict.fromkeys(p.elicit.methodology for p in cpapers))[:3]
            rep = cpapers[0]
            tax_lines.append(
                f"• {cname} ({len(cpapers)} studies [{', '.join(str(i) for i in c_idxs)}]): "
                f"Employs {'; '.join(methods_set)}. Representative evidence: {rep.elicit.key_findings} [{rep.citation_index}]."
            )
        sections.append(
            SynthesisSection(
                heading="2. Methodological & Architectural Comparison",
                content="Grouping studies by methodological paradigm highlights complementary strengths and regime assumptions:\n" + "\n".join(tax_lines),
                cited_indices=sorted(list(set(all_tax_indices))),
            )
        )

        bench_papers = papers[:6]
        bench_lines = [
            f"• [{p.citation_index}] {p.title[:80]} — System: {p.elicit.dataset_or_system} | Quantitative/Formal Result: {p.elicit.quantitative_result} | Provenance: {p.elicit.provenance_type}."
            for p in bench_papers
        ]
        sections.append(
            SynthesisSection(
                heading="3. Quantitative Evidence, Regimes & Provenance Appraisal",
                content="Traceable measurements, formal bounds, and reporting statuses across the top-ranked studies:\n"
                + "\n".join(bench_lines),
                cited_indices=[p.citation_index for p in bench_papers],
            )
        )

        deb_papers = [p for p in papers if p.stance in ("mixed", "contradicting")] or papers[-4:]
        deb_lines = [
            f"• [{p.citation_index}] ({p.stance.upper()} — {p.elicit.study_design}): {p.elicit.limitations}"
            for p in deb_papers[:5]
        ]
        sections.append(
            SynthesisSection(
                heading="4. Contradictory Findings, Methodological Caveats & Uncertainty",
                content="Critical evaluation of opposing results, conditional trade-offs, and scope limitations:\n"
                + "\n".join(deb_lines),
                cited_indices=[p.citation_index for p in deb_papers[:5]],
            )
        )

        if hf_resources:
            hf_lines = [
                f"• {hf.id} ({hf.resource_type.upper()} — {hf.pipeline_tag}, {hf.downloads:,} downloads): {hf.description} [Note: Community artifact; verify against peer-reviewed benchmarks]"
                for hf in hf_resources[:6]
            ]
            hf_text = (
                f"Open-source models and datasets on Hugging Face provide computational building blocks related to "
                f"this domain [{papers[0].citation_index}]:\n" + "\n".join(hf_lines)
            )
        else:
            hf_text = f"No domain-specific Hugging Face artifacts were matched for this query [{papers[0].citation_index}]."
        sections.append(
            SynthesisSection(
                heading="5. Open-Source Models & Datasets (Non-Peer-Reviewed Artifacts)",
                content=hf_text,
                cited_indices=[papers[0].citation_index],
            )
        )
    else:
        for cname, cpapers in list(clusters.items())[:3]:
            c_idxs = [p.citation_index for p in cpapers]
            summary_sents = [
                f"{p.authors[0] if p.authors else p.source} et al. ({p.year or 'n.d.'}) [{p.citation_index}] ({p.elicit.study_design} on {p.elicit.dataset_or_system}) "
                f"reports that {p.elicit.key_findings}"
                for p in cpapers[:3]
            ]
            sections.append(
                SynthesisSection(
                    heading=f"Claim Cluster: {cname}",
                    content=" ".join(summary_sents),
                    cited_indices=c_idxs[:4],
                )
            )

        lim_papers = [p for p in papers if p.stance in ("mixed", "contradicting")][:3] or papers[:3]
        sections.append(
            SynthesisSection(
                heading="Methodological Caveats, Opposing Evidence & Scope Limits",
                content=" ".join(f"[{p.citation_index}] ({p.stance}) notes: {p.elicit.limitations}" for p in lim_papers),
                cited_indices=[p.citation_index for p in lim_papers],
            )
        )

    verdict = "Context-Dependent Evidence Agreement"
    c_summary = (
        f"Across {len(papers)} evaluated studies [1, {min(2, len(papers))}], findings support core methodological feasibility "
        f"within tested regimes, while highlighting {domain_prof.display_name} constraints in generalization and validation."
    )
    return exec_answer, sections, verdict, c_summary


GAP_CATEGORIES_CYCLE: List[GapCategoryType] = [
    "Evidence absence",
    "Validation deficiencies",
    "Methodological limitations",
    "Population or contextual limitations",
    "Translational deficiencies",
    "Measurement/standardization limitations",
    "Mechanistic uncertainty",
    "Inconsistent findings",
]


async def build_knowledge_gap_matrix(
    query: str,
    papers: List[PaperRecord],
    hf_resources: List[HFResourceRecord],
    config: ApiKeysConfig,
    decomposition_plan: Optional[QueryDecompositionPlan] = None,
    synthesis_package: Optional[SynthesisEvidencePackage] = None,
    evidence_matrix: Optional[StructuredEvidenceMatrix] = None,
    all_candidate_pool: Optional[List[PaperRecord]] = None,
) -> KnowledgeGapMatrix:
    """
    Phase 8 & Section 3 — Universal Evidence-Gap Identification, Verification, and Research Prioritization Engine:
    - Builds a question-driven evidence coverage model and domain-adaptive 4x4 grid.
    - Classifies every gap into G1-G9 (`GapCategoryType`) with primary and secondary categories.
    - Executes bounded targeted gap verification and 10-point adversarial checks.
    - Separates Knowledge Gaps from Methodological Limitations and constructive Research Opportunities.
    """
    if synthesis_package is None:
        synthesis_package = get_last_synthesis_package()

    if not papers:
        return await build_universal_knowledge_gap_matrix(
            query=query, papers=papers, hf_resources=hf_resources, config=config,
            decomposition_plan=decomposition_plan, synthesis_package=synthesis_package,
            evidence_matrix=evidence_matrix, all_candidate_pool=all_candidate_pool,
        )

    prim_domain = "biomedical_clinical"
    if decomposition_plan and decomposition_plan.active_profile_summary:
        prim_domain = decomposition_plan.active_profile_summary.get("primary_domain", "biomedical_clinical")
    dom_prof = DOMAIN_PROFILES.get(prim_domain, DOMAIN_PROFILES["biomedical_clinical"])

    paper_summaries = "\n".join(
        f"[{p.citation_index}] {p.title} | Design: {p.elicit.study_design} | System: {p.elicit.dataset_or_system} | Limits: {p.elicit.limitations}"
        for p in papers[:24]
    )
    hf_names = [hf.id for hf in hf_resources[:6]]
    core_terms = " ".join(extract_core_scientific_terms(query, max_terms=3))

    sys_prompt = (
        f"You are executing Tasks H & I (Candidate Gap Identification & Verification) under the {dom_prof.display_name} profile.\n"
        "Build a 4x4 evidence grid specific to THIS research question (not generic to the discipline): "
        "rows = 4 sub-topics, exposures, interventions or methods the question is about; "
        "cols = 4 outcomes, populations, settings or evaluation contexts that matter for answering it. "
        "Use short labels in the question's own terms (e.g. for a disease-burden question: risk factors, prevalence, "
        "outcomes; rural vs urban; women; children).\n"
        "Map which study numbers (as given in the list) fall into each (row, col) cell. For cells with 0 or 1 studies, "
        "why_gap_exists must say concretely what is not known, naming the topic (never generic text such as "
        "'lack of validation'). Keep why_gap_exists and proposed_hypothesis under 25 words each. For cells with 2 "
        "or more studies give only row_idx, col_idx and paper_indices.\n"
        "CRITICAL SCIENTIFIC RULE: A cell with 0 papers in the retrieved subset is a 'retrieved-corpus gap' that must be "
        "classified into one of: 'Evidence absence', 'Methodological limitations', 'Inconsistent findings', "
        "'Population or contextual limitations', 'Validation deficiencies', 'Mechanistic uncertainty', "
        "'Translational deficiencies', 'Measurement/standardization limitations', or 'Theoretical or formal unresolved problems'.\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "rows": ["Method 1", "Method 2", "Method 3", "Method 4"],\n'
        '  "cols": ["Regime/Outcome 1", "Regime/Outcome 2", "Regime/Outcome 3", "Regime/Outcome 4"],\n'
        '  "assignments": [\n'
        '    {"row_idx": 0, "col_idx": 0, "paper_indices": [1, 3], "gap_category": "Validation deficiencies", "why_gap_exists": "...", "proposed_hypothesis": "...", "verification_query": "...", "impact_level": "High|Medium|Exploratory"}\n'
        "  ]\n"
        "}"
    )

    def _validate_gap_out(out_obj: TaskHIGapOutput) -> List[str]:
        errs: List[str] = []
        if len(out_obj.rows) < 3 or len(out_obj.cols) < 3:
            errs.append("Must provide at least 3 rows and 3 cols for the Knowledge Gap Matrix.")
        # Valid numbers are the studies' own reference numbers (off-topic studies are screened out but keep theirs)
        allowed = {p.citation_index for p in papers}
        for item in out_obj.assignments:
            bad_idxs = [i for i in item.paper_indices if i not in allowed]
            if bad_idxs:
                errs.append(f"Gap cell ({item.row_idx}, {item.col_idx}) cites study numbers {bad_idxs} that are not "
                            f"among the provided studies (valid: {_index_label(sorted(allowed))}).")
        return errs

    validated_hi, rec_hi = await execute_validated_llm_task(
        task_id="Task_HI",
        task_name="Tasks H-I: Candidate Research Gap Identification & Verification",
        system_prompt=sys_prompt,
        user_prompt=f"Query: {query}\nDomain: {dom_prof.display_name}\nAvailable HF Artifacts: {hf_names}\nStudies:\n{paper_summaries}",
        response_model=TaskHIGapOutput,
        config=config,
        custom_validator=_validate_gap_out,
        max_retries=2,
        temperature=0.15,
        max_tokens=2200,
        deadline_s=step_budget(config, "gaps"),
    )
    _scoped_audit_trail().append(rec_hi)

    # The model's question-specific grid (when it produced one) replaces the fixed per-domain axes, and its
    # under-covered cells become gap records alongside the evidence-derived ones
    llm_grid = validated_hi if (validated_hi and not rec_hi.degraded_to_deterministic) else None
    return await build_universal_knowledge_gap_matrix(
        query=query,
        papers=papers,
        hf_resources=hf_resources,
        config=config,
        decomposition_plan=decomposition_plan,
        synthesis_package=synthesis_package,
        evidence_matrix=evidence_matrix,
        all_candidate_pool=all_candidate_pool,
        llm_grid=llm_grid,
    )


def build_shared_evidence_base(
    papers: List[PaperRecord],
    gap_matrix: KnowledgeGapMatrix,
    graph: LiteratureGraph,
    decomposition_plan: Optional[QueryDecompositionPlan] = None,
) -> SharedEvidenceBase:
    """
    Phase 7 — Constructs the unified SharedEvidenceBase (Studies, Results, Claims, Gaps, Relationships)
    ensuring all four main report sections trace back to one consistent evidence model.
    """
    studies: List[StudyEntity] = []
    results: List[ResultEntity] = []

    for p in papers:
        sid = p.canonical_study_id or f"study:{p.citation_index}"
        studies.append(
            StudyEntity(
                study_id=sid,
                paper_id=p.id,
                citation_index=p.citation_index,
                title=p.title,
                year=p.year,
                study_design=p.elicit.study_design,
                publication_status=p.version_status,
                provenance=p.elicit.provenance_type,
                methodological_appraisal=f"Sample/Regime status: {p.elicit.sample_reporting_status}",
                limitations=p.elicit.limitations,
            )
        )
        results.append(
            ResultEntity(
                result_id=f"res:{p.citation_index}:1",
                study_id=sid,
                citation_index=p.citation_index,
                outcome_or_measure=p.elicit.dataset_or_system,
                finding_text=p.elicit.key_findings,
                quantitative_estimate=p.elicit.quantitative_result,
                reporting_status=p.elicit.sample_reporting_status,
                provenance=p.elicit.provenance_type,
                source_passage_hint="Abstract" if p.elicit.provenance_type == "abstract_derived" else "Title/Metadata",
            )
        )

    # Use validated ClaimEntity records from Tasks D-G if available, otherwise cluster-based claims
    if _scoped_claims() and all(
        all(1 <= idx <= len(papers) for idx in (c.supporting_indices + c.opposing_indices))
        for c in _scoped_claims()
    ):
        claims: List[ClaimEntity] = [c.model_copy(deep=True) for c in _scoped_claims()]
    else:
        claims = []
        clusters: Dict[str, List[PaperRecord]] = {}
        for p in papers:
            clusters.setdefault(p.cluster_label, []).append(p)

        prim_intent = "methodological_comparison"
        if decomposition_plan and decomposition_plan.question_understanding:
            ui = decomposition_plan.question_understanding.primary_intent
            if ui == "causal_intervention":
                prim_intent = "causal_intervention"
            elif ui == "associational":
                prim_intent = "association"
            elif ui == "predictive":
                prim_intent = "predictive_performance"
            elif ui == "explanatory_mechanistic":
                prim_intent = "mechanistic"
            elif ui == "theoretical":
                prim_intent = "theoretical"

        for c_idx, (cname, cpapers) in enumerate(clusters.items(), start=1):
            sup_idxs = [p.citation_index for p in cpapers if p.stance == "supporting"]
            opp_idxs = [p.citation_index for p in cpapers if p.stance in ("contradicting", "mixed")]
            rep = cpapers[0]
            eff_cat = prim_intent
            if eff_cat == "causal_intervention" and all("observational" in p.elicit.study_design.lower() for p in cpapers):
                eff_cat = "association"
            claims.append(
                ClaimEntity(
                    claim_id=f"claim:{c_idx}",
                    proposition=f"In {cname}, {rep.elicit.study_design} approaches demonstrate: {rep.elicit.key_findings}",
                    claim_category=eff_cat,
                    supporting_indices=sup_idxs or [rep.citation_index],
                    opposing_indices=opp_idxs,
                    applicability_scope=rep.elicit.dataset_or_system,
                    methodological_caveats=rep.elicit.limitations,
                    provenance=rep.elicit.provenance_type,
                    verification_state="passed_with_disclosed_limitations",
                )
            )

    # Convert structured_gaps (or top_opportunities fallback) into GapEntity records
    gaps: List[GapEntity] = []
    if gap_matrix.structured_gaps:
        for sg in gap_matrix.structured_gaps[:6]:
            gaps.append(
                GapEntity(
                    gap_id=sg.gap_id,
                    description=f"{sg.gap_title}: {sg.gap_statement}",
                    gap_category=sg.primary_gap_type,
                    supporting_observations=[sg.current_knowledge, sg.unresolved_issue],
                    related_paper_indices=sg.supporting_study_indices,
                    verification_queries=sg.verification_queries,
                    verification_status="targeted_search_checked"
                    if sg.verification_status in ("Corroborated Within Search Scope", "Partially Addressed")
                    else "verified_in_retrieved_corpus",
                    scope_limitation=sg.search_scope,
                    resolution_approaches=[sg.recommended_study_or_approach],
                )
            )
    else:
        for g_idx, opp in enumerate(gap_matrix.top_opportunities[:6], start=1):
            gaps.append(
                GapEntity(
                    gap_id=f"gap:{g_idx}",
                    description=f"{opp.method_row} × {opp.domain_col}: {opp.why_gap_exists}",
                    gap_category=opp.gap_category,
                    supporting_observations=[opp.why_gap_exists],
                    related_paper_indices=opp.paper_indices,
                    verification_queries=[opp.verification_search_query] if opp.verification_search_query else [],
                    verification_status="verified_in_retrieved_corpus",
                    scope_limitation=opp.scope_disclaimer,
                    resolution_approaches=[opp.proposed_hypothesis],
                )
            )

    # Convert graph edges into typed RelationshipEntity records
    relationships: List[RelationshipEntity] = []
    edge_type_map = {
        "citation": "cites",
        "bibliographic_coupling": "shares_references",
        "co_citation": "co_cited",
        "semantic": "semantically_related",
        "evidence_relationship": "addresses_same_question",
        "model_link": "model_implementation",
    }
    for e in graph.edges[:60]:
        relationships.append(
            RelationshipEntity(
                relationship_id=e.id,
                source_id=e.source,
                target_id=e.target,
                relationship_type=edge_type_map.get(e.edge_type, "semantically_related"),
                provenance=e.provenance,
                weight=e.weight,
                evidence_note=e.label,
            )
        )

    pkg = get_last_synthesis_package()
    atomic_findings = list(pkg.findings) if pkg else []
    evidence_clusters = list(pkg.clusters) if pkg else []

    # Add claim-to-study relationships so Section 4 (Literature Network Map) can trace claim support/challenge
    paper_id_by_idx = {p.citation_index: p.id for p in papers}
    for c in claims:
        for s_idx in c.supporting_indices:
            if s_idx in paper_id_by_idx:
                relationships.append(
                    RelationshipEntity(
                        relationship_id=f"rel:{c.claim_id}:sup:{s_idx}",
                        source_id=paper_id_by_idx[s_idx],
                        target_id=c.claim_id,
                        relationship_type="supports_claim",
                        provenance="lexical_tfidf",
                        weight=0.8,
                        evidence_note=f"Study [{s_idx}] supports {c.claim_id}",
                    )
                )
        for o_idx in c.opposing_indices:
            if o_idx in paper_id_by_idx:
                relationships.append(
                    RelationshipEntity(
                        relationship_id=f"rel:{c.claim_id}:opp:{o_idx}",
                        source_id=paper_id_by_idx[o_idx],
                        target_id=c.claim_id,
                        relationship_type="challenges_claim",
                        provenance="lexical_tfidf",
                        weight=0.8,
                        evidence_note=f"Study [{o_idx}] challenges or qualifies {c.claim_id}",
                    )
                )

    return SharedEvidenceBase(
        studies=studies,
        results=results,
        claims=claims,
        gaps=gaps,
        relationships=relationships,
        task_audit_trail=list(_scoped_audit_trail()),
        atomic_findings=atomic_findings,
        evidence_clusters=evidence_clusters,
    )
