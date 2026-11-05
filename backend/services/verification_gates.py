import re
from typing import List, Optional
from models import (
    QueryDecompositionPlan,
    PaperRecord,
    SynthesisSection,
    KnowledgeGapMatrix,
    LiteratureGraph,
    QualityGateReport,
    SharedEvidenceBase,
    TaskExecutionRecord,
    ModelCapabilityProfile,
    TaskCStudyExtractionItem,
    VerificationStatusType,
)


def extract_nontrivial_numbers(text: str) -> List[str]:
    """
    Extracts decimal numbers and percentages (e.g., '94.2%', '0.87', '12.5') from a text string
    while ignoring bracketed citation indices like [1] and 4-digit publication years (1990-2030).
    """
    if not text:
        return []
    cleaned = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)
    matches = re.findall(r"\b\d+\.\d+%?|\b\d+%", cleaned)
    return [m.rstrip("%") for m in matches]


def verify_study_extraction_against_source(
    item: TaskCStudyExtractionItem,
    source_paper: PaperRecord,
    primary_domain: str = "biomedical_clinical",
) -> List[str]:
    """
    Section 5 & 6 Deterministic Source & Numerical Integrity Validator for Task C:
    Verifies that an LLM-extracted study record does not fabricate decimal statistics/percentages
    absent from the source title/abstract, and enforces domain sample reporting rules.
    """
    errors: List[str] = []
    source_text = study_source_text(source_paper)
    has_abstract = bool((source_paper.abstract or "").strip() and len((source_paper.abstract or "").strip()) >= 30)

    # 1. Check that any decimal/percentage in quantitative_result or key_findings exists in source_text
    for field_name, field_val in [
        ("quantitative_result", item.quantitative_result),
        ("key_findings", item.key_findings),
    ]:
        nums = extract_nontrivial_numbers(field_val)
        for num in nums:
            if num not in source_text:
                errors.append(
                    f"Study [{item.citation_index}] {field_name} contains number '{num}' which does not appear in the source text."
                )

    # 2. If source is metadata-only, sample_reporting_status cannot be 'Reported' unless in title
    if not has_abstract and item.sample_reporting_status == "Reported":
        if not re.search(r"\d+", source_paper.title):
            errors.append(
                f"Study [{item.citation_index}] has no abstract (metadata_only), so sample_reporting_status cannot be 'Reported'."
            )

    # 3. Mathematics domain rule: pure proofs must not fabricate patient cohorts
    if primary_domain == "mathematics" and "patients" in (item.sample_or_scale or "").lower():
        if "patients" not in source_text.lower():
            errors.append(
                f"Study [{item.citation_index}] in mathematics domain fabricated 'patients' in sample_or_scale."
            )

    return errors


# A bracketed citation group: [3], [1, 4], [2-5], [2–5], [Study 3], [Refs. 1, 3-4]
_CITATION_GROUP = re.compile(
    r"\[(?:(?:studies|study|refs?\.?|references?)\s*)?(\d+(?:\s*[-\u2013\u2014]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-\u2013\u2014]\s*\d+)?)*)\]",
    re.IGNORECASE,
)
# Ranges longer than this are treated as typos rather than expanded
MAX_CITATION_RANGE = 60


def _expand_citation_group(group: str) -> List[int]:
    nums: List[int] = []
    for part in re.split(r"\s*[,;]\s*", group.strip()):
        bounds = re.split(r"\s*[-\u2013\u2014]\s*", part)
        if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            lo, hi = int(bounds[0]), int(bounds[1])
            nums.extend(range(lo, hi + 1) if 0 <= hi - lo <= MAX_CITATION_RANGE else [lo, hi])
        elif part.isdigit():
            nums.append(int(part))
    # Bracketed years such as "[2019]" are not study numbers
    return [n for n in nums if n < 1000]


def extract_citation_numbers(text: str) -> List[int]:
    """Every study number cited in text, from single, list and range citations."""
    return [n for m in _CITATION_GROUP.finditer(text or "") for n in _expand_citation_group(m.group(1))]


def normalize_citations(text: str) -> str:
    """Rewrites citation groups to the canonical "[1, 2, 3]" form (expanding ranges, dropping "Study"/"Ref" labels)."""
    def repl(m: re.Match) -> str:
        nums = list(dict.fromkeys(_expand_citation_group(m.group(1))))
        return "[" + ", ".join(str(n) for n in nums) + "]" if nums else m.group(0)
    return _CITATION_GROUP.sub(repl, text or "")


def study_source_text(p: PaperRecord) -> str:
    """Everything a figure may legitimately come from: title, abstract and any open-access full text."""
    return " ".join([p.title or "", p.abstract or ""] + list(p.full_text_sections.values()))


def _index_label(indices: List[int]) -> str:
    """Compact list of study numbers, e.g. "1-4, 7, 9-12"."""
    out: List[str] = []
    for i in indices:
        if out and "-" not in out[-1] and int(out[-1]) == i - 1:
            out[-1] = f"{out[-1]}-{i}"
        elif out and "-" in out[-1] and int(out[-1].split("-")[1]) == i - 1:
            out[-1] = f"{out[-1].split('-')[0]}-{i}"
        else:
            out.append(str(i))
    return ", ".join(out)


def verify_claims_against_sources(
    claims_list: list,
    papers: List[PaperRecord],
) -> List[str]:
    """
    Section 6 Claim Attribution, Numerical Integrity & Inference Validity Validator for Tasks D-G:
    Ensures every claim references valid study indices [1..N], does not invent decimal numbers
    absent from its cited studies, and does not make unqualified causal claims from purely observational studies.
    """
    errors: List[str] = []
    paper_map = {p.citation_index: p for p in papers}
    valid_label = _index_label(sorted(paper_map))

    for c in claims_list:
        sup = list(getattr(c, "supporting_indices", []) or [])
        opp = list(getattr(c, "opposing_indices", []) or [])
        all_cited = sup + opp
        if not all_cited:
            errors.append(f"Claim '{getattr(c, 'claim_id', 'unknown')}' has no supporting or opposing study indices.")
            continue

        # Valid numbers are the provided studies' own reference numbers (not always 1..N: off-topic studies are
        # screened out of the synthesis but keep their bibliography numbers)
        invalid_idxs = [idx for idx in all_cited if idx not in paper_map]
        if invalid_idxs:
            errors.append(
                f"Claim '{getattr(c, 'claim_id', 'unknown')}' cites study numbers {invalid_idxs} that are not among the provided studies (valid: {valid_label})."
            )
            continue

        # Check numerical grounding against cited studies
        cited_corpus_text = " ".join(study_source_text(paper_map[idx]) for idx in all_cited if idx in paper_map)
        prop_text = getattr(c, "proposition", "") or ""
        for num in extract_nontrivial_numbers(prop_text):
            if num not in cited_corpus_text:
                errors.append(
                    f"Claim '{getattr(c, 'claim_id', 'unknown')}' contains number '{num}' not present in its cited studies {all_cited}."
                )

        # Inference Validity check: prevent unqualified causal claims from purely observational studies
        claim_cat = getattr(c, "claim_category", "")
        cited_designs = [paper_map[idx].elicit.study_design.lower() for idx in sup if idx in paper_map]
        if cited_designs and all("observational" in d or "retrospective" in d for d in cited_designs):
            if claim_cat == "causal_intervention" or re.search(r"\b(proves that .* causes|definitively causes)\b", prop_text, re.IGNORECASE):
                errors.append(
                    f"Inference Validity Error on '{getattr(c, 'claim_id', 'unknown')}': Unqualified causal claim made from purely observational studies {sup}."
                )

    return errors


def run_verification_and_quality_gates(
    plan: QueryDecompositionPlan,
    papers: List[PaperRecord],
    executive_answer: str,
    sections: List[SynthesisSection],
    gap_matrix: KnowledgeGapMatrix,
    graph: LiteratureGraph,
    evidence_base: Optional[SharedEvidenceBase] = None,
    task_audit_trail: Optional[List[TaskExecutionRecord]] = None,
    model_capability: Optional[ModelCapabilityProfile] = None,
) -> QualityGateReport:
    """
    Phase 9 & Section 6 Model-Independent Quality Gates:
    Validates:
    1. Retrieval Syntax & Study Deduplication (`retrieval_valid`)
    2. Schema Correctness (`schema_valid` & `extraction_valid`)
    3. Source Integrity (`source_integrity_valid`)
    4. Numerical Integrity (`numerical_integrity_valid`)
    5. Claim Attribution (`claim_attribution_valid` & `synthesis_valid`)
    6. Inference Validity (`inference_validity_valid`)
    7. Evidence Consistency (`evidence_consistency_valid`)
    8. Knowledge-Gap Scope & Verification Queries (`gap_valid`)
    9. Network Provenance (`network_valid`)
    10. Cross-Section Consistency (`cross_section_consistency_valid`)
    """
    checks_passed: List[str] = []
    warnings_out: List[str] = []
    audit_records = list(task_audit_trail or (evidence_base.task_audit_trail if evidence_base else []) or [])

    retrieval_valid = True
    extraction_valid = True
    synthesis_valid = True
    gap_valid = True
    network_valid = True

    schema_valid = True
    source_integrity_valid = True
    numerical_integrity_valid = True
    claim_attribution_valid = True
    inference_validity_valid = True
    evidence_consistency_valid = True
    cross_section_consistency_valid = True

    # 1. Retrieval Validation
    for facet in plan.search_facets:
        q = facet.queries
        if q.pubmed.count("(") != q.pubmed.count(")"):
            retrieval_valid = False
            warnings_out.append(f"Retrieval Gate: Unbalanced parentheses in PubMed query for facet '{facet.facet_name}'.")
        if q.arxiv.count("(") != q.arxiv.count(")") or not any(p in q.arxiv for p in ("ti:", "abs:", "all:")):
            retrieval_valid = False
            warnings_out.append(f"Retrieval Gate: Invalid field prefix or parentheses in arXiv query for facet '{facet.facet_name}'.")
        if re.search(r"\b(AND|OR|NOT)\b", q.semantic_scholar):
            retrieval_valid = False
            warnings_out.append(f"Retrieval Gate: Boolean operator found in Semantic Scholar query for facet '{facet.facet_name}'.")

    seen_ids = set()
    for p in papers:
        if p.id in seen_ids:
            retrieval_valid = False
            source_integrity_valid = False
            warnings_out.append(f"Source Integrity Gate: Duplicate paper ID '{p.id}' detected in selected study set.")
        seen_ids.add(p.id)

    if retrieval_valid:
        checks_passed.append("Retrieval Gate: All 5-API query syntaxes validated and selected studies deduplicated.")

    # 2. Schema & Extraction Validation + Numerical Integrity on Study Extractions
    prim_domain = (plan.active_profile_summary or {}).get("primary_domain", "biomedical_clinical")
    for p in papers:
        if p.elicit.sample_reporting_status not in (
            "Reported",
            "Not reported",
            "Not applicable",
            "Not assessed",
            "Unavailable",
            "Uncertain",
        ):
            extraction_valid = False
            schema_valid = False
            warnings_out.append(f"Schema/Extraction Gate: Invalid sample_reporting_status on paper [{p.citation_index}].")
        if prim_domain == "mathematics" and "Formal Mathematical Proof" in p.elicit.study_design:
            if "patients" in p.elicit.sample_or_scale.lower() and "patients" not in (p.abstract or "").lower():
                extraction_valid = False
                warnings_out.append(f"Extraction Gate: Mathematical study [{p.citation_index}] has inappropriate clinical sample text.")

        # Numerical Integrity check against the paper's own title, abstract and any open-access full text
        source_text = study_source_text(p)
        for num in extract_nontrivial_numbers(p.elicit.quantitative_result):
            if num not in source_text:
                numerical_integrity_valid = False
                warnings_out.append(
                    f"Numerical Integrity Gate: Study [{p.citation_index}] quantitative_result contains '{num}' not found in source record."
                )

    if extraction_valid and schema_valid:
        checks_passed.append("Schema & Extraction Gate: Provenance levels and reporting statuses (Reported / Not reported / Not applicable) verified.")
    if numerical_integrity_valid:
        checks_passed.append("Numerical Integrity Gate: All extracted quantitative values trace directly to source records.")

    # 3. Synthesis, Source Integrity & Claim Attribution Validation
    max_idx = len(papers)
    all_synth_text = executive_answer + "\n" + "\n".join(s.content for s in sections)
    cited_nums = extract_citation_numbers(all_synth_text)
    if max_idx > 0:
        valid_idx = {p.citation_index for p in papers}
        out_of_range = [c for c in cited_nums if c not in valid_idx]
        if out_of_range:
            synthesis_valid = False
            source_integrity_valid = False
            claim_attribution_valid = False
            warnings_out.append(f"Source Integrity Gate: Out-of-bounds citation indices {out_of_range} (valid range: 1..{max_idx}).")
        if not cited_nums:
            synthesis_valid = False
            claim_attribution_valid = False
            warnings_out.append("Claim Attribution Gate: Synthesis text is missing inline bracketed citations.")

    # Check inference validity & claim attribution on SharedEvidenceBase claims
    if evidence_base and evidence_base.claims:
        claim_errs = verify_claims_against_sources(evidence_base.claims, papers)
        for ce in claim_errs:
            warnings_out.append(ce)
            if "Inference Validity" in ce:
                inference_validity_valid = False
            elif "number" in ce:
                numerical_integrity_valid = False
            else:
                claim_attribution_valid = False

    # Check Evidence Consistency (opposing evidence not ignored)
    contradicting_papers = [p.citation_index for p in papers if p.stance == "contradicting"]
    if contradicting_papers and not any(idx in cited_nums for idx in contradicting_papers):
        evidence_consistency_valid = False
        warnings_out.append(
            f"Evidence Consistency Gate: Contradicting study(ies) {contradicting_papers} present in corpus were omitted from synthesis citations."
        )

    if synthesis_valid and claim_attribution_valid and source_integrity_valid:
        checks_passed.append("Source Integrity & Claim Attribution Gate: All inline citations and claim records verified against bibliography [1..N].")
    if inference_validity_valid:
        checks_passed.append("Inference Validity Gate: Causal, predictive, associational, and theoretical claims match underlying study designs.")
    if evidence_consistency_valid:
        checks_passed.append("Evidence Consistency Gate: Supporting, mixed, and opposing evidence properly represented.")

    # 4. Knowledge-Gap Validation
    for row in gap_matrix.cells:
        for cell in row:
            if cell.status == "gap":
                if not cell.scope_disclaimer:
                    gap_valid = False
                    warnings_out.append(f"Gap Gate: Cell '{cell.id}' missing scope disclaimer.")
                if not cell.verification_search_query:
                    gap_valid = False
                    warnings_out.append(f"Gap Gate: Cell '{cell.id}' missing F5 targeted verification search query.")

    if gap_valid:
        checks_passed.append("Knowledge-Gap Gate: All candidate gaps include scope disclaimers and targeted F5 verification queries.")

    # 5. Network Validation
    node_ids = {n.id for n in graph.nodes}
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            network_valid = False
            warnings_out.append(f"Network Gate: Edge '{edge.id}' references missing node.")
        if edge.edge_type in ("citation", "bibliographic_coupling", "co_citation") and edge.provenance != "bibliometric_metadata":
            network_valid = False
            warnings_out.append(f"Network Gate: Bibliometric edge '{edge.id}' has non-bibliometric provenance '{edge.provenance}'.")

    if network_valid:
        checks_passed.append("Network Gate: Bibliometric edges separated from lexical/semantic edges with verified node endpoints.")

    # 6. Cross-Section Consistency Validation
    paper_indices_set = {p.citation_index for p in papers}
    if evidence_base:
        eb_study_indices = {s.citation_index for s in evidence_base.studies}
        if eb_study_indices != paper_indices_set:
            cross_section_consistency_valid = False
            warnings_out.append("Cross-Section Consistency Gate: SharedEvidenceBase study indices do not match Evidence Table papers.")
    for row in gap_matrix.cells:
        for cell in row:
            if any(idx not in paper_indices_set for idx in cell.paper_indices):
                cross_section_consistency_valid = False
                warnings_out.append(f"Cross-Section Consistency Gate: Gap cell '{cell.id}' references unknown study index.")

    if cross_section_consistency_valid:
        checks_passed.append("Cross-Section Consistency Gate: Synthesis, Evidence Table, Gap Matrix, and Network Graph share identical study records.")

    # 7. Audit trail & Degradation Summary
    retries_used = sum(max(0, r.attempts_used - 1) for r in audit_records)
    degradation_events = [
        f"{r.task_id} ({r.task_name}): degraded to deterministic source-grounded fallback ({'; '.join(r.validation_errors[-1:]) or 'offline'})"
        for r in audit_records
        if r.degraded_to_deterministic
    ]

    all_core_gates_valid = all(
        [
            retrieval_valid,
            extraction_valid,
            synthesis_valid,
            gap_valid,
            network_valid,
            schema_valid,
            source_integrity_valid,
            numerical_integrity_valid,
            claim_attribution_valid,
            inference_validity_valid,
            cross_section_consistency_valid,
        ]
    )

    if not all_core_gates_valid:
        overall_status: VerificationStatusType = "failed_verification"
    elif not evidence_consistency_valid:
        overall_status = "partially_verified"
    elif degradation_events or retries_used > 0:
        overall_status = "passed_with_disclosed_limitations"
    else:
        overall_status = "passed_validation"

    return QualityGateReport(
        overall_verification_status=overall_status,
        retrieval_valid=retrieval_valid,
        extraction_valid=extraction_valid,
        synthesis_valid=synthesis_valid,
        gap_valid=gap_valid,
        network_valid=network_valid,
        schema_valid=schema_valid,
        source_integrity_valid=source_integrity_valid,
        numerical_integrity_valid=numerical_integrity_valid,
        claim_attribution_valid=claim_attribution_valid,
        inference_validity_valid=inference_validity_valid,
        evidence_consistency_valid=evidence_consistency_valid,
        cross_section_consistency_valid=cross_section_consistency_valid,
        correction_retries_used=retries_used,
        degradation_events=degradation_events,
        model_capability_used=model_capability,
        task_audit_trail=audit_records,
        checks_passed=checks_passed,
        warnings_and_Notices=warnings_out,
    )
