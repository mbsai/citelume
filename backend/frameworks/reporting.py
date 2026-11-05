"""
Builds the review protocol report shown with every result: which source-cited rules were applied, and for
PRISMA-style and scoping reviews the record flow, the exact search strings, and the eligibility criteria used.
"""
from typing import Dict, List, Optional, Sequence

from frameworks.appraisal_routes import detect_biomedical_design
from frameworks.review_types import automation_disclosure, resolve_review_type
from frameworks.rules import select_rules
from frameworks.sources import SOURCES
from models import (
    AppliedRule,
    PaperRecord,
    PrismaFlowCounts,
    QueryDecompositionPlan,
    ReviewProtocolReport,
)

# Search-plan query fields per source label used in source_stats
_SOURCE_QUERY_FIELDS = {
    "PubMed": "pubmed",
    "Semantic Scholar": "semantic_scholar",
    "arXiv": "arxiv",
    "bioRxiv": "biorxiv_medrxiv",
}


def _search_strategies(plan: QueryDecompositionPlan, enabled_sources: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for source in enabled_sources:
        field = _SOURCE_QUERY_FIELDS.get(source)
        if not field:
            continue
        queries: List[str] = []
        seen = set()
        for facet in plan.search_facets:
            q = getattr(facet.queries, field, None)
            for item in (q if isinstance(q, list) else [q]):
                if item and item not in seen:
                    seen.add(item)
                    queries.append(f"[{facet.facet_name}] {item}")
        out[source] = queries
    if plan.search_plan and plan.search_plan.adaptive_expansion_triggered:
        out["Adaptive expansion"] = ["A broadened query on the core entities was added because the first round "
                                     "returned few records."]
    return out


def build_review_protocol_report(
    framework_id: str,
    organization_id: str,
    primary_domain: str,
    plan: QueryDecompositionPlan,
    included_papers: Sequence[PaperRecord],
    identified_by_source: Dict[str, int],
    identified_total: int,
    unique_records: int,
    enabled_sources: Sequence[str],
    year_from: Optional[int],
    year_to: Optional[int],
    constraint_excluded: int = 0,
) -> ReviewProtocolReport:
    framework, organization = resolve_review_type(framework_id, organization_id)
    # Designs seen in the included abstracts plus the study-method profiles the planner detected
    designs = sorted(
        {d for d in (detect_biomedical_design(p.abstract or "") for p in included_papers) if d}
        | set(plan.question_understanding.active_methodological_profiles if plan.question_understanding else [])
    )
    rules = select_rules(primary_domain, designs, framework.framework_id)
    applied = [
        AppliedRule(
            rule_id=r.rule_id,
            text=r.text,
            stage=r.stage,
            sources=[SOURCES[s].short_name for s in r.source_ids],
            citations=[SOURCES[s].citation for s in r.source_ids if SOURCES[s].kind not in ("field_convention", "tool_integrity")],
            enforcement=r.enforcement,
        )
        for r in rules
    ]
    report = ReviewProtocolReport(
        framework_id=framework.framework_id,
        framework_label=framework.label,
        organization_id=organization.organization_id,
        organization_label=organization.label,
        framework_citations=[SOURCES[s].citation for s in framework.source_ids + organization.source_ids
                             if SOURCES[s].kind != "field_convention"],
        rules_applied=applied,
    )
    if not framework.includes_flow_diagram:
        return report

    report.disclosure = automation_disclosure(framework)
    report.flow = PrismaFlowCounts(
        identified_by_source={k: v for k, v in identified_by_source.items() if v and k in enabled_sources},
        identified_total=identified_total,
        duplicates_removed=max(0, identified_total - unique_records),
        screened=unique_records,
        excluded_at_screening=max(0, unique_records - len(included_papers)),
        included=len(included_papers),
    )
    report.search_strategies = _search_strategies(plan, enabled_sources)
    constraints = plan.question_understanding.concepts.required_constraints if plan.question_understanding else []
    years = (f"{year_from or 'earliest'}-{year_to or 'latest'}" if (year_from or year_to) else "no date limit")
    report.eligibility_criteria = [
        f"Sources searched: {', '.join(enabled_sources)}.",
        f"Publication years: {years}.",
        "Study designs: all designs eligible; design is recorded and appraised per study.",
    ] + ([
        "Place / population: " + "; ".join(c.name for c in constraints) + f". {constraint_excluded} record(s) whose "
        "title and abstract did not mention them were excluded at screening.",
    ] if constraints else []) + [
        f"Relevance: {sum(1 for p in included_papers if p.relevance == 'off_topic')} included record(s) were judged not "
        "to address the question and were left out of the synthesis.",
        "Language and publication status: as indexed by each source; preprints included and flagged.",
        "Screening: records after duplicate removal were ranked on relevance to the question, sub-question "
        "coverage, methodological detail, independence, preservation of contradicting evidence and recency; the "
        f"top {len(included_papers)} were included. Screening was automated and single-pass, from titles and abstracts.",
    ]
    return report
