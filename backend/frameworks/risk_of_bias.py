"""
Risk-of-bias tools for a systematic-style (PRISMA) review, matched to each study's design.

The model is given the tool's domains for each study and judges each one as Low, Some concerns, High or No
information, from the abstract (or the full text when it was retrieved). The overall judgement is not left to the
model: it is computed from the domains by OVERALL_RULE, which follows the RoB 2 algorithm.

Domains are those of the published tools, shortened to fit one judgement each. They inform, but do not replace, an
assessment by two reviewers using the full signalling questions of each tool.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from frameworks.sources import SOURCES

JUDGEMENTS = ("Low", "Some concerns", "High", "No information")


@dataclass(frozen=True)
class RiskOfBiasTool:
    tool_id: str
    label: str
    domains: Tuple[str, ...]
    source_ids: Tuple[str, ...]

    @property
    def citations(self) -> List[str]:
        return [SOURCES[s].citation for s in self.source_ids if s in SOURCES]


TOOLS: Dict[str, RiskOfBiasTool] = {
    "rob2": RiskOfBiasTool(
        "rob2", "RoB 2 (randomised trials)",
        ("Randomisation process", "Deviations from intended interventions", "Missing outcome data",
         "Measurement of the outcome", "Selection of the reported result"),
        ("rob2",)),
    "robins": RiskOfBiasTool(
        "robins", "ROBINS-I / ROBINS-E (non-randomised studies)",
        ("Confounding", "Selection of participants", "Classification of the intervention or exposure",
         "Deviations from intended interventions or changes in exposure", "Missing data",
         "Measurement of the outcome", "Selection of the reported result"),
        ("robins_i_v2", "robins_e")),
    "jbi_cross_sectional": RiskOfBiasTool(
        "jbi_cross_sectional", "JBI checklist (analytical cross-sectional studies)",
        ("Sample selection and setting", "Measurement of the exposure or condition", "Confounding addressed",
         "Measurement of the outcome", "Statistical analysis"),
        ("jbi_checklists",)),
    "syrcle": RiskOfBiasTool(
        "syrcle", "SYRCLE (animal and laboratory studies)",
        ("Allocation and randomisation", "Blinding", "Incomplete outcome data", "Selective outcome reporting",
         "Controls and replication"),
        ("syrcle",)),
    "jbi_case_report": RiskOfBiasTool(
        "jbi_case_report", "JBI checklist (case reports and case series)",
        ("Patient characteristics described", "Condition and intervention described", "Outcome assessment",
         "Adverse events and completeness of reporting"),
        ("jbi_checklists",)),
    "casp_qualitative": RiskOfBiasTool(
        "casp_qualitative", "CASP (qualitative studies)",
        ("Appropriate design and recruitment", "Data collection", "Researcher reflexivity", "Rigour of analysis"),
        ("casp_qualitative",)),
    "technical_validity": RiskOfBiasTool(
        "technical_validity", "Validity assessment (computational and technical studies, after REFORMS)",
        ("Data or benchmark appropriateness", "Baselines and comparisons", "Evaluation protocol and leakage",
         "Reproducibility (code and data)", "Reporting of uncertainty"),
        ("reforms", "ml_leakage")),
    "amstar2": RiskOfBiasTool(
        "amstar2", "AMSTAR 2-informed (reviews)",
        ("Protocol and eligibility criteria", "Search and study selection", "Risk-of-bias assessment of included studies",
         "Synthesis methods and heterogeneity"),
        ("amstar2",)),
    "generic": RiskOfBiasTool(
        "generic", "Design-generic appraisal",
        ("Selection of participants or samples", "Measurement of exposure or intervention", "Confounding and controls",
         "Missing data", "Selection of the reported result"),
        ("rob2", "robins_e")),
}

# Study type (services/paper_selection.STUDY_TYPE_LABELS) -> tool
TOOL_FOR_STUDY_TYPE = {
    "rct": "rob2",
    "controlled_observational": "robins",
    "descriptive_observational": "jbi_cross_sectional",
    "experimental_lab": "syrcle",
    "case_report": "jbi_case_report",
    "qualitative": "casp_qualitative",
    "computational": "technical_validity",
    "theoretical": "technical_validity",
    "evidence_synthesis": "amstar2",
    "review": "amstar2",
}

OVERALL_RULE = (
    "Overall judgement, computed from the domains (following the RoB 2 algorithm): High if any domain is High; "
    "otherwise Some concerns if any domain has Some concerns, or if one or two domains have no information; "
    "No information if most domains have no information; otherwise Low."
)


def tool_for(study_type: str) -> RiskOfBiasTool:
    return TOOLS[TOOL_FOR_STUDY_TYPE.get(study_type or "", "generic")]


def normalise_judgement(value: str) -> str:
    v = (value or "").strip().lower()
    if v.startswith("low"):
        return "Low"
    if v.startswith("high") or v.startswith("serious") or v.startswith("critical"):
        return "High"
    if "concern" in v or v.startswith("moderate") or v.startswith("some") or v == "unclear risk":
        return "Some concerns"
    return "No information"


def overall_judgement(judgements: Sequence[str]) -> str:
    js = [normalise_judgement(j) for j in judgements]
    if not js:
        return "No information"
    if "High" in js:
        return "High"
    missing = js.count("No information")
    if missing > len(js) / 2:
        return "No information"
    if "Some concerns" in js or missing:
        return "Some concerns"
    return "Low"
