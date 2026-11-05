"""
Review frameworks (how the review is conducted and reported) and body organisations (how findings are arranged).

The two are independent: a PRISMA-style systematic review can present its results thematically or methodologically,
and a narrative review can be chronological. Frameworks follow Grant & Booth's typology; organisations follow the
standard literature-review organising approaches taught in university writing guides.
"""
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

ReviewFrameworkId = Literal["narrative", "systematic_prisma", "scoping"]
OrganizationId = Literal["thematic", "chronological", "methodological", "theoretical"]

AUTOMATION_DISCLOSURE = (
    "This review was produced by an automated, single-pass pipeline that screens and appraises studies from titles, "
    "abstracts and metadata. It follows the reporting structure of {framework} but is not a registered systematic "
    "review: there was no pre-registered protocol, no dual independent screening, and no full-text appraisal."
)


@dataclass(frozen=True)
class ReviewFramework:
    framework_id: ReviewFrameworkId
    label: str
    description: str
    source_ids: Tuple[str, ...]
    includes_flow_diagram: bool
    includes_risk_of_bias: bool
    section_plan: Tuple[str, ...]


@dataclass(frozen=True)
class Organization:
    organization_id: OrganizationId
    label: str
    body_heading: str
    prompt_instruction: str
    source_ids: Tuple[str, ...] = ("lit_review_org",)


REVIEW_FRAMEWORKS: Dict[str, ReviewFramework] = {
    "narrative": ReviewFramework(
        "narrative", "Narrative review",
        "Expert synthesis of the literature without a protocol-driven search; quality judged against SANRA.",
        ("sanra", "grant_booth"), includes_flow_diagram=False, includes_risk_of_bias=True,
        section_plan=("C. Conflicting Evidence and Alternative Interpretations",
                      "D. Reliability and Limitations of the Evidence", "E. Integrated Scientific Conclusion"),
    ),
    "systematic_prisma": ReviewFramework(
        "systematic_prisma", "Systematic-style review (PRISMA 2020 reporting)",
        "Documented search, screening flow, design-matched appraisal and certainty of evidence, reported to PRISMA 2020.",
        ("prisma2020", "grade", "grant_booth"), includes_flow_diagram=True, includes_risk_of_bias=True,
        section_plan=("C. Heterogeneity and Conflicting Evidence", "D. Risk of Bias and Certainty of Evidence",
                      "E. Conclusions and Limitations of this Review"),
    ),
    "scoping": ReviewFramework(
        "scoping", "Scoping review (PRISMA-ScR reporting)",
        "Maps what evidence exists, where and how it was studied, and where gaps remain; appraisal is optional.",
        ("prisma_scr", "grant_booth"), includes_flow_diagram=True, includes_risk_of_bias=False,
        section_plan=("C. Characteristics of the Evidence Base", "D. Evidence Gaps",
                      "E. Conclusions and Implications for Research"),
    ),
}

ORGANIZATIONS: Dict[str, Organization] = {
    "thematic": Organization(
        "thematic", "Thematic", "B. Thematic Evidence Synthesis",
        "Organise section B by theme: one paragraph per recurring finding or issue, synthesising across studies.",
    ),
    "chronological": Organization(
        "chronological", "Chronological", "B. Chronological Development of the Evidence",
        "Organise section B chronologically: describe how findings and methods developed over time, period by "
        "period, naming the years of each period. Explain what changed, not just a list of papers by date.",
    ),
    "methodological": Organization(
        "methodological", "Methodological", "B. Evidence by Study Method",
        "Organise section B by research method: group studies by design or approach, and compare what each kind "
        "of method found and its strengths and weaknesses.",
    ),
    "theoretical": Organization(
        "theoretical", "Theoretical", "B. Theoretical Frameworks and Models",
        "Organise section B by theory: group studies by the theory, model or conceptual framework they use or "
        "test, and compare how well the evidence supports each.",
    ),
}

DEFAULT_FRAMEWORK: ReviewFrameworkId = "narrative"
DEFAULT_ORGANIZATION: OrganizationId = "thematic"


def resolve_review_type(framework_id: str, organization_id: str) -> Tuple[ReviewFramework, Organization]:
    return (
        REVIEW_FRAMEWORKS.get(framework_id or DEFAULT_FRAMEWORK, REVIEW_FRAMEWORKS[DEFAULT_FRAMEWORK]),
        ORGANIZATIONS.get(organization_id or DEFAULT_ORGANIZATION, ORGANIZATIONS[DEFAULT_ORGANIZATION]),
    )


def automation_disclosure(framework: ReviewFramework) -> str:
    return AUTOMATION_DISCLOSURE.format(framework=framework.label)
