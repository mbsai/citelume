"""
Single registry of the methodology rules the tool applies, each tied to its primary source(s).

A rule's `text` is the short wording placed in LLM prompts and shown in reports. `enforcement` names the deterministic
check that enforces it regardless of model (see `GATES`), or "prompt" when only the prompt carries it. Rules are
selected per search by domain, detected study designs, review type and pipeline stage, so prompts only carry what
applies (which keeps them small enough for 3B-9B local models).
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from frameworks.sources import SOURCES

Stage = Literal["appraisal", "synthesis", "reporting", "gap"]

# "*" applies to every domain / design / review type
ANY = ("*",)

# Deterministic checks that enforce rules independently of the LLM, and where they live
GATES: Dict[str, str] = {
    "causal_overclaim_gate": "services/claim_synthesis.py (_OVERCLAIM_CAUSAL_RE) and services/verification_gates.py",
    "ungrounded_number_gate": "services/verification_gates.py and the Task J validator in services/synthesis_engine.py",
    "omitted_contradiction_gate": "Task J validator in services/synthesis_engine.py",
    "sample_policy_gate": "verify_study_extraction_against_source in services/verification_gates.py",
    "citation_range_gate": "Task C / Task J validators in services/synthesis_engine.py",
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    text: str
    stage: Stage
    source_ids: Tuple[str, ...]
    source_locator: str = ""
    domains: Tuple[str, ...] = ANY
    study_designs: Tuple[str, ...] = ANY
    review_types: Tuple[str, ...] = ANY
    enforcement: str = "prompt"

    def applies(self, domain: Optional[str], designs: Sequence[str], review_type: Optional[str]) -> bool:
        dom_ok = "*" in self.domains or (domain is not None and domain in self.domains)
        design_ok = "*" in self.study_designs or "*" in designs or any(d in self.study_designs for d in designs)
        review_ok = "*" in self.review_types or (review_type is not None and review_type in self.review_types)
        return dom_ok and design_ok and review_ok

    def citation_label(self) -> str:
        return "; ".join(SOURCES[s].short_name for s in self.source_ids)


BIOMED = ("biomedical_clinical", "interdisciplinary_biomedical_ai")
CS = ("computer_science_ai", "interdisciplinary_biomedical_ai")

RULES: List[Rule] = [
    # ---------------------------------------------------------------- universal synthesis rules
    Rule("U-causal", "Do not state causal effects from observational, correlational or predictive evidence; use "
         "associational wording.", "synthesis", ("grade",), "GRADE: evidence from observational studies starts at low "
         "certainty", enforcement="causal_overclaim_gate"),
    Rule("U-numbers", "Report only numbers that appear in the cited study records; never estimate or pool values.",
         "synthesis", ("tool_integrity",), enforcement="ungrounded_number_gate"),
    Rule("U-direction", "Summarise agreement by direction of effect (supporting / mixed / contradicting), not by "
         "statistical significance, and note that this ignores effect size and precision.", "synthesis", ("swim",),
         "SWiM: synthesis method - vote counting based on direction of effect"),
    Rule("U-heterogeneity", "Explain how studies differ (population, setting, design, outcome measures) before "
         "drawing conclusions across them.", "synthesis", ("swim",), "SWiM: investigation of heterogeneity"),
    Rule("U-opposing", "Cite contradicting or null studies alongside supporting ones.", "synthesis", ("grade",),
         "GRADE: inconsistency", enforcement="omitted_contradiction_gate"),
    Rule("U-certainty", "State how certain the evidence is, considering risk of bias, inconsistency, indirectness, "
         "imprecision and publication bias.", "synthesis", ("grade",), "GRADE: five reasons to rate down certainty"),
    Rule("U-citations", "Every claim must cite study numbers [n] that exist in the provided records.", "reporting",
         ("tool_integrity",), enforcement="citation_range_gate"),
    Rule("U-preprint", "Flag preprints as not peer reviewed.", "reporting", ("tool_integrity",)),
    # ---------------------------------------------------------------- biomedical: design-matched appraisal
    Rule("B-rct", "Randomised trials: check reporting on the RoB 2 domains - randomisation process, deviations from "
         "intended interventions, missing outcome data, outcome measurement, selection of the reported result.",
         "appraisal", ("rob2",), "RoB 2: five bias domains", BIOMED, ("rct",)),
    Rule("B-nrsi", "Non-randomised intervention studies: confounding is the main threat; check how it was "
         "controlled.", "appraisal", ("robins_i_v2",), "ROBINS-I V2: bias due to confounding", BIOMED,
         ("nrsi_intervention",)),
    Rule("B-exposure", "Observational exposure studies: check confounding control, participant selection and "
         "exposure classification.", "appraisal", ("robins_e",), "ROBINS-E: bias domains", BIOMED,
         ("observational_exposure",)),
    Rule("B-prediction", "Prediction models: check participants and data sources, predictors, outcome and analysis; "
         "prefer external validation over internal performance.", "appraisal", ("probast_ai",),
         "PROBAST+AI: four domains (development and evaluation)", BIOMED, ("prediction_model",)),
    Rule("B-prognostic", "Prognostic factor studies: check participation, attrition, factor and outcome measurement, "
         "confounding, and analysis.", "appraisal", ("quips",), "QUIPS: six bias domains", BIOMED,
         ("prognostic_factor",)),
    Rule("B-diagnostic", "Diagnostic accuracy studies: check participants, index test, target condition and "
         "analysis.", "appraisal", ("quadas3",), "QUADAS-3: four risk-of-bias domains", BIOMED,
         ("diagnostic_accuracy",)),
    Rule("B-animal", "Animal studies: check randomisation, blinding and attrition (bias) and whether the ARRIVE "
         "Essential 10 items are reported.", "appraisal", ("syrcle", "arrive2"),
         "SYRCLE: 10 items over 6 bias types; ARRIVE 2.0: Essential 10", BIOMED, ("animal",)),
    Rule("B-indirectness", "Keep in vitro, animal and human findings separate; do not extrapolate preclinical "
         "results to patients.", "synthesis", ("grade",), "GRADE: indirectness", BIOMED),
    Rule("B-omics", "High-throughput studies: check multiple-testing correction, batch effects and orthogonal "
         "validation of hits.", "appraisal", ("omics_convention",), domains=BIOMED + ("omics_high_throughput",),
         study_designs=("omics", "omics_high_throughput")),
    # ---------------------------------------------------------------- computer science & machine learning
    Rule("CS-leakage", "Check for leakage between training and test data (duplicates, temporal or similarity "
         "leakage, preprocessing on the full dataset).", "appraisal", ("ml_leakage", "reforms"),
         "Leakage taxonomy (8 types); REFORMS: data preprocessing and evaluation", CS),
    Rule("CS-reproducibility", "Note whether code, data, model weights and computing details are available.",
         "appraisal", ("reforms",), "REFORMS: computational reproducibility", CS),
    Rule("CS-benchmarks", "Compare methods only on shared datasets, splits and metrics; qualify state-of-the-art "
         "claims accordingly.", "synthesis", ("reforms",), "REFORMS: metrics and uncertainty quantification", CS),
    Rule("CS-artifacts", "Model cards and repository claims are community artefacts, not peer-reviewed evidence.",
         "reporting", ("tool_integrity",), domains=CS),
    Rule("CS-slr", "Software-engineering systematic reviews: document the protocol, search, study selection and "
         "quality assessment.", "reporting", ("kitchenham",), "Kitchenham & Charters: planning, conducting, "
         "reporting", CS, review_types=("systematic_prisma",)),
    # ---------------------------------------------------------------- chemistry
    Rule("CH-yield", "Distinguish isolated yields from crude or conversion yields.", "synthesis", ("orglett_data",),
         domains=("chemistry",)),
    # Rule text stays free of decimals and percentages: it is quoted in narratives, where the ungrounded-number
    # gate would (rightly) reject any figure not found in the retrieved studies. Thresholds go in the locator.
    Rule("CH-purity", "Note whether compound purity is documented for compounds used in bioassays.", "appraisal",
         ("joc_purity",), "JOC author guidelines: >=95% purity for compounds in physical measurements or bioassays",
         domains=("chemistry",)),
    Rule("CH-computation", "For computed results, report the level of theory and basis set; separate calculated "
         "from measured values.", "appraisal", ("iupac_qm",), domains=("chemistry",)),
    Rule("CH-conditions", "Keep units and reaction conditions (solvent, catalyst, temperature) with every result.",
         "synthesis", ("chem_convention",), domains=("chemistry",)),
    # ---------------------------------------------------------------- physics
    Rule("PH-uncertainty", "Report measured values with their stated uncertainty and coverage; do not compare values "
         "without their uncertainties.", "synthesis", ("gum",), "GUM: standard and expanded uncertainty",
         ("physics", "engineering_materials")),
    Rule("PH-components", "Check that all significant uncertainty components are evaluated and combined, not only "
         "the statistical scatter.", "appraisal", ("gum",), "GUM: combined standard uncertainty",
         ("physics", "engineering_materials")),
    Rule("PH-approximation", "Distinguish exact results from perturbative or numerical approximations and state the "
         "parameter regime and assumptions.", "synthesis", ("physics_convention",), domains=("physics",)),
    # ---------------------------------------------------------------- mathematics
    Rule("MA-status", "Distinguish proved theorems from conditional results and conjectures.", "synthesis",
         ("math_convention",), domains=("mathematics",)),
    Rule("MA-assumptions", "State the hypotheses each result depends on.", "synthesis", ("math_convention",),
         domains=("mathematics",)),
    Rule("MA-verification", "Note whether a proof is peer reviewed, preprint-only, or formally verified (e.g. Lean, "
         "Coq).", "appraisal", ("math_convention",), domains=("mathematics",)),
    Rule("MA-no-sample", "Sample sizes do not apply to proofs; never invent them.", "appraisal",
         ("tool_integrity",), domains=("mathematics",), enforcement="sample_policy_gate"),
    # ---------------------------------------------------------------- other domains
    Rule("EN-repeatability", "Check repeatability: number of specimens or runs and the test standard used.",
         "appraisal", ("engineering_convention",), domains=("engineering_materials",)),
    Rule("EN-conditions", "Compare performance only under stated, standardised test conditions.", "synthesis",
         ("engineering_convention",), domains=("engineering_materials",)),
    Rule("SS-identification", "Distinguish correlational designs from causal identification strategies (RCT, IV, "
         "DiD, RDD).", "appraisal", ("social_convention",), domains=("social_sciences",)),
    Rule("SS-context", "Do not generalise effects across institutional or demographic contexts without evidence.",
         "synthesis", ("social_convention",), domains=("social_sciences",)),
    Rule("HU-provenance", "Assess source provenance and corpus representativeness.", "appraisal",
         ("humanities_convention",), domains=("humanities_interdisciplinary",)),
    Rule("HU-interpretation", "Preserve competing scholarly interpretations and their historical context.",
         "synthesis", ("humanities_convention",), domains=("humanities_interdisciplinary",)),
    # ---------------------------------------------------------------- study-method profiles (any domain)
    # Selected by the method profiles the planner detects (services/domain_profiles.METHODOLOGICAL_PROFILES).
    Rule("M-observational-a", "Observational studies: assess confounding control and how participants were "
         "selected.", "appraisal", ("robins_e",), "ROBINS-E: confounding; selection of participants",
         study_designs=("observational",)),
    Rule("M-observational-s", "Describe observational findings as associations ('is associated with'), not effects.",
         "synthesis", ("grade",), "GRADE: observational evidence starts at low certainty",
         study_designs=("observational",)),
    Rule("M-experimental-a", "Experiments: check positive and negative controls, replicates and assay sensitivity.",
         "appraisal", ("experimental_convention",), study_designs=("experimental",)),
    Rule("M-experimental-s", "Scope experimental conclusions to the model system that was tested.", "synthesis",
         ("grade",), "GRADE: indirectness", study_designs=("experimental",)),
    Rule("M-intervention-a", "Intervention trials: check randomisation, blinding, statistical power, primary "
         "endpoints and adverse events.", "appraisal", ("rob2",), "RoB 2: five bias domains",
         study_designs=("clinical_intervention",)),
    Rule("M-intervention-s", "Distinguish surrogate markers from patient-relevant outcomes.", "synthesis",
         ("grade",), "GRADE: indirectness of outcomes", study_designs=("clinical_intervention",)),
    Rule("M-predictive-a", "Prediction models: check external validation, calibration, discrimination and leakage "
         "between development and test data.", "appraisal", ("probast_ai", "ml_leakage"),
         "PROBAST+AI: analysis domain", study_designs=("predictive_modeling",)),
    Rule("M-predictive-s", "Distinguish internal validation performance from external or prospective validation.",
         "synthesis", ("probast_ai",), "PROBAST+AI: model development vs evaluation",
         study_designs=("predictive_modeling",)),
    Rule("M-benchmark-a", "Benchmarks: check how data were split, fairness of baselines and completeness of "
         "ablations.", "appraisal", ("reforms", "ml_leakage"), "REFORMS: data and evaluation",
         study_designs=("ml_benchmarking",)),
    Rule("M-benchmark-s", "Qualify state-of-the-art claims by the split method and out-of-distribution robustness.",
         "synthesis", ("reforms",), "REFORMS: generalisability", study_designs=("ml_benchmarking",)),
    Rule("M-omics-s", "Distinguish primary screening hits from independently confirmed results.", "synthesis",
         ("omics_convention",), study_designs=("omics_high_throughput",)),
    Rule("M-theory-a", "Theoretical work: check assumptions, regularity conditions, asymptotic regime and proof "
         "completeness.", "appraisal", ("math_convention",), study_designs=("theoretical_mathematical",)),
    Rule("M-theory-s", "State the assumptions each theoretical result depends on; empirical sample sizes do not "
         "apply.", "synthesis", ("math_convention",), study_designs=("theoretical_mathematical",),
         enforcement="sample_policy_gate"),
    Rule("M-simulation-a", "Simulations: check numerical resolution, convergence and the accuracy of the model or "
         "force field used.", "appraisal", ("simulation_convention",), study_designs=("simulation_based",)),
    Rule("M-simulation-s", "Distinguish simulated quantities from experimentally measured values.", "synthesis",
         ("simulation_convention",), study_designs=("simulation_based",)),
    Rule("M-qualitative-a", "Qualitative studies: check sampling, data collection, analysis approach and researcher "
         "reflexivity.", "appraisal", ("coreq", "srqr"),
         "COREQ: research team and reflexivity, study design, analysis and findings", study_designs=("qualitative",)),
    Rule("M-qualitative-s", "Present qualitative findings as interpretations grounded in the analysed data.",
         "synthesis", ("srqr",), study_designs=("qualitative",)),
    Rule("M-review-a", "Reviews: check the protocol, search comprehensiveness, study selection and risk-of-bias "
         "assessment.", "appraisal", ("amstar2",), "AMSTAR 2: critical domains",
         study_designs=("literature_conceptual_review",)),
    Rule("M-review-s", "Do not count overlapping reviews, or a review and its own included studies, as independent "
         "evidence.", "synthesis", ("tool_integrity",), study_designs=("literature_conceptual_review",)),
    # ---------------------------------------------------------------- review frameworks
    Rule("RV-prisma-flow", "Report the flow of records: identified per source, duplicates removed, screened, "
         "included.", "reporting", ("prisma2020",), "PRISMA 2020 flow diagram", review_types=("systematic_prisma",)),
    Rule("RV-prisma-search", "Report the full search strategy used for every source.", "reporting", ("prisma2020",),
         "PRISMA 2020: search strategy", review_types=("systematic_prisma", "scoping")),
    Rule("RV-prisma-certainty", "Assess risk of bias and certainty of evidence for the main findings.", "reporting",
         ("prisma2020", "grade"), "PRISMA 2020: risk of bias and certainty assessment",
         review_types=("systematic_prisma",)),
    Rule("RV-scoping", "Chart and map the evidence (what exists, where, how studied); risk-of-bias appraisal is "
         "optional.", "reporting", ("prisma_scr",), "PRISMA-ScR: charting the data; critical appraisal (optional)",
         review_types=("scoping",)),
    Rule("RV-narrative", "State the review's aim, how the literature was found, reference key statements, and "
         "present key data with its level of evidence.", "reporting", ("sanra",), "SANRA: six items",
         review_types=("narrative",)),
]

RULES_BY_ID: Dict[str, Rule] = {r.rule_id: r for r in RULES}


def select_rules(
    domain: Optional[str],
    designs: Sequence[str] = (),
    review_type: Optional[str] = None,
    stages: Sequence[Stage] = ("appraisal", "synthesis", "reporting", "gap"),
    include_universal: bool = True,
) -> List[Rule]:
    """Rules that apply to this search, domain-specific ones first."""
    picked = [
        r for r in RULES
        if r.stage in stages
        and r.applies(domain, designs, review_type)
        and (include_universal or "*" not in r.domains)
    ]
    return sorted(picked, key=lambda r: ("*" in r.domains, RULES.index(r)))


def rule_texts(domain: str, stage: Stage) -> List[str]:
    """Domain-specific rule wording for one stage (all study designs); used to populate the legacy profile lists."""
    return [
        r.text for r in RULES
        if r.stage == stage and "*" not in r.domains and domain in r.domains and "*" in r.review_types
    ]


def method_rule_texts(method_id: str, stage: Stage) -> List[str]:
    """Rule wording for one study-method profile; used to populate METHODOLOGICAL_PROFILES."""
    return [r.text for r in RULES if r.stage == stage and method_id in r.study_designs]


def render_rules_for_prompt(rules: Sequence[Rule], max_tokens: int = 250, chars_per_token: float = 3.5) -> str:
    """Compact numbered rule block for prompts, truncated to a token budget (domain-specific rules come first)."""
    lines: List[str] = []
    used = 0
    for i, r in enumerate(rules, 1):
        line = f"{i}. {r.text}"
        cost = int(len(line) / chars_per_token) + 1
        if used + cost > max_tokens:
            break
        lines.append(line)
        used += cost
    return "\n".join(lines)
