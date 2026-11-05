"""
Routes each study to the appraisal framework that matches its design and lists the reporting gaps visible in its
abstract.

This is screening, not appraisal: tools such as RoB 2 or QUADAS-3 require full-text signalling questions, so this
module never outputs a tool rating ("low", "some concerns", "high"). It only names the matching framework and the
items an abstract does not report, so readers know what to check in the full text.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from frameworks.sources import SOURCES

SCREENING_SUFFIX = "framework-informed screening from abstract, not a formal rating"

_RCT = re.compile(r"\brandomi[sz]ed\b.{0,60}\b(trial|controlled|assigned|allocation)\b|\brct\b", re.I)
_DIAGNOSTIC = re.compile(r"\b(diagnostic accuracy|sensitivity and specificity|specificity of|reference standard|"
                         r"gold standard|index test)\b", re.I)
_PROGNOSTIC_FACTOR = re.compile(r"\b(prognostic factor|independent(ly)? (predictor|associated)|"
                                r"associated with (overall |progression-free )?survival)\b", re.I)
_PREDICTION = re.compile(r"\b(prediction model|predictive model|risk score|nomogram|risk prediction|"
                         r"prognostic model|classifier|machine learning model)\b", re.I)
_ANIMAL = re.compile(r"\b(mice|murine|mouse model|rats?|xenograft|in vivo|zebrafish|rabbits?|primates?)\b", re.I)
_EXPOSURE = re.compile(r"\b(cohort|case-control|cross-sectional|exposure|observational|registry|population-based)\b",
                       re.I)
_NRSI = re.compile(r"\b(non-?randomi[sz]ed|single-arm|retrospective(ly)? (treated|analysis of patients)|"
                   r"real-world|propensity)\b", re.I)
_OMICS = re.compile(r"\b(rna-seq|single-cell|proteomic|transcriptom|genome-wide|gwas|high-throughput|screen)\b",
                    re.I)
_CONFOUNDING = re.compile(r"\b(adjust|confound|propensity|matched|multivariable|multivariate)\w*", re.I)


@dataclass
class AppraisalRoute:
    design_id: str
    framework_label: str
    source_ids: Tuple[str, ...]
    concerns: List[str] = field(default_factory=list)


def _label(source_ids: Tuple[str, ...], design_name: str) -> str:
    names = " / ".join(SOURCES[s].short_name for s in source_ids)
    return f"{names} ({design_name}) - {SCREENING_SUFFIX}"


def detect_biomedical_design(text: str) -> Optional[str]:
    """Most specific biomedical design signalled by the abstract, or None."""
    if _DIAGNOSTIC.search(text):
        return "diagnostic_accuracy"
    if _RCT.search(text):
        return "rct"
    if _PREDICTION.search(text):
        return "prediction_model"
    if _PROGNOSTIC_FACTOR.search(text):
        return "prognostic_factor"
    if _ANIMAL.search(text):
        return "animal"
    if _NRSI.search(text):
        return "nrsi_intervention"
    if _EXPOSURE.search(text):
        return "observational_exposure"
    if _OMICS.search(text):
        return "omics"
    return None


def route_biomedical(abstract: str) -> Optional[AppraisalRoute]:
    t = abstract or ""
    design = detect_biomedical_design(t)
    if design == "rct":
        route = AppraisalRoute(design, _label(("rob2",), "randomised trial"), ("rob2",))
        if not re.search(r"\bconceal", t, re.I):
            route.concerns.append("Allocation concealment is not described in the abstract (RoB 2 domain 1: "
                                  "randomisation process) - not assessable from abstract.")
        if not re.search(r"\b(blind|mask)", t, re.I):
            route.concerns.append("Blinding is not described in the abstract (RoB 2 domains 2 and 4: deviations "
                                  "from intended interventions; measurement of the outcome) - not assessable from "
                                  "abstract.")
        return route
    if design == "diagnostic_accuracy":
        route = AppraisalRoute(design, _label(("quadas3",), "diagnostic accuracy study"), ("quadas3",))
        if not re.search(r"\b(reference standard|gold standard)\b", t, re.I):
            route.concerns.append("The reference standard defining the target condition is not described in the "
                                  "abstract (QUADAS-3: target condition domain).")
        return route
    if design == "prediction_model":
        route = AppraisalRoute(design, _label(("probast_ai",), "prediction model study"), ("probast_ai",))
        if not re.search(r"\b(external(ly)? validat|independent (cohort|dataset|centre|center)|temporal validation)",
                         t, re.I):
            route.concerns.append("External validation is not reported in the abstract; internal performance may "
                                  "overstate accuracy (PROBAST+AI: analysis domain and applicability).")
        return route
    if design == "prognostic_factor":
        route = AppraisalRoute(design, _label(("quips",), "prognostic factor study"), ("quips",))
        if not _CONFOUNDING.search(t):
            route.concerns.append("Adjustment for other prognostic factors is not described in the abstract "
                                  "(QUIPS: study confounding).")
        return route
    if design == "animal":
        route = AppraisalRoute(design, _label(("syrcle", "arrive2"), "animal study"), ("syrcle", "arrive2"))
        if not (re.search(r"\brandomi[sz]", t, re.I) and re.search(r"\b(blind|mask)", t, re.I)):
            route.concerns.append("Randomisation and blinded outcome assessment are not described in the abstract "
                                  "(SYRCLE: selection and detection bias; ARRIVE v2 Essential 10).")
        return route
    if design in ("nrsi_intervention", "observational_exposure"):
        sid = "robins_i_v2" if design == "nrsi_intervention" else "robins_e"
        name = "non-randomised intervention study" if design == "nrsi_intervention" else "observational exposure study"
        route = AppraisalRoute(design, _label((sid,), name), (sid,))
        if not _CONFOUNDING.search(t):
            route.concerns.append(f"Confounding control is not described in the abstract ({SOURCES[sid].short_name}: "
                                  "bias due to confounding).")
        return route
    if design == "omics":
        route = AppraisalRoute(design, _label(("omics_convention",), "high-throughput study"), ("omics_convention",))
        if not re.search(r"\b(fdr|false discovery|bonferroni|multiple[- ]testing|adjusted p)", t, re.I):
            route.concerns.append("Multiple-testing correction is not described in the abstract.")
        return route
    return None


def route_computational(abstract: str) -> AppraisalRoute:
    t = abstract or ""
    route = AppraisalRoute("ml_study", _label(("reforms", "ml_leakage"), "machine-learning study"),
                           ("reforms", "ml_leakage"))
    if not re.search(r"\b(leakage|scaffold split|sequence identity|homology|cluster split|temporal split|"
                     r"out-of-distribution|held-out|external test)", t, re.I):
        route.concerns.append("How train and test data were separated (leakage control) is not described in the "
                              "abstract (leakage taxonomy; REFORMS).")
    if not re.search(r"\b(code|github|open[- ]source|publicly available|zenodo|huggingface)\b", t, re.I):
        route.concerns.append("Code or data availability is not stated in the abstract (REFORMS: computational "
                              "reproducibility).")
    return route


def route_physics(abstract: str) -> AppraisalRoute:
    t = abstract or ""
    route = AppraisalRoute("physics_study", _label(("gum", "physics_convention"), "physical sciences study"),
                           ("gum", "physics_convention"))
    if re.search(r"\d", t) and not re.search(r"(±|\+/-|uncertaint|error bar|confidence|standard deviation|sigma)",
                                             t, re.I):
        route.concerns.append("Reported values are given without a stated uncertainty in the abstract (GUM).")
    return route


def route_chemistry(abstract: str) -> AppraisalRoute:
    t = abstract or ""
    route = AppraisalRoute("chemistry_study", _label(("iupac_qm", "orglett_data", "chem_convention"),
                                                     "chemistry study"),
                           ("iupac_qm", "orglett_data", "chem_convention"))
    if re.search(r"\byield", t, re.I) and not re.search(r"\b(isolated|crude|conversion)\b", t, re.I):
        route.concerns.append("The yield type (isolated vs crude or conversion) is not specified in the abstract "
                              "(Org. Lett. data-reporting standards).")
    if re.search(r"\b(dft|density functional|ab initio|hartree|coupled[- ]cluster|quantum chemical)", t, re.I) and \
            not re.search(r"\b(basis set|b3lyp|pbe|6-31|def2|cc-pv|functional)\b", t, re.I):
        route.concerns.append("The level of theory or basis set is not stated in the abstract (IUPAC computational "
                              "reporting).")
    return route


def route_mathematics(abstract: str) -> AppraisalRoute:
    route = AppraisalRoute("mathematical_result", _label(("math_convention",), "mathematical result"),
                           ("math_convention",))
    route.concerns.append("Proof steps and lemma dependencies need full-text inspection; the abstract states the "
                          "result and its assumptions only.")
    if not re.search(r"\b(lean|coq|isabelle|formali[sz]ed|formally verified)\b", abstract or "", re.I):
        route.concerns.append("No formal verification (e.g. Lean, Coq) is mentioned in the abstract.")
    return route


def route_for_module(module_key: str, abstract: str) -> Optional[AppraisalRoute]:
    """Framework route for an evidence-table domain module (keys from DOMAIN_MODULE_SPECIFICATIONS)."""
    if module_key in ("biomedical_clinical", "molecular_biology_omics"):
        return route_biomedical(abstract)
    if module_key == "ai_computer_science":
        return route_computational(abstract)
    if module_key == "physics":
        return route_physics(abstract)
    if module_key == "chemistry_materials":
        return route_chemistry(abstract)
    if module_key == "mathematics":
        return route_mathematics(abstract)
    return None
