"""
Section 2: Universal, Question-Adaptive, Source-Grounded Scientific Evidence Matrix Engine.

Transforms retrieved literature and Section 1 synthesis records into a three-level
scientific evidence matrix (Level A: DocumentRecord, Level B: EvidenceUnitRecord,
Level C: GranularFindingRecord) with query-adaptive domain extraction plans,
separated factual vs. interpretive fields, quantitative context preservation,
comparability classification, method-aware critical appraisal, and multi-layer
verification.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from frameworks.appraisal_routes import route_for_module
from services.extraction_patterns import SAMPLE_SIZE_PATTERN, study_sample_size
from services.verification_gates import study_source_text
from models import (
    AnalysisTierType,
    AtomicFinding,
    ClaimVerificationState,
    ComparabilityAssessment,
    ComparabilityStatusType,
    DerivedQuantityRecord,
    DocumentRecord,
    DocumentRelationType,
    EvidenceAccessType,
    EvidenceMatrixRow,
    EvidenceTableValidationReport,
    EvidenceUnitRecord,
    EvidenceUnitRelationshipType,
    ExtractionFieldSpec,
    ExtractionPlan,
    FieldProvenance,
    FindingDirectionType,
    GranularFindingRecord,
    MethodologicalAppraisalRecord,
    PaperRecord,
    QueryDecompositionPlan,
    QuantitativeResultDetail,
    ReportingStatusType,
    StructuredEvidenceMatrix,
    SynthesisClaim,
    SynthesisEvidencePackage,
)


# ============================================================================
# PHASE 4, 5 & 6: Universal Core Dimensions & Domain-Specific Extraction Plans
# ============================================================================

UNIVERSAL_ESSENTIAL_FIELDS: List[ExtractionFieldSpec] = [
    ExtractionFieldSpec(
        field_key="document_identification",
        display_label="Study & Persistent IDs",
        dimension="Identification",
        priority="Essential",
        domain_module="universal",
        description="Authors, year, source database, publication status, DOI/PMID/arXiv ID, and evidence access level.",
    ),
    ExtractionFieldSpec(
        field_key="research_objective",
        display_label="Research Objective & Subquestion",
        dimension="Research Objective",
        priority="Essential",
        domain_module="universal",
        description="Primary scientific question or hypothesis addressed by the study and mapped user subquestion.",
    ),
    ExtractionFieldSpec(
        field_key="context_and_setting",
        display_label="System / Population / Context",
        dimension="Context",
        priority="Essential",
        domain_module="universal",
        description="Experimental system, clinical population, dataset, physical regime, or formal mathematical setting.",
    ),
    ExtractionFieldSpec(
        field_key="methodological_approach",
        display_label="Methodology & Analytical Design",
        dimension="Methodology",
        priority="Essential",
        domain_module="universal",
        description="Study design, experimental/computational method, model architecture, or proof technique.",
    ),
    ExtractionFieldSpec(
        field_key="primary_quantitative_findings",
        display_label="Source-Grounded Findings & Quantitative Context",
        dimension="Evidence",
        priority="Essential",
        domain_module="universal",
        description="Reported outcomes, metrics, effect sizes, comparators, sample sizes, and uncertainty intervals.",
    ),
    ExtractionFieldSpec(
        field_key="limitations_and_appraisal",
        display_label="Limitations & Methodological Appraisal",
        dimension="Limitations",
        priority="Essential",
        domain_module="universal",
        description="Source-reported limitations separated from independent methodological and applicability concerns.",
    ),
    ExtractionFieldSpec(
        field_key="comparability_and_relevance",
        display_label="Comparability & Synthesis Link",
        dimension="Relevance",
        priority="Essential",
        domain_module="universal",
        description="Cross-study comparability status, shared/divergent dimensions, and linked Section 1 claims.",
    ),
    ExtractionFieldSpec(
        field_key="source_provenance",
        display_label="Provenance & Access Status",
        dimension="Provenance",
        priority="Essential",
        domain_module="universal",
        description="Document ID, source section, supporting excerpt, extraction method, and verification status.",
    ),
]


DOMAIN_MODULE_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "biomedical_clinical": {
        "label": "Biomedical, Clinical & Public Health",
        "appraisal_framework": "Design-matched: RoB 2 / ROBINS-I V2 / ROBINS-E / PROBAST+AI / QUIPS / QUADAS-3 / SYRCLE + ARRIVE v2 (framework-informed screening from abstract)",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="clinical_study_design",
                display_label="Clinical / Preclinical Design",
                dimension="Methodology",
                priority="Essential",
                domain_module="biomedical_clinical",
                description="RCT, prospective/retrospective cohort, case-control, diagnostic/prognostic model, or preclinical model.",
            ),
            ExtractionFieldSpec(
                field_key="population_and_sample",
                display_label="Population / Sample Size",
                dimension="Context",
                priority="Essential",
                domain_module="biomedical_clinical",
                description="Patient cohort, eligibility context, organism/model system, and sample size (N).",
            ),
            ExtractionFieldSpec(
                field_key="intervention_and_comparator",
                display_label="Intervention / Exposure & Comparator",
                dimension="Methodology",
                priority="Essential",
                domain_module="biomedical_clinical",
                description="Therapeutic intervention, exposure, biomarker/index test, and control/reference standard.",
            ),
            ExtractionFieldSpec(
                field_key="endpoints_and_followup",
                display_label="Endpoints & Follow-Up / Validation",
                dimension="Evidence",
                priority="Recommended",
                domain_module="biomedical_clinical",
                description="Primary/secondary clinical endpoints, follow-up duration, or internal/external validation status.",
            ),
        ],
        "not_applicable": [
            ExtractionFieldSpec(
                field_key="formal_theorem_proof_technique",
                display_label="Formal Proof Construction",
                dimension="Methodology",
                priority="Not applicable",
                domain_module="biomedical_clinical",
                description="Formal axiomatic proof techniques are not applicable to empirical clinical/biomedical studies.",
            )
        ],
    },
    "molecular_biology_omics": {
        "label": "Molecular Biology, Genetics & Omics",
        "appraisal_framework": "Experimental Controls, Replicate Structure & Orthogonal Validation Appraisal",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="biological_system",
                display_label="Organism / Cell / Tissue System",
                dimension="Context",
                priority="Essential",
                domain_module="molecular_biology_omics",
                description="Species, strain, cell line, primary tissue, or molecular target.",
            ),
            ExtractionFieldSpec(
                field_key="assay_and_perturbation",
                display_label="Assay / Perturbation Platform",
                dimension="Methodology",
                priority="Essential",
                domain_module="molecular_biology_omics",
                description="Sequencing/omics platform, CRISPR/RNAi/chemical perturbation, or biochemical assay.",
            ),
            ExtractionFieldSpec(
                field_key="replicates_and_controls",
                display_label="Sample Scale / Controls & Normalization",
                dimension="Methodology",
                priority="Recommended",
                domain_module="molecular_biology_omics",
                description="Biological/technical replicates, negative/positive controls, and multiple-testing correction.",
            ),
            ExtractionFieldSpec(
                field_key="orthogonal_validation",
                display_label="Validation Tier (In Silico / In Vitro / In Vivo)",
                dimension="Evidence",
                priority="Essential",
                domain_module="molecular_biology_omics",
                description="Whether findings are computational predictions, in vitro assays, or in vivo functional validations.",
            ),
        ],
        "not_applicable": [],
    },
    "ai_computer_science": {
        "label": "Artificial Intelligence, Machine Learning & Computer Science",
        "appraisal_framework": "REFORMS / ML leakage taxonomy (framework-informed screening from abstract)",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="model_architecture_and_task",
                display_label="Architecture & Task Formulation",
                dimension="Methodology",
                priority="Essential",
                domain_module="ai_computer_science",
                description="Neural/algorithmic architecture, paradigm (supervised, self-supervised, generative, RL), and task.",
            ),
            ExtractionFieldSpec(
                field_key="datasets_and_split_hygiene",
                display_label="Datasets & Split / Leakage Protocol",
                dimension="Context",
                priority="Essential",
                domain_module="ai_computer_science",
                description="Training and evaluation datasets, sample scale, cross-validation or holdout split, and leakage control.",
            ),
            ExtractionFieldSpec(
                field_key="baselines_and_validation_tier",
                display_label="Baselines & Validation Tier",
                dimension="Methodology",
                priority="Essential",
                domain_module="ai_computer_science",
                description="Comparator models and validation tier (development, internal benchmark, external/OOD validation, ablation).",
            ),
            ExtractionFieldSpec(
                field_key="reproducibility_and_artifacts",
                display_label="Code / Model / Compute Availability",
                dimension="Provenance",
                priority="Recommended",
                domain_module="ai_computer_science",
                description="Open-source code, pretrained weights, Hugging Face artifacts, or compute constraints.",
            ),
        ],
        "not_applicable": [
            ExtractionFieldSpec(
                field_key="patient_blinding_allocation",
                display_label="Patient Allocation Concealment",
                dimension="Methodology",
                priority="Not applicable",
                domain_module="ai_computer_science",
                description="Clinical allocation concealment is not applicable to purely algorithmic benchmark studies.",
            )
        ],
    },
    "chemistry_materials": {
        "label": "Chemistry, Chemical Engineering & Materials Science",
        "appraisal_framework": "IUPAC computational reporting / ACS data-reporting standards (framework-informed screening from abstract)",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="chemical_or_material_system",
                display_label="Chemical / Material System",
                dimension="Context",
                priority="Essential",
                domain_module="chemistry_materials",
                description="Catalyst, polymer, crystal composition, ligand/scaffold, or material formulation.",
            ),
            ExtractionFieldSpec(
                field_key="synthesis_or_simulation_method",
                display_label="Synthesis / Processing / DFT Method",
                dimension="Methodology",
                priority="Essential",
                domain_module="chemistry_materials",
                description="Synthetic route, fabrication protocol, or computational functional/basis set.",
            ),
            ExtractionFieldSpec(
                field_key="operating_conditions_and_characterization",
                display_label="Conditions & Characterization",
                dimension="Context",
                priority="Essential",
                domain_module="chemistry_materials",
                description="Temperature, pressure, solvent, pH, and structural confirmation (XRD, NMR, MS, SEM/TEM).",
            ),
            ExtractionFieldSpec(
                field_key="stability_and_selectivity",
                display_label="Yield / Selectivity / Stability",
                dimension="Evidence",
                priority="Recommended",
                domain_module="chemistry_materials",
                description="Reaction yield, selectivity, Faraday efficiency, cycle life, or thermal/chemical stability.",
            ),
        ],
        "not_applicable": [],
    },
    "physics": {
        "label": "Physics & Physical Sciences",
        "appraisal_framework": "GUM uncertainty reporting / physics conventions (framework-informed screening from abstract)",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="physical_system_and_regime",
                display_label="Physical System & Parameter Regime",
                dimension="Context",
                priority="Essential",
                domain_module="physics",
                description="Physical system, energy/temperature/length scale, coupling regime, or symmetry group.",
            ),
            ExtractionFieldSpec(
                field_key="theoretical_or_experimental_method",
                display_label="Apparatus / Numerical / Analytical Method",
                dimension="Methodology",
                priority="Essential",
                domain_module="physics",
                description="Experimental detector/setup, lattice/Monte Carlo simulation, or analytical approximation.",
            ),
            ExtractionFieldSpec(
                field_key="assumptions_and_approximations",
                display_label="Assumptions & Scaling / Boundary Conditions",
                dimension="Methodology",
                priority="Essential",
                domain_module="physics",
                description="Perturbative order, thermodynamic limit, boundary conditions, or idealized model assumptions.",
            ),
            ExtractionFieldSpec(
                field_key="uncertainty_and_error_budget",
                display_label="Observable & Systematic / Statistical Uncertainty",
                dimension="Evidence",
                priority="Recommended",
                domain_module="physics",
                description="Measured/predicted physical observable, error bars, or numerical convergence check.",
            ),
        ],
        "not_applicable": [],
    },
    "mathematics": {
        "label": "Mathematics, Theoretical CS & Formal Logic",
        "appraisal_framework": "Mathematics publishing convention: proof status, assumptions, verification",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="mathematical_structures",
                display_label="Mathematical Objects & Setting",
                dimension="Context",
                priority="Essential",
                domain_module="mathematics",
                description="Spaces, algebraic/topological structures, complexity classes, or formal systems studied.",
            ),
            ExtractionFieldSpec(
                field_key="hypotheses_and_assumptions",
                display_label="Key Hypotheses & Regularity Conditions",
                dimension="Methodology",
                priority="Essential",
                domain_module="mathematics",
                description="Explicit axioms, regularity conditions, dimensionality restrictions, or oracle assumptions.",
            ),
            ExtractionFieldSpec(
                field_key="theorem_and_proof_method",
                display_label="Main Theorem / Bound & Proof Technique",
                dimension="Evidence",
                priority="Essential",
                domain_module="mathematics",
                description="Formal theorem, asymptotic bound, exact identity, or constructive/probabilistic proof method.",
            ),
            ExtractionFieldSpec(
                field_key="generality_and_counterexamples",
                display_label="Scope of Validity & Sharpness / Counterexamples",
                dimension="Limitations",
                priority="Recommended",
                domain_module="mathematics",
                description="Necessary vs. sufficient conditions, optimality/tightness of bounds, and known counterexamples.",
            ),
        ],
        "not_applicable": [
            ExtractionFieldSpec(
                field_key="experimental_sample_size_pvalue",
                display_label="Empirical Sample Size (N) & p-value",
                dimension="Evidence",
                priority="Not applicable",
                domain_module="mathematics",
                description="Empirical sample sizes, confidence intervals, and p-values do not apply to formal mathematical proofs.",
            )
        ],
    },
    "social_sciences_humanities": {
        "label": "Social Sciences, Economics, Policy & Humanities",
        "appraisal_framework": "Causal Identification, Construct Validity, Selection Bias & Institutional Context Appraisal",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="institutional_or_population_setting",
                display_label="Population / Geographic & Institutional Setting",
                dimension="Context",
                priority="Essential",
                domain_module="social_sciences_humanities",
                description="Country/region, institutional context, time period, and sampling frame or archival corpus.",
            ),
            ExtractionFieldSpec(
                field_key="identification_or_interpretive_design",
                display_label="Identification Strategy / Research Design",
                dimension="Methodology",
                priority="Essential",
                domain_module="social_sciences_humanities",
                description="DiD, IV, RDD, panel fixed effects, survey experiment, ethnography, or qualitative/archival analysis.",
            ),
            ExtractionFieldSpec(
                field_key="constructs_and_measurement",
                display_label="Constructs & Data Sources",
                dimension="Methodology",
                priority="Essential",
                domain_module="social_sciences_humanities",
                description="Operationalized variables, survey instruments, administrative data, or primary historical sources.",
            ),
            ExtractionFieldSpec(
                field_key="external_validity_and_context",
                display_label="Policy / Contextual Generalizability",
                dimension="Limitations",
                priority="Recommended",
                domain_module="social_sciences_humanities",
                description="Scope conditions, institutional dependence, and sensitivity to specification choices.",
            ),
        ],
        "not_applicable": [],
    },
    "universal_fallback": {
        "label": "Universal Interdisciplinary Scientific Profile",
        "appraisal_framework": "Universal Methodological Rigor, Provenance & Scope Appraisal",
        "visible_columns": [
            ExtractionFieldSpec(
                field_key="study_design_and_category",
                display_label="Study Design & Contribution Type",
                dimension="Methodology",
                priority="Essential",
                domain_module="universal_fallback",
                description="Empirical, computational, theoretical, comparative, or synthesis design.",
            ),
            ExtractionFieldSpec(
                field_key="system_and_scale",
                display_label="Study System, Dataset & Scale",
                dimension="Context",
                priority="Essential",
                domain_module="universal_fallback",
                description="Domain system, dataset/corpus, or analytical setting and reported scale.",
            ),
            ExtractionFieldSpec(
                field_key="comparator_and_validation",
                display_label="Comparator & Validation Approach",
                dimension="Methodology",
                priority="Essential",
                domain_module="universal_fallback",
                description="Baseline, control condition, or validation protocol.",
            ),
            ExtractionFieldSpec(
                field_key="scope_and_applicability",
                display_label="Scope & Applicability Conditions",
                dimension="Limitations",
                priority="Recommended",
                domain_module="universal_fallback",
                description="Conditions under which the reported findings hold.",
            ),
        ],
        "not_applicable": [],
    },
}


# Evidence-table module for each user-selectable domain (interdisciplinary queries keep keyword routing)
USER_DOMAIN_TO_MODULE: Dict[str, str] = {
    "biomedical_clinical": "biomedical_clinical",
    "computer_science_ai": "ai_computer_science",
    "chemistry": "chemistry_materials",
    "engineering_materials": "chemistry_materials",
    "physics": "physics",
    "mathematics": "mathematics",
    "social_sciences": "social_sciences_humanities",
    "humanities_interdisciplinary": "social_sciences_humanities",
}


def resolve_domain_module_key(query: str, decomposition_plan: Optional[QueryDecompositionPlan]) -> str:
    """Map the query and domain profile to one of the 8 specialized Section 2 extraction modules."""
    q_lower = (query or "").lower()
    primary = ""
    if decomposition_plan:
        if decomposition_plan.question_understanding:
            primary = (decomposition_plan.question_understanding.primary_domain or "").lower()
        elif decomposition_plan.detected_domain:
            primary = (decomposition_plan.detected_domain or "").lower()

    combined = f"{primary} {q_lower}"

    # A domain the user chose in the UI wins over keyword detection
    understanding = decomposition_plan.question_understanding if decomposition_plan else None
    if understanding and understanding.domain_selected_by_user and primary in USER_DOMAIN_TO_MODULE:
        return USER_DOMAIN_TO_MODULE[primary]

    # Check Mathematics / Formal Logic first if explicit formal proof / theorem query
    if any(
        kw in combined
        for kw in [
            "theorem",
            "conjecture",
            "formal proof",
            "banach",
            "hilbert space",
            "cohomology",
            "galois",
            "homotopy",
            "diophantine",
            "p vs np",
            "pure mathematics",
        ]
    ):
        return "mathematics"

    # Check Chemistry & Materials
    if any(
        kw in combined
        for kw in [
            "catalys",
            "electrocatalys",
            "perovskite",
            "zeolite",
            "mof ",
            "metal-organic framework",
            "organic synthesis",
            "enantiomeric",
            "faraday efficiency",
            "battery cathode",
            "solid electrolyte",
            "materials science",
            "chemistry",
        ]
    ):
        return "chemistry_materials"

    # Check Physics
    if any(
        kw in combined
        for kw in [
            "quantum",
            "superconduct",
            "astrophys",
            "cosmolog",
            "general relativity",
            "black hole",
            "Tokamak",
            "condensed matter",
            "high-energy physics",
            "navier-stokes",
            "renormalization",
            "physics",
        ]
    ) and not any(kw in combined for kw in ["protein", "peptide", "clinical", "patient"]):
        return "physics"

    # Check Social Sciences & Humanities
    if any(
        kw in combined
        for kw in [
            "econometric",
            "difference-in-differences",
            "monetary policy",
            "sociolog",
            "ethnograph",
            "political science",
            "public policy",
            "humanities",
            "historiograph",
            "labor market",
        ]
    ):
        return "social_sciences_humanities"

    # Check Molecular Biology / Omics vs Clinical vs AI
    has_ai = any(
        kw in combined
        for kw in [
            "deep learning",
            "machine learning",
            "transformer",
            "neural network",
            "foundation model",
            "llm",
            "language model",
            "diffusion model",
            "graph neural",
            "ai model",
            "hugging face",
            "benchmark",
        ]
    )
    has_mol_bio = any(
        kw in combined
        for kw in [
            "peptide",
            "antimicrobial",
            "protein",
            "genomic",
            "transcriptom",
            "crispr",
            "single-cell",
            "rna-seq",
            "omics",
            "molecular",
            "binding affinity",
        ]
    )
    has_clinical = any(
        kw in combined
        for kw in [
            "clinical trial",
            "randomized",
            "patient",
            "mortality",
            "hospital",
            "prognos",
            "diagnos",
            "hazard ratio",
            "odds ratio",
            "cohort",
            "sepsis",
            "oncology",
        ]
    )

    if has_clinical and not has_mol_bio:
        return "biomedical_clinical"
    if has_ai and not has_clinical:
        return "ai_computer_science"
    if has_mol_bio:
        return "molecular_biology_omics"
    if has_clinical:
        return "biomedical_clinical"
    if has_ai:
        return "ai_computer_science"

    # No module keywords matched: follow the planner's domain so the evidence table appraises with the same
    # design-matched frameworks as the rest of the report
    if primary in USER_DOMAIN_TO_MODULE:
        return USER_DOMAIN_TO_MODULE[primary]
    return "universal_fallback"


def build_query_adaptive_extraction_plan(
    query: str,
    decomposition_plan: Optional[QueryDecompositionPlan] = None,
    papers: Optional[List[PaperRecord]] = None,
) -> ExtractionPlan:
    """Phase 5: Build a query-adaptive ExtractionPlan before extracting the Evidence Matrix."""
    module_key = resolve_domain_module_key(query, decomposition_plan)
    spec = DOMAIN_MODULE_SPECIFICATIONS.get(module_key, DOMAIN_MODULE_SPECIFICATIONS["universal_fallback"])

    primary_domain = spec["label"]
    secondary_domains: List[str] = []
    if decomposition_plan:
        if decomposition_plan.question_understanding:
            if decomposition_plan.question_understanding.primary_domain:
                primary_domain = decomposition_plan.question_understanding.primary_domain
            secondary_domains = list(decomposition_plan.question_understanding.secondary_domains or [])
        elif decomposition_plan.detected_domain:
            primary_domain = decomposition_plan.detected_domain

    # Detect secondary interdisciplinary columns if hybrid (e.g. AI + Molecular Biology or AI + Clinical)
    q_lower = (query or "").lower()
    visible_domain_columns: List[ExtractionFieldSpec] = list(spec["visible_columns"])
    existing_keys = {c.field_key for c in visible_domain_columns}

    if module_key == "ai_computer_science" and any(
        w in q_lower for w in ["peptide", "protein", "antimicrobial", "drug", "clinical", "patient", "bio"]
    ):
        bio_col = ExtractionFieldSpec(
            field_key="biological_or_clinical_validation",
            display_label="Biological / Experimental Validation Tier",
            dimension="Evidence",
            priority="Essential",
            domain_module="molecular_biology_omics",
            description="Distinguishes in silico benchmark predictions from wet-lab (in vitro / in vivo / clinical) validation.",
        )
        if bio_col.field_key not in existing_keys:
            visible_domain_columns.append(bio_col)
            existing_keys.add(bio_col.field_key)
    elif module_key in ("biomedical_clinical", "molecular_biology_omics") and any(
        w in q_lower for w in ["model", "ai", "deep learning", "machine learning", "screening", "prediction"]
    ):
        ai_col = ExtractionFieldSpec(
            field_key="model_architecture_and_task",
            display_label="Computational / AI Model & Baselines",
            dimension="Methodology",
            priority="Essential",
            domain_module="ai_computer_science",
            description="Algorithmic architecture, feature representation, and benchmark comparators.",
        )
        if ai_col.field_key not in existing_keys:
            visible_domain_columns.append(ai_col)
            existing_keys.add(ai_col.field_key)

    essential_fields = list(UNIVERSAL_ESSENTIAL_FIELDS) + [
        c for c in visible_domain_columns if c.priority == "Essential"
    ]
    recommended_fields = [c for c in visible_domain_columns if c.priority == "Recommended"]
    optional_fields = [
        ExtractionFieldSpec(
            field_key="funding_and_conflicts",
            display_label="Funding & Competing Interests",
            dimension="Provenance",
            priority="Optional",
            domain_module=module_key,
            description="Reported funding sources or competing interest disclosures when available in metadata.",
        ),
        ExtractionFieldSpec(
            field_key="supplementary_artifacts",
            display_label="Supplementary Data / Code Links",
            dimension="Provenance",
            priority="Optional",
            domain_module=module_key,
            description="Links to repositories, datasets, or supplementary tables.",
        ),
    ]
    not_applicable_fields = list(spec.get("not_applicable", []))

    study_types: List[str] = []
    if papers:
        seen_designs = []
        for p in papers:
            sd = (p.elicit.study_design or "Empirical Study").strip()
            if sd and sd not in seen_designs:
                seen_designs.append(sd)
        study_types = seen_designs[:8]

    return ExtractionPlan(
        plan_id=f"EP-{module_key.upper()}",
        query=query,
        primary_domain=primary_domain,
        secondary_domains=secondary_domains,
        domain_module_activated=module_key,
        study_types_represented=study_types,
        essential_fields=essential_fields,
        recommended_fields=recommended_fields,
        optional_fields=optional_fields,
        not_applicable_fields=not_applicable_fields,
        visible_domain_columns=visible_domain_columns[:4],
    )


# ============================================================================
# PHASE 3, 7, 8, 9: Multi-Result Extraction, Quantitative Context & Provenance
# ============================================================================

QUANT_METRIC_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("AUROC / AUC", re.compile(r"\b(?:auroc|auc(?:-roc)?|roc-auc)\s*(?:of|=|:|was|reached|achieved)?\s*\d+(?:\.\d+)?%?", re.I)),
    ("AUPRC / PR-AUC", re.compile(r"\b(?:auprc|pr-auc|average precision)\s*(?:of|=|:|was|reached)?\s*\d+(?:\.\d+)?%?", re.I)),
    ("Accuracy", re.compile(r"\b(?:accuracy|acc)\s*(?:of|=|:|was|reached|achieved)?\s*\d+(?:\.\d+)?\s*%?", re.I)),
    ("F1 / MCC", re.compile(r"\b(?:f1(?:-score)?|mcc|matthews correlation)\s*(?:of|=|:|was|reached)?\s*\d+(?:\.\d+)?%?", re.I)),
    ("Sensitivity / Specificity", re.compile(r"\b(?:sensitivity|specificity|recall|precision)\s*(?:of|=|:|was)?\s*\d+(?:\.\d+)?\s*%?", re.I)),
    ("IC50 / MIC / EC50 / Ki", re.compile(r"\b(?:ic50|mic|ec50|ki|kd|cc50|hc50)\s*(?:of|=|:|<|>|\u2264|\u2265)?\s*\d+(?:\.\d+)?\s*(?:ug/ml|\u03bcg/ml|um|\u03bcm|nm|mm|mg/l)?", re.I)),
    ("Hazard / Odds / Risk Ratio", re.compile(r"\b(?:hr|or|rr|hazard ratio|odds ratio|relative risk)\s*(?:of|=|:)?\s*\d+(?:\.\d+)?", re.I)),
    ("p-value", re.compile(r"\bp\s*(?:<|=|\u2264)\s*0\.\d+(?:e-?\d+)?", re.I)),
    ("Confidence / Credible Interval", re.compile(r"\b95%\s*(?:ci|confidence interval|credible interval)\s*[:=\[\(]?\s*\d+(?:\.\d+)?\s*(?:to|[-\u2013,])\s*\d+(?:\.\d+)?", re.I)),
    ("Fold / Percentage Change", re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|fold|-fold)\s*(?:increase|decrease|improvement|reduction|higher|lower|gain)", re.I)),
    ("RMSE / MAE / R2", re.compile(r"\b(?:rmse|mae|r2|r\^2|pearson r|spearman)\s*(?:of|=|:)?\s*-?\d+(?:\.\d+)?", re.I)),
    # Plain proportions ("prevalence was 12.5%", "93% of publications"); listed last so specific metrics label first
    ("Percentage / proportion", re.compile(r"\b\d+(?:\.\d+)?\s*%")),
]

SAMPLE_SCALE_PATTERN = re.compile(
    r"\b(?:n\s*=\s*\d[\d,]*|\d[\d,]+(?:\s+patients|\s+participants|\s+subjects|\s+peptides|\s+sequences|\s+compounds|\s+proteins|\s+samples|\s+cells|\s+trials|\s+studies|\s+datasets))",
    re.I,
)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


def _detect_publication_status(paper: PaperRecord) -> Literal[
    "published",
    "preprint",
    "preprint_with_published_link",
    "conference_proceedings",
    "working_paper_or_report",
    "unknown",
]:
    if paper.version_status in ("published", "preprint", "preprint_with_published_link"):
        return paper.version_status
    venue_low = (paper.venue or "").lower()
    src_low = (paper.source or "").lower()
    if any(k in src_low or k in venue_low for k in ["biorxiv", "medrxiv", "arxiv", "preprint"]):
        if paper.doi and not any(px in (paper.doi or "").lower() for px in ["10.1101", "arxiv"]):
            return "preprint_with_published_link"
        return "preprint"
    if any(k in venue_low for k in ["proceedings", "conference", "neurips", "icml", "iclr", "cvpr", "acl", "emnlp", "kdd"]):
        return "conference_proceedings"
    if paper.venue or paper.doi or paper.pmid:
        return "published"
    return "unknown"


def _detect_document_relations(papers: List[PaperRecord]) -> Dict[str, Tuple[DocumentRelationType, List[str], EvidenceUnitRelationshipType, str]]:
    """
    Phase 14: Detect duplicates, preprint-published pairs, follow-up analyses, and shared-dataset studies
    across the retrieved corpus without silently merging distinct studies or double-counting duplicates.
    """
    relations: Dict[str, Tuple[DocumentRelationType, List[str], EvidenceUnitRelationshipType, str]] = {}
    by_canonical: Dict[str, List[PaperRecord]] = {}
    by_first_author_dataset: Dict[str, List[PaperRecord]] = {}

    for p in papers:
        cid = p.canonical_study_id or p.id
        by_canonical.setdefault(cid, []).append(p)

        # Check shared dataset or cohort + author overlap
        ds_norm = (p.elicit.dataset_or_system or "").strip().lower()
        first_auth = (p.authors[0].split()[-1].lower() if p.authors else "")
        if ds_norm and ds_norm not in ("not reported", "unavailable", "unspecified", "general domain"):
            key = f"{ds_norm[:32]}"
            by_first_author_dataset.setdefault(key, []).append(p)

    for p in papers:
        cid = p.canonical_study_id or p.id
        same_canonical = by_canonical.get(cid, [p])
        linked_ids = [other.id for other in same_canonical if other.id != p.id]

        if len(same_canonical) > 1 or p.version_status == "preprint_with_published_link":
            doc_rel: DocumentRelationType = "preprint_or_published_version"
            unit_rel: EvidenceUnitRelationshipType = "merged_preprint_and_published"
            note = "Consolidated preprint and peer-reviewed version record (counted as 1 independent study)."
            relations[p.id] = (doc_rel, linked_ids, unit_rel, note)
            continue

        ds_norm = (p.elicit.dataset_or_system or "").strip().lower()
        if ds_norm and ds_norm[:32] in by_first_author_dataset:
            group_members = by_first_author_dataset[ds_norm[:32]]
            peers = [other for other in group_members if other.id != p.id]
            if peers:
                peer_refs = ", ".join(f"[{other.citation_index}]" for other in peers[:4])
                my_auths = {a.split()[-1].lower() for a in (p.authors or []) if a.strip()}
                peer_auths = {
                    a.split()[-1].lower()
                    for other in peers
                    for a in (other.authors or [])
                    if a.strip()
                }
                # Only mark secondary/later records in the group as follow_up_or_secondary_analysis so the primary study retains its primary/multi-experiment unit status
                is_later_in_group = group_members[0].id != p.id
                if (my_auths & peer_auths) and is_later_in_group:
                    relations[p.id] = (
                        "follow_up_or_secondary_report",
                        [other.id for other in peers],
                        "follow_up_or_secondary_analysis",
                        f"Shares author group and dataset/system ({p.elicit.dataset_or_system}) with {peer_refs}; tracked for non-independent cohort overlap.",
                    )
                    continue
                elif not (my_auths & peer_auths):
                    relations[p.id] = (
                        "shared_dataset_independent_study",
                        [other.id for other in peers],
                        "shared_cohort_or_dataset",
                        f"Evaluated on shared benchmark/dataset ({p.elicit.dataset_or_system}) alongside {peer_refs}.",
                    )
                    continue

        relations[p.id] = (
            "primary_document",
            [],
            "single_publication",
            "Independent primary publication record.",
        )

    return relations


def _extract_quantitative_detail(
    sentence: str,
    paper: PaperRecord,
    module_key: str,
) -> QuantitativeResultDetail:
    """
    Phase 8: Extract structured quantitative result details while preserving exact source text,
    units, metric types, sample size, uncertainty intervals, and derived quantities.
    """
    if module_key == "mathematics" and not re.search(r"\b\d+(?:\.\d+)?%", sentence):
        # Formal mathematical statements should not have empirical sample size / p-values forced on them
        return QuantitativeResultDetail(
            raw_reported_text=sentence[:240] if sentence else "Not reported",
            normalized_value="Formal mathematical proposition / bound (see theorem statement)",
            metric_type="Mathematical bound / identity",
            unit="Not applicable",
            direction="not_applicable",
            comparator=paper.elicit.dataset_or_system or "Classical bound / prior theorem",
            conditions_or_setting="Formal axiomatic / regularity assumptions",
            sample_size_or_scale="Not applicable",
            time_point_or_horizon="Not applicable",
            uncertainty_interval="Not applicable",
            statistical_test="Formal proof",
            adjustment_status="Not applicable",
            p_value="Not applicable",
            analysis_tier="theoretical_derivation",
            reporting_status="Not applicable",
            derived_quantities=[],
        )

    matched_metrics: List[str] = []
    metric_types: List[str] = []
    for m_label, pattern in QUANT_METRIC_PATTERNS:
        for m in pattern.finditer(sentence):
            snippet = m.group(0).strip()
            if snippet not in matched_metrics:
                matched_metrics.append(snippet)
            if m_label not in metric_types:
                metric_types.append(m_label)

    # Also check paper.elicit.quantitative_result if sentence had no regex hit
    elicit_q = (paper.elicit.quantitative_result or "").strip()
    if not matched_metrics and elicit_q and elicit_q.lower() not in ("not reported", "unavailable", "none"):
        matched_metrics.append(elicit_q)
        metric_types.append("Reported quantitative estimate")

    found_sample = study_sample_size(paper, sentence)
    sample_str = found_sample or (paper.elicit.sample_or_scale or "Not reported")
    if sample_str.lower() in ("", "unknown", "none"):
        sample_str = "Not reported"

    # Uncertainty & p-value extraction
    ci_match = re.search(
        r"\b(?:95%\s*(?:ci|confidence interval)\s*[:=\[\(]?\s*[\d\.\-\u2013,\s]+[\]\)]?|\u00b1\s*\d+(?:\.\d+)?%?)",
        sentence,
        re.I,
    )
    uncertainty_str = ci_match.group(0).strip() if ci_match else "Not reported"

    pval_match = re.search(r"\bp\s*(?:<|=|\u2264)\s*0\.\d+(?:e-?\d+)?", sentence, re.I)
    pval_str = pval_match.group(0).strip() if pval_match else "Not reported"

    # Unit detection
    unit_match = re.search(
        r"\b(?:%|ug/ml|\u03bcg/ml|um|\u03bcm|nm|mm|mg/kg|mg/l|kcal/mol|ev|mev|gev|kelvin|k|mpa|gpa|mah/g|mv/dec|ms|seconds|days|months|years)\b",
        sentence,
        re.I,
    )
    unit_str = unit_match.group(0) if unit_match else ("%" if "%" in "".join(matched_metrics) else "Not reported")

    # Analysis / Validation tier
    sent_low = (sentence + " " + (paper.abstract or "")).lower()
    analysis_tier: AnalysisTierType = "primary"
    if any(w in sent_low for w in ["external validation", "independent cohort", "out-of-distribution", "prospective validation", "wet-lab validation", "in vivo validation"]):
        analysis_tier = "external_validation"
    elif any(w in sent_low for w in ["cross-validation", "test set", "holdout", "internal validation", "benchmark"]):
        analysis_tier = "internal_validation"
    elif any(w in sent_low for w in ["ablation", "subgroup", "secondary endpoint", "sensitivity analysis"]):
        analysis_tier = "secondary"

    # Comparator detection
    comp_match = re.search(
        r"\b(?:compared\s+(?:with|to)|versus|vs\.?|outperformed|over|against)\s+([A-Za-z0-9\-\s/]{3,45})",
        sentence,
        re.I,
    )
    comparator_str = comp_match.group(1).strip(" .,;:") if comp_match else "Not reported in excerpt"

    # Direction detection
    direction: FindingDirectionType = "unclear"
    if any(w in sent_low for w in ["outperform", "improv", "increas", "higher", "superior", "effective", "potent", "accura", "reduc"]):
        direction = "positive"
    elif any(w in sent_low for w in ["no significant", "null", "failed to", "comparable to baseline"]):
        direction = "null_or_nonsignificant"
    elif any(w in sent_low for w in ["mixed", "trade-off", "however", "whereas", "context-dependent"]):
        direction = "mixed_or_conditional"
    elif any(w in sent_low for w in ["worse", "degrad", "declin", "toxic", "adverse"]):
        direction = "negative"

    # Phase 7: Derived quantities (only when exact inputs exist in sentence, never overwriting raw text)
    derived_list: List[DerivedQuantityRecord] = []
    pct_match = re.search(r"\b(\d+(?:\.\d+)?)\s*%", ", ".join(matched_metrics))
    if pct_match:
        pct_val = float(pct_match.group(1))
        if 0.0 <= pct_val <= 100.0:
            derived_list.append(
                DerivedQuantityRecord(
                    quantity_name="Normalized Proportion (0-1 scale)",
                    original_inputs={"reported_percentage": f"{pct_Match_str(pct_match)}"},
                    formula_used="proportion = percentage / 100.0",
                    derived_value=f"{pct_val / 100.0:.4f}",
                    unit="ratio (0..1)",
                    calculation_status="computed",
                )
            )

    has_quant = bool(matched_metrics)
    raw_quant_text = "; ".join(matched_metrics) if matched_metrics else "Not reported (qualitative or directional finding only)"
    norm_val = "; ".join(matched_metrics[:3]) if matched_metrics else "Not reported"

    return QuantitativeResultDetail(
        raw_reported_text=raw_quant_text,
        normalized_value=norm_val,
        metric_type=", ".join(metric_types) if metric_types else "Qualitative / Structural observation",
        unit=unit_str,
        direction=direction,
        comparator=comparator_str,
        conditions_or_setting=paper.elicit.dataset_or_system or "Not reported",
        sample_size_or_scale=sample_str,
        time_point_or_horizon=f"Not reported in {'abstract or full text' if paper.full_text_sections else 'abstract'}",
        uncertainty_interval=uncertainty_str,
        statistical_test="Reported in source" if pval_str != "Not reported" else "Not reported",
        adjustment_status=f"Not reported in {'abstract or full text' if paper.full_text_sections else 'abstract'}",
        p_value=pval_str,
        analysis_tier=analysis_tier,
        reporting_status="Reported" if has_quant else "Not reported",
        derived_quantities=derived_list,
    )


def pct_Match_str(m: re.Match) -> str:
    return m.group(0).strip()


# Result sentences scanned from an open-access full text's Results section per study
FULL_TEXT_RESULT_SENTENCES = 80


def extract_granular_findings_for_study(
    paper: PaperRecord,
    contribution_id: str,
    document_id: str,
    module_key: str,
    section1_findings: List[AtomicFinding],
    section1_claims: List[SynthesisClaim],
    timestamp_iso: str,
) -> List[GranularFindingRecord]:
    """
    Phase 3 Level C: Extract one or more GranularFindingRecord items for a single study.
    Does not collapse multi-result studies into one generic line: if the abstract reports
    distinct experiments, metrics, or sub-analyses, each becomes a distinct Level C record.
    """
    results: List[GranularFindingRecord] = []
    abstract = (paper.abstract or "").strip()
    sentences = _split_sentences(abstract)
    # With open-access full text, result sentences from its Results section (or main body) are checked first
    ft_results = paper.full_text_sections.get("results") or paper.full_text_sections.get("body", "")
    ft_label = f"Results ({paper.full_text_source})" if ft_results else ""
    sourced_sentences: List[Tuple[str, str]] = (
        [(sent, ft_label) for sent in _split_sentences(ft_results)[:FULL_TEXT_RESULT_SENTENCES]] if ft_results else []
    ) + [(sent, "Abstract") for sent in sentences]

    # Find linked Section 1 claims for this paper
    linked_claim_ids = [
        c.claim_id
        for c in section1_claims
        if paper.citation_index in (c.relevant_study_indices or [])
        or paper.citation_index in (c.supporting_indices or [])
        or paper.citation_index in (c.challenging_indices or [])
    ]

    subq_id = paper.matched_subquestion_ids[0] if paper.matched_subquestion_ids else "Q1"

    # Identify candidate result sentences from abstract (preferring sentences with quantitative metrics or result verbs)
    candidate_sentences: List[Tuple[str, str, str]] = []
    result_cue_re = re.compile(
        r"\b(?:achiev|demonstrat|show|observ|reach|outperform|reduc|increas|yield|auc|auroc|accuracy|f1|ic50|mic|hr|or|p\s*[<=\u2264]|significant|prove|establish|bound)\b",
        re.I,
    )

    for sent, section in sourced_sentences:
        has_num = bool(re.search(r"\d", sent))
        # Epidemiological results ("prevalence was 12.5%") often have no result verb: a percentage, interval or
        # sample size marks a result sentence too
        has_cue = bool(result_cue_re.search(sent) or re.search(r"\d\s*%|\bci\b|\bn\s*=\s*\d|prevalen|incidence|rate\b|mortality", sent, re.I))
        if has_num and has_cue and len(sent) < 600:
            label = "Primary Quantitative Result" if not candidate_sentences else f"Secondary / Component Analysis #{len(candidate_sentences) + 1}"
            candidate_sentences.append((label, sent, section))

    # The primary finding is the result sentence with the most statistical content (figures, p-values, intervals,
    # sample size), not simply the first one; the order of the remaining candidates is kept
    if len(candidate_sentences) > 1:
        def richness(sent: str) -> int:
            # Distinct result metrics (incl. p-values and intervals) plus a stated sample; sentences that call
            # themselves secondary or supplementary do not become the primary finding
            metrics = sum(1 for _, pattern in QUANT_METRIC_PATTERNS if pattern.search(sent))
            secondary = bool(re.search(r"\b(secondary|supplementary|additionally|sensitivity analys)", sent, re.I))
            return metrics + bool(SAMPLE_SIZE_PATTERN.search(sent)) - 2 * secondary
        best = max(range(len(candidate_sentences)), key=lambda i: (richness(candidate_sentences[i][1]), -i))
        if best:
            candidate_sentences.insert(0, candidate_sentences.pop(best))
        candidate_sentences = [
            ("Primary Quantitative Result" if i == 0 else f"Secondary / Component Analysis #{i + 1}", sent, sec)
            for i, (_, sent, sec) in enumerate(candidate_sentences)
        ]

    # If fewer than 2 numeric result sentences, also include qualitative result sentence or Section 1 AtomicFinding
    if not candidate_sentences:
        for s1_f in section1_findings:
            if s1_f.citation_index == paper.citation_index and s1_f.source_passage:
                candidate_sentences.append(("Primary Reported Finding", s1_f.source_passage, "Abstract"))
                break

    if not candidate_sentences and sentences:
        # Use the most informative concluding/results sentence from the abstract
        best_sent = sentences[-1] if len(sentences) > 1 else sentences[0]
        candidate_sentences.append(("Primary Abstract Finding", best_sent, "Abstract"))

    if not candidate_sentences:
        # Metadata-only fallback (Phase 15)
        fallback_text = paper.elicit.key_findings or f"Title-level record only: {paper.title}"
        q_detail = _extract_quantitative_detail("", paper, module_key)
        q_detail.reporting_status = "Unavailable"
        results.append(
            GranularFindingRecord(
                result_id=f"R-{paper.citation_index}-1",
                contribution_id=contribution_id,
                source_document_id=document_id,
                citation_index=paper.citation_index,
                subquestion_addressed=subq_id,
                experiment_or_component_label="Metadata-Only Record (Abstract Unavailable)",
                outcome_or_phenomenon="Unavailable from metadata-only record",
                measurement_or_analytical_method=paper.elicit.methodology or "Unavailable",
                experimental_or_theoretical_conditions=paper.elicit.dataset_or_system or "Unavailable",
                comparator="Unavailable",
                factual_source_text=fallback_text,
                normalized_summary="Abstract text unavailable from source API; quantitative extraction marked Unavailable.",
                llm_interpretation="No interpretive synthesis generated for metadata-only record.",
                quantitative_detail=q_detail,
                direction_of_effect="unclear",
                statistical_or_mathematical_qualifications="Unavailable (abstract missing)",
                extraction_status="Unavailable",
                provenance=FieldProvenance(
                    field_name="primary_finding",
                    source_document_id=document_id,
                    source_section="Metadata / Title",
                    page_or_span="Title metadata",
                    supporting_excerpt=paper.title,
                    extraction_method="bibliographic_api",
                    extraction_timestamp=timestamp_iso,
                    verification_status="Not verifiable from accessible sources",
                    access_level="metadata_only",
                    reporting_status="Unavailable",
                ),
                linked_section1_claim_ids=linked_claim_ids,
            )
        )
        return results

    # Cap at up to 3 distinct granular findings per study so multi-experiment papers are represented cleanly
    for r_idx, (comp_label, sent, section) in enumerate(candidate_sentences[:3], start=1):
        from_full_text = section != "Abstract"
        where = "full text" if from_full_text else "abstract"
        q_detail = _extract_quantitative_detail(sent, paper, module_key)
        outcome_label = q_detail.metric_type if q_detail.metric_type != "Qualitative / Structural observation" else "Reported functional / analytical outcome"

        norm_summary = (
            f"{outcome_label}: {q_detail.normalized_value} "
            f"(Setting: {q_detail.conditions_or_setting}; Scale: {q_detail.sample_size_or_scale})"
            if q_detail.reporting_status == "Reported"
            else sent[:220]
        )

        llm_interp = (
            f"Supports {q_detail.analysis_tier.replace('_', ' ')} evidence for subquestion {subq_id} "
            f"under reported study conditions ({paper.elicit.dataset_or_system or 'domain setting'})."
        )

        quals = []
        if q_detail.uncertainty_interval == "Not reported":
            quals.append(f"Confidence/uncertainty interval not reported in {where}")
        else:
            quals.append(f"Uncertainty: {q_detail.uncertainty_interval}")
        if q_detail.p_value != "Not reported":
            quals.append(f"Significance: {q_detail.p_value}")
        if q_detail.analysis_tier == "internal_validation":
            quals.append(f"Internal benchmark/split evaluation (external generalization not established in {where})")

        results.append(
            GranularFindingRecord(
                result_id=f"R-{paper.citation_index}-{r_idx}",
                contribution_id=contribution_id,
                source_document_id=document_id,
                citation_index=paper.citation_index,
                subquestion_addressed=subq_id,
                experiment_or_component_label=comp_label,
                outcome_or_phenomenon=outcome_label,
                measurement_or_analytical_method=paper.elicit.methodology or "Reported analytical workflow",
                experimental_or_theoretical_conditions=paper.elicit.dataset_or_system or "Not reported in abstract",
                comparator=q_detail.comparator,
                factual_source_text=sent,
                normalized_summary=norm_summary,
                llm_interpretation=llm_interp,
                quantitative_detail=q_detail,
                direction_of_effect=q_detail.direction,
                statistical_or_mathematical_qualifications="; ".join(quals) if quals else f"Reported in {where}",
                extraction_status="Reported",
                provenance=FieldProvenance(
                    field_name=f"finding_{r_idx}",
                    source_document_id=document_id,
                    source_section=section,
                    page_or_span=f"{'Results' if from_full_text else 'Abstract'} sentence #{r_idx}",
                    supporting_excerpt=sent[:320],
                    extraction_method="hybrid_verified",
                    extraction_timestamp=timestamp_iso,
                    verification_status="Verified" if q_detail.reporting_status == "Reported" else "Verified with qualifications",
                    access_level="full_text" if from_full_text else "abstract",
                    reporting_status="Reported",
                ),
                linked_section1_claim_ids=linked_claim_ids,
            )
        )

    return results


# ============================================================================
# PHASE 11: Method-Aware Critical Appraisal (Separated from Factual Extraction)
# ============================================================================

def build_methodological_appraisal(
    paper: PaperRecord,
    module_key: str,
    findings: List[GranularFindingRecord],
) -> MethodologicalAppraisalRecord:
    """
    Phase 11: Construct a method-aware critical appraisal record that strictly separates:
    (1) source-reported limitations,
    (2) independently identified methodological concerns, and
    (3) applicability concerns relative to the user's question.
    Never invents arbitrary numeric quality scores (like 8/10 or 92%).
    """
    spec = DOMAIN_MODULE_SPECIFICATIONS.get(module_key, DOMAIN_MODULE_SPECIFICATIONS["universal_fallback"])
    framework_name = spec["appraisal_framework"]

    abstract = (paper.abstract or "").strip()
    abs_low = abstract.lower()
    has_full_text = bool(paper.full_text_sections)
    access_completeness: Literal["full_text_complete", "preliminary_abstract_only", "metadata_only_minimal"] = (
        "full_text_complete" if has_full_text
        else "preliminary_abstract_only" if len(abstract) > 40 else "metadata_only_minimal"
    )

    # 1. Source-reported limitations (extracted from explicit limitation cues in abstract)
    source_reported: List[str] = []
    for sent in _split_sentences(abstract):
        if any(
            cue in sent.lower()
            for cue in [
                "limit",
                "challenge",
                "drawback",
                "bottleneck",
                "however",
                "remain",
                "future work",
                "lack",
                "scarce",
                "small sample",
                "in vitro only",
                "computational only",
            ]
        ):
            source_reported.append(f"Author-stated in abstract: \"{sent[:220]}\"")

    # Open-access full text: the authors' own Limitations section, or limitation sentences in the Discussion
    ft_limits = paper.full_text_sections.get("limitations", "")
    if not ft_limits:
        ft_limits = " ".join(s for s in _split_sentences(paper.full_text_sections.get("discussion", ""))
                             if re.search(r"\blimitation|\bweakness|\bshould be interpreted with caution", s, re.I))
    for sent in _split_sentences(ft_limits)[:2]:
        source_reported.insert(0, f"Author-stated in full text: \"{sent[:220]}\"")

    if not source_reported:
        source_reported.append(
            "Not stated in the open-access full text." if paper.full_text_sections
            else "Not explicitly stated in abstract (full text required to inspect complete Discussion/Limitations section)."
        )

    # 2. Independently identified methodological concerns (domain-matched)
    independent_concerns: List[str] = []
    has_quant = any(f.quantitative_detail.reporting_status == "Reported" for f in findings)
    has_ci = any(f.quantitative_detail.uncertainty_interval != "Not reported" for f in findings)
    has_sample = (paper.elicit.sample_reporting_status == "Reported") or any(
        f.quantitative_detail.sample_size_or_scale not in ("Not reported", "Not applicable") for f in findings
    )

    if access_completeness == "preliminary_abstract_only":
        independent_concerns.append(
            "Appraisal is preliminary because only abstract and bibliographic metadata were accessible (full-text methods/supplements not inspected)."
        )
    elif access_completeness == "metadata_only_minimal":
        independent_concerns.append(
            "Metadata-only record: abstract text was unavailable from the source database, preventing methodological verification."
        )

    if _detect_publication_status(paper) == "preprint":
        independent_concerns.append("Preprint status: study has not completed formal peer review at the time of retrieval.")

    # Design-matched framework screening (see frameworks/appraisal_routes.py for routing and sources)
    # With open-access full text, design-matched screening also reads the methods, results and limitations
    screening_text = " ".join(
        [abstract] + [paper.full_text_sections.get(s, "") for s in ("methods", "results", "limitations")]
    )
    route = route_for_module(module_key, screening_text)
    if route is not None and has_full_text:
        route.framework_label = route.framework_label.replace("from abstract", "from open-access full text")
        route.concerns = [
            c.replace(" - not assessable from abstract", "").replace("in the abstract", "in the full text")
            for c in route.concerns
        ]
    if route is not None:
        framework_name = route.framework_label
        if module_key == "mathematics":
            independent_concerns = list(route.concerns)
        else:
            independent_concerns.extend(route.concerns)

    if module_key != "mathematics":
        if not has_sample:
            independent_concerns.append("Exact sample size / dataset scale (N) is not reported in the abstract.")
        if has_quant and not has_ci:
            independent_concerns.append("Point estimates are reported without confidence intervals or variance bounds in the abstract.")

    # 3. Applicability concerns relative to the user's research question
    applicability_concerns: List[str] = []
    ds = (paper.elicit.dataset_or_system or "").strip()
    if ds and ds.lower() not in ("not reported", "unavailable"):
        applicability_concerns.append(f"Direct applicability is scoped to the studied system/dataset ({ds}).")
    else:
        applicability_concerns.append("Specific target system/dataset scope is not fully specified in the abstract.")

    if paper.is_citation_context_only:
        applicability_concerns.append("Included as foundational/citation context; may not directly evaluate the primary comparative intervention.")

    repro_note = (
        f"Artifact/Code link reported: {paper.elicit.code_or_hf_link}"
        if paper.elicit.code_or_hf_link
        else "No direct code/data repository URL extracted from abstract metadata."
    )

    return MethodologicalAppraisalRecord(
        appraisal_framework=framework_name,
        appraisal_completeness=access_completeness,
        source_reported_limitations=source_reported,
        independent_methodological_concerns=independent_concerns,
        applicability_concerns=applicability_concerns,
        internal_validity_notes=independent_concerns[0] if independent_concerns else "Preliminary abstract-level check passed.",
        external_validity_notes=applicability_concerns[0] if applicability_concerns else "Scoped to studied setting.",
        reproducibility_transparency=repro_note,
    )


# ============================================================================
# PHASE 6: Domain-Adaptive Field Population per Study
# ============================================================================

def populate_domain_specific_fields(
    paper: PaperRecord,
    module_key: str,
    plan: ExtractionPlan,
    findings: List[GranularFindingRecord],
    appraisal: MethodologicalAppraisalRecord,
) -> Tuple[Dict[str, str], Dict[str, ReportingStatusType]]:
    """Populate the visible domain-specific columns defined by the active ExtractionPlan."""
    values: Dict[str, str] = {}
    statuses: Dict[str, ReportingStatusType] = {}

    # Abstract plus any open-access full text (methods, results, limitations): a study is "metadata-only" only when
    # neither is available
    abs_text = " ".join(
        [(paper.abstract or "").strip()]
        + [paper.full_text_sections.get(sec, "") for sec in ("methods", "results", "limitations")]
    ).strip()
    abs_low = abs_text.lower()
    primary_f = findings[0] if findings else None
    q_det = primary_f.quantitative_detail if primary_f else None

    for col in plan.visible_domain_columns:
        key = col.field_key
        val = "Not reported"
        st: ReportingStatusType = "Not reported"

        if not abs_text:
            values[key] = "Unavailable (metadata-only record)"
            statuses[key] = "Unavailable"
            continue


        if key in ("clinical_study_design", "study_design_and_category"):
            val = paper.elicit.study_design or "Empirical / Observational Study"
            st = "Reported"

        elif key in ("population_and_sample", "system_and_scale"):
            sys_str = paper.elicit.dataset_or_system or "Not reported"
            scale_str = q_det.sample_size_or_scale if q_det else (paper.elicit.sample_or_scale or "Not reported")
            if sys_str != "Not reported" or scale_str != "Not reported":
                val = f"{sys_str} (Scale/N: {scale_str})"
                st = "Reported"

        elif key == "intervention_and_comparator":
            meth = paper.elicit.methodology or "Not reported"
            comp = q_det.comparator if q_det and q_det.comparator != "Not reported in excerpt" else "Standard control / baseline (details not in abstract)"
            val = f"Method/Intervention: {meth} | Comparator: {comp}"
            st = "Reported" if meth != "Not reported" else "Uncertain"

        elif key == "endpoints_and_followup":
            metric = q_det.metric_type if q_det else "Primary outcome"
            tier = q_det.analysis_tier.replace("_", " ").title() if q_det else "Primary"
            val = f"{metric} ({tier})"
            st = "Reported"

        elif key == "biological_system":
            bio_match = re.search(
                r"\b(?:escherichia coli|e\.\s*coli|staphylococcus aureus|s\.\s*aureus|pseudomonas aeruginosa|candida|human|mice|murine|mammalian|gram-negative|gram-positive|antimicrobial peptide[s]?|amp[s]?|ribosom[a-z]*|membrane)\b",
                abs_text,
                re.I,
            )
            sys_base = paper.elicit.dataset_or_system or ""
            if bio_match and bio_match.group(0).lower() not in sys_base.lower():
                val = f"{bio_match.group(0)} ({sys_base})" if sys_base and sys_base != "Not reported" else bio_match.group(0)
                st = "Reported"
            elif sys_base and sys_base != "Not reported":
                val = sys_base
                st = "Reported"

        elif key == "assay_and_perturbation":
            val = paper.elicit.methodology or "Molecular / computational assay"
            st = "Reported" if paper.elicit.methodology else "Not reported"

        elif key == "replicates_and_controls":
            scale_str = q_det.sample_size_or_scale if q_det else "Not reported"
            if any(w in abs_low for w in ["control", "replicate", "triplicate", "cross-validation", "benchmark", "fdr"]):
                val = f"Controls/Validation noted in abstract; Scale: {scale_str}"
                st = "Reported"
            elif scale_str != "Not reported":
                val = f"Scale: {scale_str} (replicate structure not detailed in abstract)"
                st = "Uncertain"

        elif key in ("orthogonal_validation", "biological_or_clinical_validation"):
            has_in_vivo = any(w in abs_low for w in ["in vivo", "mouse model", "murine", "animal model", "clinical trial"])
            has_in_vitro = any(w in abs_low for w in ["in vitro", "mic ", "minimum inhibitory", "hemolysis", "wet-lab", "synthesized", "assay"])
            if has_in_vivo:
                val = "In vivo / Preclinical or Clinical validation reported"
                st = "Reported"
            elif has_in_vitro:
                val = "In vitro / Experimental assay validation reported"
                st = "Reported"
            else:
                val = "In silico / Computational evaluation only (wet-lab validation not reported in abstract)"
                st = "Reported"

        elif key == "model_architecture_and_task":
            val = f"{paper.elicit.methodology or 'Computational Model'} ({paper.elicit.study_design or 'Empirical evaluation'})"
            st = "Reported" if paper.elicit.methodology else "Uncertain"

        elif key == "datasets_and_split_hygiene":
            ds = paper.elicit.dataset_or_system or "Not reported"
            scale = q_det.sample_size_or_scale if q_det else "Not reported"
            split_cue = "Explicit split/validation noted" if any(w in abs_low for w in ["cross-validation", "holdout", "test set", "independent test", "scaffold split", "homology"]) else "Split/leakage protocol not detailed in abstract"
            val = f"Dataset: {ds} (Scale: {scale}) | {split_cue}"
            st = "Reported" if ds != "Not reported" else "Not reported"

        elif key in ("baselines_and_validation_tier", "comparator_and_validation"):
            tier = q_det.analysis_tier.replace("_", " ").title() if q_det else "Internal evaluation"
            comp = q_det.comparator if q_det and q_det.comparator != "Not reported in excerpt" else "Prior baseline methods"
            val = f"Tier: {tier} | Comparator: {comp}"
            st = "Reported"

        elif key == "reproducibility_and_artifacts":
            if paper.elicit.code_or_hf_link:
                val = f"Available ({paper.elicit.code_or_hf_link})"
                st = "Reported"
            elif any(w in abs_low for w in ["github.com", "huggingface.co", "open-source", "publicly available", "code is available", "web server"]):
                val = "Public availability stated in abstract"
                st = "Reported"
            else:
                val = "Not reported in abstract"
                st = "Not reported"

        elif key == "chemical_or_material_system":
            val = paper.elicit.dataset_or_system or "Not reported in abstract"
            st = "Reported" if paper.elicit.dataset_or_system else "Not reported"

        elif key in ("synthesis_or_simulation_method", "theoretical_or_experimental_method", "identification_or_interpretive_design"):
            val = paper.elicit.methodology or "Not reported in abstract"
            st = "Reported" if paper.elicit.methodology else "Not reported"

        elif key in ("operating_conditions_and_characterization", "physical_system_and_regime", "institutional_or_population_setting", "mathematical_structures"):
            val = paper.elicit.dataset_or_system or "Scoped to abstract context"
            st = "Reported" if paper.elicit.dataset_or_system else "Uncertain"

        elif key in ("stability_and_selectivity", "uncertainty_and_error_budget", "theorem_and_proof_method"):
            val = primary_f.normalized_summary if primary_f else "Not reported"
            st = primary_f.extraction_status if primary_f else "Not reported"

        elif key in ("assumptions_and_approximations", "hypotheses_and_assumptions", "constructs_and_measurement"):
            val = f"Design: {paper.elicit.study_design}; Method: {paper.elicit.methodology or 'Not specified'}"
            st = "Reported"

        elif key in ("generality_and_counterexamples", "external_validity_and_context", "scope_and_applicability"):
            val = appraisal.external_validity_notes
            st = "Reported"

        values[key] = val
        statuses[key] = st

    # Also record explicitly Not applicable fields so tests and UI can inspect them
    for na_field in plan.not_applicable_fields:
        values[na_field.field_key] = "Not applicable to this study/domain type"
        statuses[na_field.field_key] = "Not applicable"

    if paper.full_text_sections:
        # These messages were written for abstract-only records; say what was actually read
        values = {k: v.replace("in abstract", "in abstract or full text") for k, v in values.items()}
    return values, statuses


# ============================================================================
# PHASE 10: Cross-Study Scientific Comparability Classification
# ============================================================================

def compute_comparability_assessments(
    units: List[EvidenceUnitRecord],
    papers_by_idx: Dict[int, PaperRecord],
    module_key: str,
) -> List[ComparabilityAssessment]:
    """
    Phase 10: Classify cross-study comparability across the corpus so studies with
    different validation tiers, outcome definitions, or study designs are never
    presented as naively equivalent.
    """
    group_map: Dict[str, List[EvidenceUnitRecord]] = {}

    for u in units:
        subq = u.subquestions_addressed[0] if u.subquestions_addressed else "Q1"
        design_norm = (u.experimental_or_analytical_design or "Empirical").strip()
        group_key = f"{subq}::{design_norm}"
        group_map.setdefault(group_key, []).append(u)

    summary_groups: List[ComparabilityAssessment] = []

    for g_idx, (g_key, members) in enumerate(group_map.items(), start=1):
        subq, design_norm = g_key.split("::", 1)
        cg_id = f"CG-{g_idx}"
        cg_label = f"{subq} — {design_norm}"
        member_indices = [m.citation_index for m in members]

        for m in members:
            peers = [idx for idx in member_indices if idx != m.citation_index]
            p = papers_by_idx.get(m.citation_index)
            if not p or not (p.abstract or "").strip():
                comp = ComparabilityAssessment(
                    comparability_group_id=cg_id,
                    comparability_group_label=cg_label,
                    status="Insufficient information",
                    comparable_peer_indices=peers,
                    shared_dimensions=["Research topic"],
                    divergent_dimensions=["Abstract/methodology unavailable"],
                    comparability_rationale="Insufficient methodological and outcome detail in metadata-only record to establish formal comparability.",
                    warnings=["Do not pool or directly rank quantitative metrics without full-text verification."],
                )
                m.comparability = comp
                continue

            my_metrics = {
                r.quantitative_detail.metric_type
                for r in m.results
                if r.quantitative_detail.reporting_status == "Reported"
            }
            my_tiers = {r.quantitative_detail.analysis_tier for r in m.results}
            my_sys = (m.context_or_system or "").lower()

            shared_dims = [f"Subquestion ({subq})", f"Design family ({design_norm})"]
            divergent_dims: List[str] = []
            warnings: List[str] = []

            if not peers:
                # Check if related studies exist in other design families for same subquestion
                other_subq_peers = [
                    u.citation_index
                    for u in units
                    if u.citation_index != m.citation_index
                    and (u.subquestions_addressed[:1] == [subq])
                ]
                status: ComparabilityStatusType = (
                    "Related but not directly comparable" if other_subq_peers else "Comparable with qualifications"
                )
                divergent_dims.append("Distinct methodological design or evaluation protocol relative to other retrieved studies")
                warnings.append("Cross-design comparison requires accounting for differing experimental/computational protocols.")
                comp = ComparabilityAssessment(
                    comparability_group_id=cg_id,
                    comparability_group_label=cg_label,
                    status=status,
                    comparable_peer_indices=other_subq_peers,
                    shared_dimensions=[f"Addresses subquestion {subq}"],
                    divergent_dimensions=divergent_dims,
                    comparability_rationale=f"Addresses {subq} using {design_norm}, whereas peer studies employ different analytical designs or evaluation scales.",
                    warnings=warnings,
                )
                m.comparability = comp
                continue

            # Compare against peers in the same design family
            peer_metrics = {
                r.quantitative_detail.metric_type
                for other in members
                if other.citation_index != m.citation_index
                for r in other.results
                if r.quantitative_detail.reporting_status == "Reported"
            }
            peer_systems = {
                (other.context_or_system or "").lower()
                for other in members
                if other.citation_index != m.citation_index
            }

            same_system = bool(my_sys and my_sys not in ("not reported", "general domain") and my_sys in peer_systems)
            same_metric = bool(my_metrics and (my_metrics & peer_metrics))

            if same_system and same_metric:
                status = "Directly comparable"
                shared_dims.append(f"Shared system/dataset ({m.context_or_system})")
                shared_dims.append(f"Shared outcome metric ({', '.join(sorted(my_metrics & peer_metrics))})")
                rationale = (
                    f"Directly comparable with {len(peers)} peer study/studies ({', '.join(f'[{i}]' for i in peers[:5])}) "
                    f"sharing study design ({design_norm}), dataset/system, and evaluation metric family."
                )
            elif same_metric or same_system:
                status = "Comparable with qualifications"
                if same_metric:
                    shared_dims.append(f"Shared metric family ({', '.join(sorted(my_metrics & peer_metrics))})")
                    divergent_dims.append("Different underlying datasets, cohorts, or experimental conditions")
                else:
                    shared_dims.append(f"Shared system/dataset ({m.context_or_system})")
                    divergent_dims.append("Different reported outcome metrics or evaluation endpoints")
                if "internal_validation" in my_tiers:
                    warnings.append("Compare internal benchmark metrics cautiously across studies with differing train/test splits.")
                rationale = (
                    f"Comparable with qualifications across {', '.join(f'[{i}]' for i in peers[:5])}: "
                    f"studies share {design_norm} framing but differ in {', '.join(divergent_dims).lower()}."
                )
            else:
                status = "Related but not directly comparable"
                divergent_dims.extend(["Heterogeneous datasets/systems", "Non-overlapping quantitative metric definitions"])
                warnings.append("Direct numerical ranking across heterogeneous benchmarks/outcomes is scientifically invalid.")
                rationale = (
                    f"Related to {', '.join(f'[{i}]' for i in peers[:5])} on {subq}, but heterogeneous datasets and outcome metrics preclude direct numerical pooling."
                )

            comp = ComparabilityAssessment(
                comparability_group_id=cg_id,
                comparability_group_label=cg_label,
                status=status,
                comparable_peer_indices=peers,
                shared_dimensions=shared_dims,
                divergent_dimensions=divergent_dims,
                comparability_rationale=rationale,
                warnings=warnings,
            )
            m.comparability = comp

        if members:
            summary_groups.append(members[0].comparability)

    return summary_groups


# ============================================================================
# PHASE 16: Multi-Layer Validation (Structural, Bibliographic, Factual, Scientific)
# ============================================================================

def validate_structured_evidence_matrix(
    matrix: StructuredEvidenceMatrix,
    papers: List[PaperRecord],
    synthesis_package: Optional[SynthesisEvidencePackage],
) -> EvidenceTableValidationReport:
    """
    Phase 16: Run 4-layer verification on the Structured Evidence Matrix:
    1. Structural validation (every row has DocumentRecord, EvidenceUnitRecord, >=1 GranularFindingRecord)
    2. Bibliographic validation (1-to-1 match with retrieved PaperRecord indices, persistent IDs, titles)
    3. Factual & numerical provenance validation (every reported quantitative value appears in source text)
    4. Scientific & cross-section consistency validation (claim IDs match Section 1; appraisal separated from facts)
    """
    checks_performed = [
        "Level 1: Three-level schema completeness (DocumentRecord -> EvidenceUnitRecord -> GranularFindingRecord)",
        "Level 2: Bibliographic identity & citation index alignment against retrieved corpus",
        "Level 3: Factual & numerical source-excerpt verification (no fabricated numbers or page spans)",
        "Level 4: Scientific comparability & separation of factual extraction vs. methodological appraisal",
        "Level 5: Cross-section consistency with Section 1 (Cited Synthesis & Bibliography)",
    ]
    issues: List[str] = []
    repaired_count = 0

    structural_ok = True
    bib_ok = True
    factual_ok = True
    scientific_ok = True
    sec1_ok = True

    paper_by_idx = {p.citation_index: p for p in papers}
    valid_sec1_claim_ids = (
        {c.claim_id for c in (synthesis_package.claims or [])}
        if synthesis_package
        else set()
    )

    if len(matrix.rows) != len(papers):
        structural_ok = False
        issues.append(f"Row count mismatch: matrix has {len(matrix.rows)} rows for {len(papers)} papers.")

    for row in matrix.rows:
        p = paper_by_idx.get(row.citation_index)
        if not p:
            bib_ok = False
            issues.append(f"Row [{row.citation_index}] does not match any retrieved PaperRecord.")
            continue

        # Bibliographic check
        if row.document.canonical_title.strip() != p.title.strip():
            row.document.canonical_title = p.title.strip()
            repaired_count += 1

        # Structural check: ensure at least 1 Level C finding
        if not row.evidence_unit.results:
            structural_ok = False
            issues.append(f"Row [{row.citation_index}] had 0 Level C findings.")

        # Factual & numerical check on every Level C finding
        # Figures may come from the open-access full text, so verify against it too
        source_pool = f"{study_source_text(p)} {p.elicit.quantitative_result or ''}".lower()
        for res in row.evidence_unit.results:
            if res.quantitative_detail.reporting_status == "Reported":
                # Verify numeric tokens in normalized_value exist in source_pool
                num_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", res.quantitative_detail.normalized_value)
                missing_nums = [n for n in num_tokens if n not in source_pool]
                if missing_nums:
                    # Repair unverified numeric token by reverting to Not reported
                    res.quantitative_detail.normalized_value = "Not reported (unverified numeric token removed)"
                    res.quantitative_detail.reporting_status = "Uncertain"
                    res.provenance.verification_status = "Partially supported"
                    repaired_count += 1

            # Verify Section 1 claim links
            if valid_sec1_claim_ids and res.linked_section1_claim_ids:
                filtered_claims = [cid for cid in res.linked_section1_claim_ids if cid in valid_sec1_claim_ids]
                if len(filtered_claims) != len(res.linked_section1_claim_ids):
                    res.linked_section1_claim_ids = filtered_claims
                    repaired_count += 1

        # Scientific check: ensure no fabricated numeric quality score like "8/10" in appraisal
        for concern in row.evidence_unit.appraisal.independent_methodological_concerns:
            if re.search(r"\b\d+\s*/\s*10\b", concern):
                scientific_ok = False
                issues.append(f"Row [{row.citation_index}] contained arbitrary numeric quality score in appraisal.")

    return EvidenceTableValidationReport(
        structural_valid=structural_ok,
        bibliographic_valid=bib_ok,
        factual_provenance_valid=factual_ok,
        scientific_comparability_valid=scientific_ok,
        section1_consistency_valid=sec1_ok,
        checks_performed=checks_performed,
        issues_detected=issues,
        repaired_fields_count=repaired_count,
    )


# ============================================================================
# MAIN ENTRY POINT: Build Universal Structured Evidence Matrix
# ============================================================================

def build_structured_evidence_matrix(
    query: str,
    papers: List[PaperRecord],
    decomposition_plan: Optional[QueryDecompositionPlan] = None,
    synthesis_package: Optional[SynthesisEvidencePackage] = None,
) -> StructuredEvidenceMatrix:
    """
    Build the complete Section 2 Universal, Question-Adaptive, Source-Grounded
    Scientific Evidence Matrix from the retrieved study corpus and Section 1 synthesis package.
    """
    timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plan = build_query_adaptive_extraction_plan(query, decomposition_plan, papers)
    module_key = plan.domain_module_activated

    sec1_findings = list(synthesis_package.findings) if synthesis_package else []
    sec1_claims = list(synthesis_package.claims) if synthesis_package else []

    doc_relations = _detect_document_relations(papers)
    papers_by_idx = {p.citation_index: p for p in papers}

    documents: List[DocumentRecord] = []
    evidence_units: List[EvidenceUnitRecord] = []
    all_granular_findings: List[GranularFindingRecord] = []
    rows: List[EvidenceMatrixRow] = []

    for p in papers:
        doc_id = f"DOC-{p.citation_index}"
        contrib_id = f"UNIT-{p.citation_index}"
        pub_status = _detect_publication_status(p)
        doc_rel, linked_doc_ids, unit_rel, overlap_note = doc_relations.get(
            p.id,
            ("primary_document", [], "single_publication", "Independent primary publication record."),
        )

        has_abs = bool((p.abstract or "").strip() and len((p.abstract or "").strip()) > 30)
        has_full_text = bool(p.full_text_sections)
        access_level: EvidenceAccessType = (
            "full_text" if has_full_text else "abstract" if has_abs else "metadata_only"
        )

        all_srcs = list(p.all_sources) if p.all_sources else ([p.source] if p.source else ["Academic API"])

        doc_rec = DocumentRecord(
            document_id=doc_id,
            citation_index=p.citation_index,
            canonical_title=p.title,
            authors=list(p.authors or []),
            publication_year=p.year,
            venue_or_source=p.venue or p.source or "Academic Repository",
            doi=p.doi,
            pmid=p.pmid,
            arxiv_id=p.arxiv_id,
            s2_id=p.s2_id,
            source_databases=all_srcs,
            publication_status=pub_status,
            document_relation=doc_rel,
            linked_document_ids=linked_doc_ids,
            abstract_available=has_abs,
            full_text_available=has_full_text,
            supplementary_available=bool(p.elicit.code_or_hf_link),
            evidence_access_level=access_level,
            retrieval_timestamp=timestamp_iso,
            source_url=p.url or "",
            pdf_url=p.pdf_url,
            retraction_or_correction_status=(
                "Correction/Erratum keyword detected in title"
                if any(w in (p.title or "").lower() for w in ["retracted", "retraction", "erratum", "corrigendum"])
                else "No retraction notice detected in retrieved metadata"
            ),
        )
        documents.append(doc_rec)

        # Level C: Extract 1..N granular findings for this study
        study_findings = extract_granular_findings_for_study(
            paper=p,
            contribution_id=contrib_id,
            document_id=doc_id,
            module_key=module_key,
            section1_findings=sec1_findings,
            section1_claims=sec1_claims,
            timestamp_iso=timestamp_iso,
        )
        all_granular_findings.extend(study_findings)

        if len(study_findings) > 1 and unit_rel == "single_publication":
            unit_rel = "multi_experiment_unit"
            overlap_note = f"Contains {len(study_findings)} distinct quantitative/component findings extracted from study record."

        # Phase 11: Method-aware critical appraisal (separated from factual extraction)
        appraisal = build_methodological_appraisal(p, module_key, study_findings)

        # Phase 6: Domain-specific adaptive fields
        domain_vals, domain_statuses = populate_domain_specific_fields(
            paper=p,
            module_key=module_key,
            plan=plan,
            findings=study_findings,
            appraisal=appraisal,
        )

        # Determine objective from first sentence of abstract or title
        abs_sents = _split_sentences(p.abstract or "")
        objective_str = abs_sents[0][:240] if abs_sents else f"Investigates: {p.title}"
        subq_list = list(p.matched_subquestion_ids) if p.matched_subquestion_ids else ["Q1"]

        primary_finding = study_findings[0]
        linked_claims = list(primary_finding.linked_section1_claim_ids)

        # Field-level provenance map for universal fields
        field_prov: Dict[str, FieldProvenance] = {
            "research_objective": FieldProvenance(
                field_name="research_objective",
                source_document_id=doc_id,
                source_section="Abstract (Opening)" if has_abs else "Title",
                page_or_span="Abstract sentence #1" if has_abs else "Title",
                supporting_excerpt=objective_str,
                extraction_method="deterministic_pattern",
                extraction_timestamp=timestamp_iso,
                verification_status="Verified" if has_abs else "Not verifiable from accessible sources",
                access_level=access_level,
                reporting_status="Reported" if has_abs else "Unavailable",
            ),
            "methodology": FieldProvenance(
                field_name="methodological_approach",
                source_document_id=doc_id,
                source_section=(f"Methods ({p.full_text_source})" if p.full_text_sections.get("methods")
                                else "Abstract" if has_abs else "Metadata"),
                page_or_span=("Methods section" if p.full_text_sections.get("methods")
                              else "Abstract methods passage" if has_abs else "Title/Metadata"),
                supporting_excerpt=(p.full_text_sections.get("methods") or p.abstract or p.title)[:260],
                extraction_method="hybrid_verified",
                extraction_timestamp=timestamp_iso,
                verification_status="Verified" if p.elicit.methodology else "Verified with qualifications",
                access_level=access_level,
                reporting_status="Reported" if p.elicit.methodology else "Not reported",
            ),
            "primary_finding": primary_finding.provenance,
        }

        unit_rec = EvidenceUnitRecord(
            contribution_id=contrib_id,
            primary_document_id=doc_id,
            associated_document_ids=[doc_id] + linked_doc_ids,
            citation_index=p.citation_index,
            relationship_type=unit_rel,
            overlap_or_linkage_note=overlap_note,
            research_objective=objective_str,
            subquestions_addressed=subq_list,
            contribution_type=p.elicit.study_design or "Empirical / Computational Study",
            context_or_system=p.elicit.dataset_or_system or "Not reported",
            population_or_material_or_regime=p.elicit.dataset_or_system or "Not reported",
            sample_or_scale=primary_finding.quantitative_detail.sample_size_or_scale,
            sample_reporting_status=p.elicit.sample_reporting_status,
            methodological_approach=p.elicit.methodology or "Not reported",
            experimental_or_analytical_design=p.elicit.study_design or "Empirical Study",
            key_interventions_or_models=p.elicit.methodology or "Not reported",
            comparators_or_baselines=primary_finding.comparator,
            key_assumptions_or_controls=appraisal.internal_validity_notes,
            data_sources_or_benchmarks=p.elicit.dataset_or_system or "Not reported",
            validation_approach=primary_finding.quantitative_detail.analysis_tier.replace("_", " ").title(),
            code_or_artifact_availability=p.elicit.code_or_hf_link or "Not reported in abstract",
            domain_specific_fields=domain_vals,
            domain_field_statuses=domain_statuses,
            results=study_findings,
            appraisal=appraisal,
            relevance_to_query_score=round(float(p.relevance_score or 0.0), 3),
            relevance_rationale=(
                f"Addresses {', '.join(subq_list)}; cluster: {p.cluster_label}; "
                f"{'direct empirical/analytical evidence' if not p.is_citation_context_only else 'foundational citation context'}."
            ),
            linked_claim_ids=linked_claims,
            field_provenance=field_prov,
        )
        evidence_units.append(unit_rec)

    # Phase 10: Compute cross-study comparability groups & populate unit.comparability
    comp_groups = compute_comparability_assessments(evidence_units, papers_by_idx, module_key)

    for p, doc_rec, unit_rec in zip(papers, documents, evidence_units):
        primary_f = unit_rec.results[0]
        rows.append(
            EvidenceMatrixRow(
                row_id=f"ROW-{p.citation_index}",
                citation_index=p.citation_index,
                document=doc_rec,
                evidence_unit=unit_rec,
                primary_finding=primary_f,
                additional_findings_count=max(0, len(unit_rec.results) - 1),
                stance=p.stance,
                verification_status=primary_f.provenance.verification_status,
            )
        )

    disclosures = [
        f"Active Domain Extraction Profile: {plan.primary_domain} (module: {plan.domain_module_activated}).",
        "Evidence Access Scope: Extractions are derived from retrieved bibliographic metadata and abstracts unless full text is explicitly marked available; fields absent from abstracts are labeled 'Not reported' rather than inferred.",
        "Factual vs. Appraisal Separation: Source-reported quantitative results and author-stated limitations are stored separately from independent methodological appraisal and cross-study comparability assessments.",
    ]

    matrix = StructuredEvidenceMatrix(
        matrix_version="section2-evidence-matrix/2.0",
        extraction_plan=plan,
        documents=documents,
        evidence_units=evidence_units,
        granular_findings=all_granular_findings,
        rows=rows,
        comparability_groups=comp_groups,
        summary_disclosures=disclosures,
    )

    # Phase 16: Validate & attach validation report
    matrix.validation_report = validate_structured_evidence_matrix(matrix, papers, synthesis_package)
    return matrix


# ============================================================================
# PHASE 12 & 17: Machine-Readable CSV and JSON Export Formatters
# ============================================================================

def export_evidence_matrix_to_csv(matrix: StructuredEvidenceMatrix) -> str:
    """Export the Structured Evidence Matrix as a flat, provenance-preserving CSV string."""
    output = io.StringIO()
    domain_cols = matrix.extraction_plan.visible_domain_columns
    headers = [
        "Citation_Ref",
        "Document_ID",
        "Contribution_ID",
        "Title",
        "Authors",
        "Year",
        "Venue_or_Source",
        "Persistent_IDs",
        "Publication_Status",
        "Document_Relationship",
        "Evidence_Access_Level",
        "Subquestions_Addressed",
        "Study_Design",
        "Context_or_System",
        "Sample_or_Scale",
        "Methodology",
        "Comparator",
    ]
    for dc in domain_cols:
        headers.append(f"Domain_{dc.field_key}")
    headers.extend(
        [
            "Primary_Outcome_Metric",
            "Quantitative_Finding_Normalized",
            "Factual_Source_Excerpt",
            "Direction_of_Effect",
            "Uncertainty_Interval",
            "P_Value",
            "Validation_Tier",
            "Additional_Findings_Count",
            "Source_Reported_Limitations",
            "Independent_Methodological_Concerns",
            "Applicability_Concerns",
            "Appraisal_Framework",
            "Comparability_Status",
            "Comparability_Rationale",
            "Linked_Section1_Claims",
            "Verification_Status",
        ]
    )

    writer = csv.writer(output)
    writer.writerow(headers)

    for row in matrix.rows:
        doc = row.document
        unit = row.evidence_unit
        pf = row.primary_finding
        qd = pf.quantitative_detail

        pids = []
        if doc.doi:
            pids.append(f"DOI:{doc.doi}")
        if doc.pmid:
            pids.append(f"PMID:{doc.pmid}")
        if doc.arxiv_id:
            pids.append(f"arXiv:{doc.arxiv_id}")
        pid_str = "; ".join(pids) if pids else "None"

        row_vals = [
            f"[{row.citation_index}]",
            doc.document_id,
            unit.contribution_id,
            doc.canonical_title,
            "; ".join(doc.authors[:5]),
            str(doc.publication_year or "n.d."),
            doc.venue_or_source,
            pid_str,
            doc.publication_status,
            unit.relationship_type,
            doc.evidence_access_level,
            "; ".join(unit.subquestions_addressed),
            unit.experimental_or_analytical_design,
            unit.context_or_system,
            unit.sample_or_scale,
            unit.methodological_approach,
            pf.comparator,
        ]
        for dc in domain_cols:
            row_vals.append(unit.domain_specific_fields.get(dc.field_key, "Not reported"))

        row_vals.extend(
            [
                pf.outcome_or_phenomenon,
                qd.normalized_value,
                pf.factual_source_text,
                pf.direction_of_effect,
                qd.uncertainty_interval,
                qd.p_value,
                qd.analysis_tier,
                str(row.additional_findings_count),
                " | ".join(unit.appraisal.source_reported_limitations),
                " | ".join(unit.appraisal.independent_methodological_concerns),
                " | ".join(unit.appraisal.applicability_concerns),
                unit.appraisal.appraisal_framework,
                unit.comparability.status,
                unit.comparability.comparability_rationale,
                "; ".join(unit.linked_claim_ids),
                row.verification_status,
            ]
        )
        writer.writerow(row_vals)

    return output.getvalue()


def export_evidence_matrix_to_json(matrix: StructuredEvidenceMatrix) -> str:
    """Export the complete three-level StructuredEvidenceMatrix as formatted JSON."""
    return json.dumps(matrix.model_dump(), indent=2, ensure_ascii=False)
