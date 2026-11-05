from dataclasses import dataclass, field, replace
from typing import List, Dict, Any
from models import DomainCategoryType, MethodologicalProfileType
from frameworks.rules import method_rule_texts, rule_texts


@dataclass(frozen=True)
class DomainProfile:
    domain_id: DomainCategoryType
    display_name: str
    legacy_label: str
    implementation_tier: str  # "specialized" | "generic_fallback"
    terminology_ontologies: List[str]
    concept_normalization_map: Dict[str, str]
    validated_mesh_mappings: Dict[str, str]
    arxiv_category_hints: List[str]
    preferred_sources: List[str]
    evidence_types: List[str]
    extraction_columns: List[str]
    gap_identification_rules: List[str]
    default_gap_rows: List[str]
    default_gap_cols: List[str]
    requires_sample_size: bool = True
    # Filled from the source-cited registry (frameworks.rules) below; never written inline
    methodological_appraisal_rules: List[str] = field(default_factory=list)
    synthesis_interpretation_rules: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MethodologicalProfile:
    method_id: MethodologicalProfileType
    display_name: str
    required_evidence_elements: List[str]
    sample_reporting_policy: str  # "required" | "optional" | "not_applicable"
    priority_weight: int = 5
    # Filled from the source-cited registry (frameworks.rules) below; never written inline
    appraisal_criteria: List[str] = field(default_factory=list)
    synthesis_qualification_rule: str = ""


DOMAIN_PROFILES: Dict[DomainCategoryType, DomainProfile] = {
    "biomedical_clinical": DomainProfile(
        domain_id="biomedical_clinical",
        display_name="Biomedical & Clinical Sciences",
        legacy_label="BIOMEDICAL/AI",
        implementation_tier="specialized",
        terminology_ontologies=["MeSH", "ChEBI", "UniProt", "Gene Ontology"],
        concept_normalization_map={
            "amp": "antimicrobial peptide",
            "amps": "antimicrobial peptides",
            "rct": "randomized controlled trial",
            "moa": "mechanism of action",
            "mic": "minimum inhibitory concentration",
            "ic50": "half maximal inhibitory concentration",
            "pk": "pharmacokinetics",
            "pd": "pharmacodynamics",
            "admet": "absorption distribution metabolism excretion toxicity",
        },
        validated_mesh_mappings={
            "antimicrobial peptide": '"Antimicrobial Cationic Peptides"[MeSH Terms]',
            "antimicrobial peptides": '"Antimicrobial Cationic Peptides"[MeSH Terms]',
            "peptide": '"Peptides"[MeSH Terms]',
            "protein": '"Proteins"[MeSH Terms]',
            "crispr": '"CRISPR-Cas Systems"[MeSH Terms]',
            "clinical trial": '"Clinical Trial"[Publication Type]',
            "drug resistance": '"Drug Resistance, Microbial"[MeSH Terms]',
            "machine learning": '"Machine Learning"[MeSH Terms]',
            "deep learning": '"Deep Learning"[MeSH Terms]',
            "biomarker": '"Biomarkers"[MeSH Terms]',
        },
        arxiv_category_hints=["q-bio.BM", "q-bio.GN", "q-bio.QM"],
        preferred_sources=["PubMed", "bioRxiv", "Semantic Scholar", "arXiv"],
        evidence_types=[
            "Randomized clinical trials & clinical cohorts",
            "In vivo animal studies",
            "In vitro biochemical & cellular assays",
            "Systematic reviews & meta-analyses",
        ],
        extraction_columns=["Study Design", "Cohort / Assay System", "Quantitative Outcome (MIC/IC50/HR)", "Safety & Limitations"],
        gap_identification_rules=[
            "Check whether promising in vitro or in silico candidates lack in vivo stability, toxicity, or resistance profiling.",
            "Identify population, physiological-condition, or standardization gaps.",
        ],
        default_gap_rows=[
            "Experimental High-Throughput Screening & Assays",
            "Structure-Activity & Mechanistic Modeling",
            "Sequence-Based Predictive Classifiers",
            "De Novo Design & Lead Optimization",
        ],
        default_gap_cols=[
            "In Vitro Potency & Target Specificity",
            "In Vivo Efficacy, PK/PD & Stability",
            "Toxicity, Hemolysis & Off-Target Safety",
            "Clinical Translation & Resistance Barriers",
        ],
        requires_sample_size=True,
    ),
    "computer_science_ai": DomainProfile(
        domain_id="computer_science_ai",
        display_name="Computer Science & Artificial Intelligence",
        legacy_label="COMPUTATIONAL",
        implementation_tier="specialized",
        terminology_ontologies=["ACM CCS", "arXiv CS Taxonomy", "PapersWithCode Tasks"],
        concept_normalization_map={
            "llm": "large language model",
            "llms": "large language models",
            "plm": "protein language model",
            "gnn": "graph neural network",
            "ood": "out-of-distribution",
            "sota": "state-of-the-art",
            "rlhf": "reinforcement learning from human feedback",
            "rag": "retrieval-augmented generation",
        },
        validated_mesh_mappings={
            "machine learning": '"Machine Learning"[MeSH Terms]',
            "deep learning": '"Deep Learning"[MeSH Terms]',
            "neural network": '"Neural Networks, Computer"[MeSH Terms]',
            "artificial intelligence": '"Artificial Intelligence"[MeSH Terms]',
        },
        arxiv_category_hints=["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"],
        preferred_sources=["arXiv", "Semantic Scholar", "Hugging Face", "PubMed"],
        evidence_types=[
            "Empirical benchmark evaluations & ablation studies",
            "Theoretical convergence / complexity proofs",
            "Open-source pretrained models & reproducible datasets",
        ],
        extraction_columns=["Architecture / Algorithm", "Evaluation Benchmark & Splits", " Quantitative Metrics (AUC/Acc/F1)", "Compute & Generalization Limits"],
        gap_identification_rules=[
            "Identify absence of standardized OOD benchmarks, data leakage risks, or missing interpretability/uncertainty calibration.",
        ],
        default_gap_rows=[
            "Autoregressive & Masked Transformer Foundation Models",
            "Equivariant Graph & Geometric Deep Learning",
            "Diffusion & Flow-Matching Generative Models",
            "Hybrid Physics/Rule-Guided Neural Architectures",
        ],
        default_gap_cols=[
            "In-Distribution Standard Benchmark Accuracy",
            "Zero-Shot & Out-of-Distribution Generalization",
            "Uncertainty Calibration & Mechanistic Interpretability",
            "Compute Efficiency & Real-World Deployment",
        ],
        requires_sample_size=False,
    ),
    "interdisciplinary_biomedical_ai": DomainProfile(
        domain_id="interdisciplinary_biomedical_ai",
        display_name="Interdisciplinary Biomedical & Computational AI",
        legacy_label="BIOMEDICAL/AI",
        implementation_tier="specialized",
        terminology_ontologies=["MeSH", "arXiv q-bio/cs.LG", "UniProt", "Hugging Face Bio-ML"],
        concept_normalization_map={
            "amp": "antimicrobial peptide",
            "amps": "antimicrobial peptides",
            "plm": "protein language model",
            "plms": "protein language models",
            "gnn": "graph neural network",
            "admet": "absorption distribution metabolism excretion toxicity",
            "qsar": "quantitative structure-activity relationship",
            "ood": "out-of-distribution",
        },
        validated_mesh_mappings={
            "antimicrobial peptide": '"Antimicrobial Cationic Peptides"[MeSH Terms]',
            "antimicrobial peptides": '"Antimicrobial Cationic Peptides"[MeSH Terms]',
            "protein": '"Proteins"[MeSH Terms]',
            "peptide": '"Peptides"[MeSH Terms]',
            "machine learning": '"Machine Learning"[MeSH Terms]',
            "deep learning": '"Deep Learning"[MeSH Terms]',
        },
        arxiv_category_hints=["q-bio.BM", "q-bio.QM", "cs.LG", "cs.CE"],
        preferred_sources=["PubMed", "Semantic Scholar", "arXiv", "bioRxiv", "Hugging Face"],
        evidence_types=[
            "Computational model development with wet-lab experimental validation",
            "In silico benchmark evaluation across biological sequence/structure datasets",
            "Pretrained biological foundation models & curated bio-datasets",
        ],
        extraction_columns=["Computational Architecture", "Biological System / Dataset", "Predictive & Wet-Lab Metrics", "Translational & Generalization Limits"],
        gap_identification_rules=[
            "Identify disconnects between high in silico AUROC/accuracy and lack of wet-lab synthesis, toxicity counter-screening, or non-canonical modification support.",
        ],
        default_gap_rows=[
            "Pretrained Protein/Genomic Language Models (pLMs)",
            "3D Structure-Aware & Equivariant Graph Networks",
            "Conditional Generative & Diffusion Design Models",
            "Physics-Informed & Multi-Objective Active Learning",
        ],
        default_gap_cols=[
            "Retrospective Sequence/Activity Classification",
            "Homology-Aware Out-of-Distribution Generalization",
            "Multi-Parameter Toxicity, Stability & PK/PD Co-Optimization",
            "Closed-Loop Wet-Lab Experimental Verification",
        ],
        requires_sample_size=True,
    ),
    "chemistry": DomainProfile(
        domain_id="chemistry",
        display_name="Chemistry & Molecular Sciences",
        legacy_label="CHEMISTRY",
        implementation_tier="specialized",
        terminology_ontologies=["IUPAC", "ChEBI", "PubChem", "arXiv physics.chem-ph"],
        concept_normalization_map={
            "dft": "density functional theory",
            "md": "molecular dynamics",
            "sar": "structure-activity relationship",
            "qsar": "quantitative structure-activity relationship",
            "nmr": "nuclear magnetic resonance",
            "hplc": "high-performance liquid chromatography",
            "mof": "metal-organic framework",
        },
        validated_mesh_mappings={
            "catalysis": '"Catalysis"[MeSH Terms]',
            "molecular dynamics": '"Molecular Dynamics Simulation"[MeSH Terms]',
            "small molecule": '"Small Molecule Libraries"[MeSH Terms]',
        },
        arxiv_category_hints=["physics.chem-ph", "cond-mat.mtrl-sci", "cs.LG"],
        preferred_sources=["Semantic Scholar", "PubMed", "arXiv", "Hugging Face"],
        evidence_types=[
            "Synthetic route & reaction yield characterization",
            "Quantum chemical / DFT & molecular dynamics simulations",
            "Spectroscopic & crystallographic structural determination",
        ],
        extraction_columns=["Chemical Method / Theory Level", "Molecular System / Substrate Scope", "Yield / Selectivity / Energy Error", "Synthetic & Scalability Constraints"],
        gap_identification_rules=[
            "Identify unaddressed functional group tolerance, non-equilibrium reaction pathways, or computational scaling bottlenecks.",
        ],
        default_gap_rows=[
            "Density Functional Theory (DFT) & Ab Initio Methods",
            "Machine-Learned Interatomic Potentials (MLIPs)",
            "Generative 3D Molecular & Reaction Synthesis Models",
            "High-Throughput Experimental Catalysis & Screening",
        ],
        default_gap_cols=[
            "Equilibrium Geometry & Ground-State Energetics",
            "Transition-State & Reaction Barrier Accuracy",
            "Complex Solvent, Electrolyte & Interface Dynamics",
            "Experimental Synthesizability & Scale-Up",
        ],
        requires_sample_size=False,
    ),
    "physics": DomainProfile(
        domain_id="physics",
        display_name="Physics & Physical Sciences",
        legacy_label="PHYSICS",
        implementation_tier="specialized",
        terminology_ontologies=["APS Physics Subject Headings (PhySH)", "arXiv Physics Taxonomy"],
        concept_normalization_map={
            "qcd": "quantum chromodynamics",
            "qed": "quantum electrodynamics",
            "nisq": "noisy intermediate-scale quantum",
            "vqe": "variational quantum eigensolver",
            "pde": "partial differential equation",
            "mhd": "magnetohydrodynamics",
        },
        validated_mesh_mappings={},
        arxiv_category_hints=["quant-ph", "cond-mat", "physics.comp-ph", "hep-th", "astro-ph"],
        preferred_sources=["arXiv", "Semantic Scholar"],
        evidence_types=[
            "Analytical derivations & exact/perturbative solutions",
            "Numerical simulations (Monte Carlo, tensor networks, PDE solvers)",
            "Experimental physical measurements & detector observations",
        ],
        extraction_columns=["Theoretical / Experimental Regime", "Physical System & Parameters", "Scaling / Observables / Precision", "Approximation & Noise Limits"],
        gap_identification_rules=[
            "Identify regimes where perturbative approximations break down, strong-coupling limits, or experimental coherence barriers.",
        ],
        default_gap_rows=[
            "Analytical & Perturbative Field-Theoretic Methods",
            "Quantum Monte Carlo & Tensor Network Simulations",
            "Variational Quantum & Hybrid Classical-Quantum Algorithms",
            "Neural PDE Surrogates & Operator Learning",
        ],
        default_gap_cols=[
            "Weakly Coupled / Low-Dimensional Regimes",
            "Strongly Correlated & Non-Equilibrium Dynamics",
            "Finite-Temperature & Open-System Decoherence",
            "Experimental Realization & Metrological Verification",
        ],
        requires_sample_size=False,
    ),
    "mathematics": DomainProfile(
        domain_id="mathematics",
        display_name="Mathematics & Mathematical Foundations",
        legacy_label="MATHEMATICS",
        implementation_tier="specialized",
        terminology_ontologies=["MSC2020 (Mathematics Subject Classification)", "arXiv math Taxonomy"],
        concept_normalization_map={
            "pde": "partial differential equation",
            "ode": "ordinary differential equation",
            "sde": "stochastic differential equation",
            "rkhs": "reproducing kernel hilbert space",
        },
        validated_mesh_mappings={},
        arxiv_category_hints=["math.PR", "math.AP", "math.OC", "math.ST", "math.NA", "math.AT"],
        preferred_sources=["arXiv", "Semantic Scholar"],
        evidence_types=[
            "Formal mathematical proofs & theorems",
            "Counterexamples & sharpness bounds",
            "Numerical convergence verification & formalization (Lean/Coq)",
        ],
        extraction_columns=["Mathematical Framework", "Assumptions & Function Spaces", "Main Theorem / Convergence Rate", "Open Conjectures & Sharpness Limits"],
        gap_identification_rules=[
            "Identify unproved conjectures, non-sharp constants, restrictive convexity/smoothness assumptions, or high-dimensional extensions.",
        ],
        default_gap_rows=[
            "Exact Analytical & Variational Proof Techniques",
            "Probabilistic, Concentration & Martingale Methods",
            "Spectral, Operator & Harmonic Analysis",
            "Computer-Assisted Proof & Formal Verification",
        ],
        default_gap_cols=[
            "Smooth / Convex / Low-Dimensional Settings",
            "Non-Convex, Singular & Rough Regularity Regimes",
            "Minimax Optimal & Non-Asymptotic Sharp Bounds",
            "Algorithmic Tractability & Constructive Guarantees",
        ],
        requires_sample_size=False,
    ),
    "engineering_materials": DomainProfile(
        domain_id="engineering_materials",
        display_name="Engineering & Materials Science",
        legacy_label="PHYSICS",
        implementation_tier="generic_fallback",
        terminology_ontologies=["arXiv cond-mat.mtrl-sci", "Engineering Village Concepts"],
        concept_normalization_map={},
        validated_mesh_mappings={},
        arxiv_category_hints=["cond-mat.mtrl-sci", "physics.app-ph", "cs.RO", "eess.SY"],
        preferred_sources=["Semantic Scholar", "arXiv", "PubMed"],
        evidence_types=["Materials characterization", "Finite-element / continuum simulation", "Prototype testing"],
        extraction_columns=["Fabrication / Simulation Method", "Material / System", "Performance Metrics", "Durability & Scaling Limits"],
        gap_identification_rules=["Identify manufacturability, lifecycle stability, and multi-scale coupling gaps."],
        default_gap_rows=["Ab Initio & Multiscale Simulation", "Data-Driven Materials Screening", "Additive & Precision Fabrication", "In-Situ Characterization"],
        default_gap_cols=["Lab-Scale Property Optimization", "Long-Term Operational Stability", "High-Throughput Manufacturability", "Extreme-Environment Reliability"],
        requires_sample_size=False,
    ),
    "social_sciences": DomainProfile(
        domain_id="social_sciences",
        display_name="Social, Behavioral & Economic Sciences",
        legacy_label="COMPUTATIONAL",
        implementation_tier="generic_fallback",
        terminology_ontologies=["JEL Classification", "APA Thesaurus"],
        concept_normalization_map={"rct": "randomized controlled trial", "did": "difference-in-differences", "iv": "instrumental variable"},
        validated_mesh_mappings={},
        arxiv_category_hints=["econ.EM", "q-fin", "cs.CY", "physics.soc-ph"],
        preferred_sources=["Semantic Scholar", "PubMed", "arXiv"],
        evidence_types=["Quasi-experimental & econometric studies", "Survey & longitudinal panels", "Behavioral experiments"],
        extraction_columns=["Identification Strategy", "Population / Panel Dataset", "Estimated Effect", "External Validity & Confounders"],
        gap_identification_rules=["Identify unmeasured confounding, attrition bias, and cross-cultural replication gaps."],
        default_gap_rows=["Randomized Field Experiments", "Quasi-Experimental Econometrics", "Longitudinal Panel Surveys", "Computational Agent-Based Modeling"],
        default_gap_cols=["Short-Run Behavioral Effects", "Long-Run Equilibrium Outcomes", "Cross-Population Generalizability", "Policy Implementation Scalability"],
        requires_sample_size=True,
    ),
    "humanities_interdisciplinary": DomainProfile(
        domain_id="humanities_interdisciplinary",
        display_name="Humanities & Interdisciplinary Research",
        legacy_label="COMPUTATIONAL",
        implementation_tier="generic_fallback",
        terminology_ontologies=["OpenAlex Interdisciplinary Concepts"],
        concept_normalization_map={},
        validated_mesh_mappings={},
        arxiv_category_hints=["cs.DL", "cs.CL", "physics.hist-ph"],
        preferred_sources=["Semantic Scholar", "arXiv", "PubMed"],
        evidence_types=["Archival & textual analysis", "Comparative historical/corpus studies", "Conceptual synthesis"],
        extraction_columns=["Analytical Framework", "Corpus / Historical Scope", "Key Interpretation", "Source Coverage Limitations"],
        gap_identification_rules=["Identify under-digitized corpora, linguistic biases, and methodological silos."],
        default_gap_rows=["Close Textual & Archival Analysis", "Computational Corpus & NLP Analysis", "Comparative Cross-Cultural Frameworks", "Network & Provenance Reconstruction"],
        default_gap_cols=["Canonical Corpus Coverage", "Underrepresented Languages & Archives", "Temporal & Diachronic Shifts", "Interdisciplinary Validation"],
        requires_sample_size=False,
    ),
}


# Appraisal and synthesis rule wording comes from the source-cited registry (`frameworks.rules`), so prompts, the
# evidence table and reports all quote the same verified rules.
DOMAIN_PROFILES = {
    dom_id: replace(
        prof,
        methodological_appraisal_rules=rule_texts(dom_id, "appraisal"),
        synthesis_interpretation_rules=rule_texts(dom_id, "synthesis"),
    )
    for dom_id, prof in DOMAIN_PROFILES.items()
}


METHODOLOGICAL_PROFILES: Dict[MethodologicalProfileType, MethodologicalProfile] = {
    "observational": MethodologicalProfile(
        method_id="observational",
        display_name="Observational & Cohort Research",
        required_evidence_elements=["Cohort definition", "Exposure/outcome measure", "Adjusted association estimate"],
        sample_reporting_policy="required",
        priority_weight=6,
    ),
    "experimental": MethodologicalProfile(
        method_id="experimental",
        display_name="Experimental Laboratory Research",
        required_evidence_elements=["Experimental system", "Intervention/perturbation", "Quantitative assay readout"],
        sample_reporting_policy="optional",
        priority_weight=7,
    ),
    "clinical_intervention": MethodologicalProfile(
        method_id="clinical_intervention",
        display_name="Clinical Intervention & Trial Studies",
        required_evidence_elements=["Trial phase/design", "Patient population (N)", "Primary endpoint effect & safety"],
        sample_reporting_policy="required",
        priority_weight=9,
    ),
    "predictive_modeling": MethodologicalProfile(
        method_id="predictive_modeling",
        display_name="Prognostic & Predictive Modeling",
        required_evidence_elements=["Model family", "Training/validation dataset", "Discrimination & calibration metrics"],
        sample_reporting_policy="optional",
        priority_weight=8,
    ),
    "ml_benchmarking": MethodologicalProfile(
        method_id="ml_benchmarking",
        display_name="Machine Learning Benchmarking & Architecture Evaluation",
        required_evidence_elements=["Architecture", "Benchmark dataset", "Comparative metric vs baselines"],
        sample_reporting_policy="optional",
        priority_weight=8,
    ),
    "omics_high_throughput": MethodologicalProfile(
        method_id="omics_high_throughput",
        display_name="Omics & High-Throughput Screening",
        required_evidence_elements=["Assay/sequencing modality", "Library/sample scale", "Hit rate or enriched pathways"],
        sample_reporting_policy="optional",
        priority_weight=7,
    ),
    "theoretical_mathematical": MethodologicalProfile(
        method_id="theoretical_mathematical",
        display_name="Theoretical & Mathematical Research",
        required_evidence_elements=["Formal problem setting", "Key assumptions", "Theorem/bound/complexity result"],
        sample_reporting_policy="not_applicable",
        priority_weight=10,
    ),
    "simulation_based": MethodologicalProfile(
        method_id="simulation_based",
        display_name="Simulation & Computational Physics/Chemistry",
        required_evidence_elements=["Simulation method/solver", "System size/timescale", "Computed observable & error"],
        sample_reporting_policy="not_applicable",
        priority_weight=7,
    ),
    "qualitative": MethodologicalProfile(
        method_id="qualitative",
        display_name="Qualitative & Conceptual Inquiry",
        required_evidence_elements=["Analytical framework", "Source corpus", "Synthesized themes"],
        sample_reporting_policy="optional",
        priority_weight=5,
    ),
    "literature_conceptual_review": MethodologicalProfile(
        method_id="literature_conceptual_review",
        display_name="Systematic & Conceptual Literature Review",
        required_evidence_elements=["Review scope", "Included studies", "Synthesized consensus & gaps"],
        sample_reporting_policy="optional",
        priority_weight=6,
    ),
}


METHODOLOGICAL_PROFILES = {
    method_id: replace(
        prof,
        appraisal_criteria=method_rule_texts(method_id, "appraisal"),
        synthesis_qualification_rule=" ".join(method_rule_texts(method_id, "synthesis")),
    )
    for method_id, prof in METHODOLOGICAL_PROFILES.items()
}


def compose_profiles(
    primary_domain: DomainCategoryType,
    secondary_domains: List[DomainCategoryType],
    method_profiles: List[MethodologicalProfileType],
) -> Dict[str, Any]:
    """
    Composes the primary domain profile, any secondary domain profiles, and active
    methodological profiles using an explicit conflict-resolution hierarchy:
    1. Primary domain sets the core ontology, preferred sources, and default gap matrix dimensions.
    2. Secondary domains union additional concept normalizations, MeSH mappings, and arXiv categories
       without overwriting primary domain mappings.
    3. Methodological profiles ordered by `priority_weight` govern appraisal criteria,
       synthesis qualification rules, and sample-size reporting policy (e.g., if the question is
       purely `theoretical_mathematical` without clinical/observational profiles, sample_reporting_policy
       resolves to `not_applicable`).
    """
    prim = DOMAIN_PROFILES.get(primary_domain, DOMAIN_PROFILES["biomedical_clinical"])
    sec_objs = [DOMAIN_PROFILES[d] for d in secondary_domains if d in DOMAIN_PROFILES and d != primary_domain]
    meth_objs = sorted(
        [METHODOLOGICAL_PROFILES[m] for m in method_profiles if m in METHODOLOGICAL_PROFILES],
        key=lambda x: x.priority_weight,
        reverse=True,
    )
    if not meth_objs:
        meth_objs = [METHODOLOGICAL_PROFILES["experimental"]]

    # Merge concept normalization maps (primary wins on key collision)
    norm_map: Dict[str, str] = {}
    for s in reversed(sec_objs):
        norm_map.update(s.concept_normalization_map)
    norm_map.update(prim.concept_normalization_map)

    # Merge validated MeSH mappings
    mesh_map: Dict[str, str] = {}
    for s in reversed(sec_objs):
        mesh_map.update(s.validated_mesh_mappings)
    mesh_map.update(prim.validated_mesh_mappings)

    # Merge arXiv category hints
    arxiv_cats = list(dict.fromkeys(
        prim.arxiv_category_hints + [c for s in sec_objs for c in s.arxiv_category_hints]
    ))

    # Merge preferred sources preserving primary priority order
    pref_sources = list(dict.fromkeys(
        prim.preferred_sources + [src for s in sec_objs for src in s.preferred_sources]
    ))

    # Resolve sample reporting policy across active methodological profiles
    policies = [m.sample_reporting_policy for m in meth_objs]
    if "required" in policies and prim.requires_sample_size:
        resolved_sample_policy = "required"
    elif all(p == "not_applicable" for p in policies) or primary_domain in ("mathematics", "physics"):
        resolved_sample_policy = "not_applicable"
    else:
        resolved_sample_policy = "optional"

    appraisal_rules = list(dict.fromkeys(
        prim.methodological_appraisal_rules
        + [r for m in meth_objs for r in m.appraisal_criteria]
        + [r for s in sec_objs for r in s.methodological_appraisal_rules[:1]]
    ))

    synthesis_rules = list(dict.fromkeys(
        [m.synthesis_qualification_rule for m in meth_objs]
        + prim.synthesis_interpretation_rules
    ))

    return {
        "primary_domain": prim.domain_id,
        "primary_domain_display": prim.display_name,
        "legacy_domain_label": prim.legacy_label,
        "implementation_tier": prim.implementation_tier,
        "secondary_domains": [s.domain_id for s in sec_objs],
        "active_methodological_profiles": [m.method_id for m in meth_objs],
        "concept_normalization_map": norm_map,
        "validated_mesh_mappings": mesh_map,
        "arxiv_category_hints": arxiv_cats,
        "preferred_sources": pref_sources,
        "extraction_columns": prim.extraction_columns,
        "sample_reporting_policy": resolved_sample_policy,
        "appraisal_rules": appraisal_rules[:6],
        "synthesis_rules": synthesis_rules[:5],
        "gap_rules": prim.gap_identification_rules,
        "default_gap_rows": prim.default_gap_rows,
        "default_gap_cols": prim.default_gap_cols,
    }
