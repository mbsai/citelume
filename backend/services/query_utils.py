import re
from typing import List, Dict, Any, Tuple, Optional
from models import (
    ApiKeysConfig,
    AmbiguousTermNote,
    EntityRelationship,
    TechnicalConceptExtraction,
    RequiredConstraint,
    ResearchSubquestion,
    UniversalQuestionUnderstanding,
    FacetApiQueries,
    SearchFacet,
    UniversalSearchPlan,
    QueryDecompositionPlan,
    DomainCategoryType,
    ResearchIntentType,
    MethodologicalProfileType,
)
from services.domain_profiles import DOMAIN_PROFILES, compose_profiles
from services.llm_service import call_llm, extract_json_block
from services.question_parsing import (
    QUESTION_FRAMING_WORDS,
    concept_phrases,
    detect_constraints,
    detect_model_systems,
)

# Bounded in-memory cache for Stage A-C decomposition plans (Phase 11 cost/latency control)
_DECOMPOSITION_CACHE: Dict[str, QueryDecompositionPlan] = {}

CONVERSATIONAL_STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "want", "wanted", "wants",
    "would", "like", "looking", "look", "into", "find", "search", "explore",
    "investigate", "check", "know", "understand", "see", "show", "tell",
    "please", "can", "could", "should", "may", "might", "will", "shall",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "the", "a", "an", "and", "but",
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "now", "also", "yet", "still", "already", "even",
    "current", "currently", "existing", "available", "present", "recent",
    "new", "novel", "potential", "possible", "future", "next", "better",
    "best", "good", "bad", "main", "major", "key", "important", "different",
    "various", "several", "many", "much", "drawbacks", "drawback", "pros",
    "cons", "limitations", "limitation", "challenges", "challenge", "issues",
    "issue", "problems", "problem", "overcome", "overcoming", "solve",
    "solving", "address", "addressing", "improve", "improving", "approaches",
    "approach", "methods", "method", "techniques", "technique", "tools",
    "tool", "systems", "system", "models", "model", "huggingface", "hugging",
    "face", "papers", "paper", "literature", "review", "study", "studies",
    "research", "analysis", "overview", "summary", "using", "used", "use",
    "based", "via", "that", "this", "these", "those", "their", "them", "they",
    "it", "its", "ai", "does", "affect", "effect", "effects", "impact",
    "vitro", "vivo", "silico", "efficacy", "effective", "efficiency",
}

COMMON_TYPO_CORRECTIONS = {
    "anitmicrobial": "antimicrobial",
    "antimicrobal": "antimicrobial",
    "peptite": "peptide",
    "peptites": "peptides",
    "protien": "protein",
    "protiens": "proteins",
    "transfomer": "transformer",
    "transfomers": "transformers",
    "machien": "machine",
    "learing": "learning",
    "screnning": "screening",
    "bioinformatcs": "bioinformatics",
    "alzheimer": "alzheimers",
    "schrodinger": "schroedinger",
}

# Domain keyword signatures for Stage A Universal Question Understanding
# Note: Ambiguous acronyms (amp, dft, gan, iv) are scored via AMBIGUOUS_ACRONYM_REGISTRY
DOMAIN_KEYWORD_SIGNATURES: Dict[DomainCategoryType, List[str]] = {
    "biomedical_clinical": [
        "peptide", "peptides", "protein", "proteins", "antimicrobial",
        "genome", "genomic", "dna", "rna", "crispr", "clinical", "drug", "therapeutic",
        "cell", "antibody", "enzyme", "disease", "patient", "patients", "receptor",
        "toxicity", "hemolysis", "vaccine", "oncology", "tumor", "cancer", "cohort",
        "biomarker", "mortality", "sepsis", "pharmacokinetic", "pathology", "microbiome",
    ],
    "computer_science_ai": [
        "transformer", "transformers", "neural", "algorithm", "gpu", "distributed",
        "compiler", "diffusion", "reinforcement", "attention", "embedding", "llm",
        "language model", "pretrained", "fine-tuning", "zero-shot", "benchmark",
        "graph neural", "gnn", "convolutional", "backpropagation", "quantization",
        "autoregressive", "tokenization", "vision-language", "multimodal", "pytorch",
    ],
    "chemistry": [
        "smiles", "iupac", "molecule", "molecules", "molecular", "catalyst", "catalysis",
        "synthesis", "reaction", "polymer", "ligand", "docking", "conformational",
        "electrolyte", "solvent", "stereoselective", "enantiomeric", "organometallic",
        "chirality", "spectroscopy", "calorimetry", "retrosynthesis", "kohn-sham",
    ],
    "physics": [
        "quantum", "photon", "relativity", "astrophysics", "plasma", "superconductor",
        "superconducting", "lattice", "thermodynamics", "hamiltonian", "spin", "fluid",
        "turbulence", "navier-stokes", "decoherence", "entanglement", "qubit", "boson",
        "fermion", "cosmology", "magnetohydrodynamics", "interferometry",
    ],
    "mathematics": [
        "theorem", "proof", "topology", "topological", "manifold", "algebraic",
        "stochastic", "differential equation", "pde", "ode", "convex", "eigenvalue",
        "conjecture", "riemann", "hilbert", "banach", "martingale", "ergodic",
        "cohomology", "isomorphism", "minimax", "non-asymptotic", "lebesgue",
    ],
    "engineering_materials": [
        "alloy", "perovskite", "battery", "anode", "cathode", "tensile", "microstructure",
        "semiconductor", "robotics", "actuator", "control system", "metamaterial",
        "piezoelectric", "thermal conductivity", "additive manufacturing",
    ],
    "social_sciences": [
        "econometric", "socioeconomic", "policy", "wage", "inflation", "voting",
        "education", "inequality", "survey", "demographic", "behavioral", "governance",
    ],
    "humanities_interdisciplinary": [
        "historiography", "epigraphy", "archival", "literary", "philology", "philosophy",
        "ethics", "linguistics", "archaeological",
    ],
}

# Known ambiguous acronyms across disciplines (Phase 2.1B & Phase 10 edge cases)
AMBIGUOUS_ACRONYM_REGISTRY: Dict[str, List[Tuple[str, DomainCategoryType, List[str]]]] = {
    "amp": [
        ("antimicrobial peptide", "biomedical_clinical", ["peptide", "bacteria", "screening", "resistance", "protein", "toxicity"]),
        ("adenosine monophosphate", "biomedical_clinical", ["kinase", "ampk", "atp", "cyclic", "camp", "metabolism"]),
        ("automatic mixed precision", "computer_science_ai", ["fp16", "bf16", "gpu", "pytorch", "training", "cuda"]),
    ],
    "dft": [
        ("density functional theory", "chemistry", ["quantum", "molecule", "electronic", "b3lyp", "kohn-sham", "functional", "energy"]),
        ("discrete fourier transform", "mathematics", ["signal", "fft", "frequency", "spectral", "harmonic", "polynomial"]),
    ],
    "gan": [
        ("generative adversarial network", "computer_science_ai", ["generator", "discriminator", "deep", "neural", "image", "synthetic"]),
        ("gallium nitride", "engineering_materials", ["semiconductor", "transistor", "hemt", "bandgap", "led", "epitaxy"]),
    ],
    "iv": [
        ("instrumental variable", "social_sciences", ["causal", "endogeneity", "regression", "econometric", "2sls"]),
        ("intravenous", "biomedical_clinical", ["dose", "infusion", "patient", "pharmacokinetic", "injection", "clinical"]),
    ],
}


def normalize_and_correct_query(query: str) -> str:
    cleaned = re.sub(r"[^\w\s\-/<>]", " ", query or "")
    words = cleaned.split()
    fixed_words = []
    for w in words:
        lw = w.lower()
        if lw in COMMON_TYPO_CORRECTIONS:
            fixed_words.append(COMMON_TYPO_CORRECTIONS[lw])
        else:
            fixed_words.append(w)
    return " ".join(fixed_words)


def detect_domains_and_profiles(
    query: str,
    domain_override: Optional[DomainCategoryType] = None,
) -> Tuple[DomainCategoryType, List[DomainCategoryType], List[MethodologicalProfileType], List[AmbiguousTermNote]]:
    """
    Stage A Domain & Methodological Profile Detector.
    Supports single-domain and interdisciplinary queries without defaulting non-biomedical
    queries to biomedical.
    """
    q_norm = normalize_and_correct_query(query)
    q_low = q_norm.lower()

    scores: Dict[DomainCategoryType, int] = {dom: 0 for dom in DOMAIN_KEYWORD_SIGNATURES}
    for dom, kw_list in DOMAIN_KEYWORD_SIGNATURES.items():
        for kw in kw_list:
            if re.search(rf"\b{re.escape(kw)}\b", q_low):
                scores[dom] += 2 if " " in kw else 1

    # Disambiguate ambiguous acronyms using context keywords
    ambiguous_notes: List[AmbiguousTermNote] = []
    for acr, meanings in AMBIGUOUS_ACRONYM_REGISTRY.items():
        if re.search(rf"\b{re.escape(acr)}s?\b", q_low):
            best_meaning = meanings[0][0]
            best_dom = meanings[0][1]
            best_hits = -1
            for meaning_str, dom_cat, ctx_words in meanings:
                hits = sum(1 for cw in ctx_words if cw in q_low)
                if hits > best_hits:
                    best_hits = hits
                    best_meaning = meaning_str
                    best_dom = dom_cat
            scores[best_dom] = scores.get(best_dom, 0) + 2
            ambiguous_notes.append(
                AmbiguousTermNote(
                    term=acr.upper(),
                    candidate_meanings=[m[0] for m in meanings],
                    resolved_meaning=best_meaning,
                )
            )

    ranked_domains = [dom for dom, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True) if sc > 0]

    # Check if both biomedical_clinical and computer_science_ai are active -> interdisciplinary_biomedical_ai
    if "biomedical_clinical" in ranked_domains and "computer_science_ai" in ranked_domains:
        primary_domain: DomainCategoryType = "interdisciplinary_biomedical_ai"
        secondary_domains: List[DomainCategoryType] = [
            "biomedical_clinical",
            "computer_science_ai",
        ] + [d for d in ranked_domains if d not in ("biomedical_clinical", "computer_science_ai")][:1]
    elif ranked_domains:
        primary_domain = ranked_domains[0]
        secondary_domains = ranked_domains[1:3]
    else:
        # If no specific domain keywords matched, inspect mathematical/computational vs general scientific cues
        if any(w in q_low for w in ("equation", "function", "space", "bound", "convergence", "proof")):
            primary_domain = "mathematics"
        elif any(w in q_low for w in ("model", "prediction", "classification", "dataset", "network")):
            primary_domain = "computer_science_ai"
        else:
            primary_domain = "biomedical_clinical"
        secondary_domains = []

    if domain_override:
        # The user chose the field explicitly; keep any other detected domains as secondary context
        secondary_domains = [d for d in [primary_domain] + secondary_domains if d != domain_override][:2]
        primary_domain = domain_override

    # Detect applicable methodological profiles
    method_profiles: List[MethodologicalProfileType] = []
    if re.search(r"\b(theorem|proof|conjecture|lemma|convex|asymptotic|minimax|pde|ode|hilbert|manifold)\b", q_low):
        method_profiles.append("theoretical_mathematical")
    if re.search(r"\b(simulation|molecular dynamics|dft|monte carlo|finite element|navier-stokes|lattice)\b", q_low):
        method_profiles.append("simulation_based")
    if re.search(r"\b(clinical trial|randomized|rct|placebo|phase ii|phase iii|intervention|therapy|treatment)\b", q_low):
        method_profiles.append("clinical_intervention")
    if re.search(r"\b(cohort|observational|longitudinal|case-control|epidemiolog|risk factor|survival|hazard)\b", q_low):
        method_profiles.append("observational")
    if re.search(r"\b(predict|prognostic|classifier|classification|screening|risk score|auroc|auc)\b", q_low):
        method_profiles.append("predictive_modeling")
    if re.search(r"\b(benchmark|transformer|deep learning|neural|language model|llm|diffusion|gnn|ablation|huggingface)\b", q_low):
        method_profiles.append("ml_benchmarking")
    if re.search(r"\b(omics|rna-seq|single-cell|high-throughput|genome-wide|gwas|proteomics|screen)\b", q_low):
        method_profiles.append("omics_high_throughput")
    if re.search(r"\b(in vitro|in vivo|assay|wet-lab|synthesis|characterization|spectroscopy)\b", q_low):
        method_profiles.append("experimental")
    if re.search(r"\b(review|overview|landscape|taxonomy|survey|meta-analysis)\b", q_low):
        method_profiles.append("literature_conceptual_review")

    if not method_profiles:
        if primary_domain == "mathematics":
            method_profiles.append("theoretical_mathematical")
        elif primary_domain in ("physics", "chemistry"):
            method_profiles.append("simulation_based")
        elif primary_domain in ("computer_science_ai", "interdisciplinary_biomedical_ai"):
            method_profiles.append("ml_benchmarking")
        else:
            method_profiles.append("experimental")

    return primary_domain, secondary_domains, method_profiles, ambiguous_notes


def detect_research_intents(query: str) -> Tuple[ResearchIntentType, List[ResearchIntentType]]:
    """
    Stage A Research Intent Classifier.
    Identifies primary and genuine secondary research intents without padding irrelevant intents.
    """
    q_low = (query or "").lower()
    intents: List[ResearchIntentType] = []

    if re.search(r"\b(gap|gaps|limitation|limitations|drawback|drawbacks|unresolved|missing|open problem|bottleneck|novel approach)\b", q_low):
        intents.append("gap_discovery")
    if re.search(r"\b(compare|comparison|versus|vs\b|outperform|trade-off|benchmark|better than)\b", q_low):
        intents.append("comparative")
    if re.search(r"\b(predict|prediction|forecast|screening|classify|classification|prognos|early detection)\b", q_low):
        intents.append("predictive")
    if re.search(r"\b(cause|causal|effect of|efficacy|intervention|treatment|reduce|improve outcome|trial)\b", q_low):
        intents.append("causal_intervention")
    if re.search(r"\b(mechanism|pathway|how does|why does|structural basis|kinetics|dynamics)\b", q_low):
        intents.append("explanatory_mechanistic")
    if re.search(r"\b(theorem|proof|convergence|bound|existence|uniqueness|regularity|conjecture)\b", q_low):
        intents.append("theoretical")
    if re.search(r"\b(method|algorithm|architecture|pipeline|tool|framework|design)\b", q_low):
        intents.append("methodological")
    if re.search(r"\b(association|associated|correlat|risk factor|prevalence)\b", q_low):
        intents.append("associational")
    if re.search(r"\b(map|landscape|survey|review|overview|state of the art)\b", q_low):
        intents.append("exploratory_mapping")

    if not intents:
        intents.append("descriptive")

    return intents[0], intents[1:4]


def detect_scientific_domain(query: str) -> str:
    """
    Backward-compatible helper returning the legacy domain label ('BIOMEDICAL/AI',
    'CHEMISTRY', 'PHYSICS', 'MATHEMATICS', or 'COMPUTATIONAL').
    """
    prim, _, _, _ = detect_domains_and_profiles(query)
    prof = DOMAIN_PROFILES.get(prim, DOMAIN_PROFILES["biomedical_clinical"])
    return prof.legacy_label


def extract_core_scientific_terms(query: str, max_terms: int = 6) -> List[str]:
    """
    Strips conversational filler words, resolves ambiguous acronyms where appropriate,
    corrects common typos, and returns the core domain/scientific terms.
    """
    corrected = normalize_and_correct_query(query)
    raw_tokens = re.findall(r"[A-Za-z0-9-]+", corrected)
    core_terms: List[str] = []
    seen = set()

    for tok in raw_tokens:
        low = tok.lower().strip("-")
        if len(low) < 2:
            continue
        if low in CONVERSATIONAL_STOP_WORDS or low in QUESTION_FRAMING_WORDS:
            continue
        if low not in seen:
            seen.add(low)
            core_terms.append(tok if tok.isupper() and len(tok) <= 6 else low)

    if not core_terms:
        fallback = [w.lower() for w in raw_tokens if len(w) > 3 and w.lower() not in {"want", "look", "into"}]
        return fallback[:max_terms] or ["scientific", "analysis"]

    return core_terms[:max_terms]


def build_concise_search_phrase(query: str, max_terms: int = 5) -> str:
    terms = extract_core_scientific_terms(query, max_terms=max_terms)
    return " ".join(terms)


# ============================================================================
# STAGE A: Universal Question Understanding Engine
# ============================================================================

def understand_research_question(
    query: str, domain_override: Optional[DomainCategoryType] = None
) -> UniversalQuestionUnderstanding:
    """
    Stage A of Phase 2: Produces a domain-adaptive, source-independent
    UniversalQuestionUnderstanding object with intents, domains, methodological profiles,
    normalized technical concepts, and complexity-appropriate subquestions.
    """
    norm_q = normalize_and_correct_query(query)
    prim_dom, sec_doms, meth_profiles, amb_notes = detect_domains_and_profiles(norm_q, domain_override)
    prim_intent, sec_intents = detect_research_intents(norm_q)
    composed = compose_profiles(prim_dom, sec_doms, meth_profiles)

    raw_entities = extract_core_scientific_terms(norm_q, max_terms=8)
    norm_map = composed["concept_normalization_map"]

    normalized_terms: List[str] = []
    synonyms_map: Dict[str, List[str]] = {}
    for ent in raw_entities:
        low_e = ent.lower()
        # Check if ambiguous acronym was resolved
        resolved_override = next((a.resolved_meaning for a in amb_notes if a.term.lower() == low_e.rstrip("s")), None)
        if resolved_override:
            normalized_terms.append(resolved_override)
            synonyms_map[ent] = [resolved_override]
        elif low_e in norm_map:
            normalized_terms.append(norm_map[low_e])
            synonyms_map[ent] = [norm_map[low_e]]
        else:
            normalized_terms.append(ent)

    # Deduplicate normalized_terms preserving order
    normalized_terms = list(dict.fromkeys(normalized_terms))

    # Extract constraints and exclusions if present in query
    constraints: List[str] = []
    exclusions: List[str] = []
    if re.search(r"\b(in vivo|clinical|human|pediatric|adult)\b", norm_q, re.I):
        m = re.search(r"\b(in vivo|clinical|human|pediatric|adult)\b", norm_q, re.I)
        if m:
            constraints.append(f"Context constraint: {m.group(0)}")
    for neg_match in re.finditer(r"\b(?:without|excluding|except)\s+([a-zA-Z0-9-]+)", norm_q, re.I):
        exclusions.append(neg_match.group(1).lower())
    required = [
        RequiredConstraint(kind=c.kind, name=c.name, spellings=list(c.spellings)) for c in detect_constraints(norm_q)
    ]
    constraints.extend(f"{c.kind.capitalize()} constraint: {c.name}" for c in required)
    phrases = concept_phrases(norm_q, CONVERSATIONAL_STOP_WORDS, max_phrases=6)
    model_systems = [
        RequiredConstraint(kind="model_system", name=m.name, spellings=list(m.spellings))
        for m in detect_model_systems(norm_q)
    ]

    # Determine complexity: narrow (1-4 core terms, 1 intent) vs moderate vs broad_interdisciplinary
    word_count = len(norm_q.split())
    if len(raw_entities) <= 4 and len(sec_intents) == 0 and word_count <= 7:
        complexity = "narrow"
    elif len(sec_doms) >= 1 or len(sec_intents) >= 2 or word_count >= 18:
        complexity = "broad_interdisciplinary"
    else:
        complexity = "moderate"

    primary_subject = " ".join(normalized_terms[:2]) if len(normalized_terms) >= 2 else (normalized_terms[0] if normalized_terms else norm_q)
    secondary_focus = " ".join(normalized_terms[2:4]) if len(normalized_terms) > 2 else composed["primary_domain_display"]

    # Build entity relationships
    relationships: List[EntityRelationship] = []
    if len(normalized_terms) >= 2:
        relationships.append(
            EntityRelationship(
                source_entity=normalized_terms[0],
                relation="investigated_with",
                target_entity=normalized_terms[1],
            )
        )
    if len(normalized_terms) >= 3:
        relationships.append(
            EntityRelationship(
                source_entity=" ".join(normalized_terms[:2]),
                relation="evaluated_for",
                target_entity=normalized_terms[2],
            )
        )

    # Generate complexity-adaptive subquestions (1-2 for narrow, 2-3 for moderate, 3-4 for broad)
    subquestions: List[ResearchSubquestion] = [
        ResearchSubquestion(
            id="SQ1",
            question_text=f"What is the primary direct evidence regarding {primary_subject} ({ secondary_focus })?",
            relationship_to_main="Core direct evidence addressing the principal research question",
            category="direct_evidence",
            priority="high",
            required_evidence=", ".join(composed["extraction_columns"][:2]),
            depends_on=[],
        )
    ]

    if complexity in ("moderate", "broad_interdisciplinary") or "comparative" in [prim_intent] + sec_intents or "methodological" in [prim_intent] + sec_intents:
        subquestions.append(
            ResearchSubquestion(
                id="SQ2",
                question_text=f"What foundational frameworks, benchmarks, or comparative baselines govern {primary_subject}?",
                relationship_to_main="Establishes methodological baselines and theoretical/experimental context",
                category="comparative_evidence",
                priority="high",
                required_evidence="Benchmark datasets, theoretical assumptions, or reference standards",
                depends_on=["SQ1"],
            )
        )

    if complexity == "broad_interdisciplinary" or "gap_discovery" in [prim_intent] + sec_intents:
        subquestions.append(
            ResearchSubquestion(
                id="SQ3",
                question_text=f"What methodological limitations, conflicting findings, or unvalidated regimes persist for {primary_subject}?",
                relationship_to_main="Evaluates robustness, failure modes, and verified knowledge gaps",
                category="validation_robustness",
                priority="medium",
                required_evidence="Negative results, OOD generalization tests, or unaddressed constraints",
                depends_on=["SQ1"],
            )
        )

    if complexity == "broad_interdisciplinary" and len(normalized_terms) >= 3:
        subquestions.append(
            ResearchSubquestion(
                id="SQ4",
                question_text=f"What emerging methods or transferable architectures address current bottlenecks in {primary_subject}?",
                relationship_to_main="Identifies recent innovations and open-source implementations",
                category="contextual_evidence",
                priority="exploratory",
                required_evidence="Recent preprints, novel formulations, or open model artifacts",
                depends_on=["SQ2", "SQ3"] if len(subquestions) >= 3 else ["SQ1"],
            )
        )

    concepts = TechnicalConceptExtraction(
        concept_phrases=phrases,
        required_constraints=required,
        model_systems=model_systems,
        main_entities=raw_entities[:6],
        normalized_terms=normalized_terms[:6],
        synonyms_and_aliases=synonyms_map,
        entity_relationships=relationships,
        outcomes_or_targets=normalized_terms[2:5] if len(normalized_terms) > 2 else normalized_terms[:2],
        experimental_or_theoretical_context=composed["primary_domain_display"],
        constraints=constraints,
        exclusion_criteria=exclusions,
    )

    return UniversalQuestionUnderstanding(
        domain_selected_by_user=bool(domain_override),
        raw_query=query,
        normalized_query=norm_q,
        question_complexity=complexity,
        primary_intent=prim_intent,
        secondary_intents=sec_intents,
        primary_domain=prim_dom,
        secondary_domains=sec_doms,
        active_methodological_profiles=meth_profiles,
        ambiguous_terms=amb_notes,
        concepts=concepts,
        subquestions=subquestions,
    )


# ============================================================================
# STAGE C: Validated Source-Specific Query Compilers (Phase 5)
# ============================================================================

def compile_pubmed_query(
    core_terms: List[str],
    modifier_terms: List[str],
    composed_profile: Dict[str, Any],
    exclusions: Optional[List[str]] = None,
    core_override: Optional[str] = None,
) -> str:
    """
    Compiles a syntax-validated PubMed query combining validated MeSH headings
    (when present in the domain profile) with [Title/Abstract] terms and balanced parentheses.
    Never invents unverified MeSH headings.
    """
    mesh_map: Dict[str, str] = composed_profile.get("validated_mesh_mappings", {})
    primary_phrase = " ".join(core_terms[:2]) if len(core_terms) >= 2 else (core_terms[0] if core_terms else "science")

    core_clauses: List[str] = [f'"{primary_phrase}"[Title/Abstract]']
    if primary_phrase.lower() in mesh_map:
        core_clauses.append(mesh_map[primary_phrase.lower()])
    for t in core_terms[:2]:
        if t.lower() in mesh_map and mesh_map[t.lower()] not in core_clauses:
            core_clauses.append(mesh_map[t.lower()])

    if len(core_terms) >= 2:
        and_pair = f'({core_terms[0]}[Title/Abstract] AND {core_terms[1]}[Title/Abstract])'
        if and_pair not in core_clauses:
            core_clauses.append(and_pair)

    core_block = core_override or "(" + " OR ".join(core_clauses) + ")"

    if modifier_terms:
        mod_clauses = []
        for m in modifier_terms[:4]:
            if m.startswith('"') and "[" in m:
                mod_clauses.append(m)
            elif m.lower() in mesh_map:
                mod_clauses.append(f'({mesh_map[m.lower()]} OR "{m}"[Title/Abstract])')
            else:
                mod_clauses.append(f'"{m}"[Title/Abstract]')
        query_str = f"{core_block} AND ({' OR '.join(mod_clauses)})"
    else:
        query_str = core_block

    if exclusions:
        excl_clauses = [f'"{ex}"[Title/Abstract]' for ex in exclusions[:2] if ex]
        if excl_clauses:
            query_str = f"{query_str} NOT ({' OR '.join(excl_clauses)})"

    return query_str


def _tiab(term: str) -> str:
    return f'"{term}"[Title/Abstract]' if " " in term else f"{term}[Title/Abstract]"


def build_required_pubmed_core(phrases: List[str], constraints: List[RequiredConstraint]) -> Optional[str]:
    """
    Required part of every PubMed query: the main concept phrases (each as a phrase or all of its words) AND every
    place/population constraint (any of its spellings). Returns None when the question has no phrases.
    """
    if not phrases:
        return None
    blocks = []
    for ph in phrases[:2]:
        words = ph.split()
        alts = [_tiab(ph)] + ([f"({' AND '.join(_tiab(w) for w in words)})"] if len(words) > 1 else [])
        blocks.append("(" + " OR ".join(alts) + ")")
    for c in constraints:
        blocks.append("(" + " OR ".join(_tiab(sp) for sp in c.spellings[:4]) + ")")
    return " AND ".join(blocks)


def build_required_arxiv_core(phrases: List[str], constraints: List[RequiredConstraint]) -> Optional[str]:
    if not phrases:
        return None
    blocks = []
    for ph in phrases[:2]:
        words = [re.sub(r"[^a-zA-Z0-9_-]", "", w) for w in ph.split()]
        words = [w for w in words if w]
        if not words:
            continue
        blocks.append(f'all:"{" ".join(words)}"' if len(words) > 1 else f"(ti:{words[0]} OR abs:{words[0]})")
    for c in constraints:
        sps = [re.sub(r"[^a-zA-Z0-9 _-]", "", sp) for sp in c.spellings[:3]]
        blocks.append("(" + " OR ".join(f'all:"{sp}"' if " " in sp else f"all:{sp}" for sp in sps if sp) + ")")
    return " AND ".join(blocks) if blocks else None


def apply_required_constraints(queries: FacetApiQueries, constraints: List[RequiredConstraint]) -> FacetApiQueries:
    """Adds any missing place/population constraint to every source query (used on LLM-written queries)."""
    if not constraints:
        return queries
    q = queries.model_copy(deep=True)
    for c in constraints:
        def has(text: str) -> bool:
            low = (text or "").lower()
            return any(sp in low for sp in c.spellings)
        if q.pubmed and not has(q.pubmed):
            q.pubmed = f"({q.pubmed}) AND (" + " OR ".join(_tiab(sp) for sp in c.spellings[:4]) + ")"
        if q.arxiv and not has(q.arxiv):
            q.arxiv = f"({q.arxiv}) AND (" + " OR ".join(f"all:{sp}" for sp in c.spellings[:3] if " " not in sp) + ")"
        if q.semantic_scholar and not has(q.semantic_scholar):
            q.semantic_scholar = f"{q.semantic_scholar} {c.name}"
        q.biorxiv_medrxiv = [b if has(b) else f"{b} {c.name}" for b in (q.biorxiv_medrxiv or [])]
    return q


def compile_semantic_scholar_query(
    core_terms: List[str],
    modifier_terms: List[str],
) -> str:
    """
    Compiles a clean, dense natural-language phrase (no Boolean operators)
    for Semantic Scholar's relevance search endpoint.
    """
    combined = list(dict.fromkeys(core_terms[:4] + modifier_terms[:3]))
    cleaned = [re.sub(r'["\[\]()\']', "", t).strip() for t in combined if t.strip()]
    return " ".join(cleaned[:7])


def compile_arxiv_query(
    core_terms: List[str],
    modifier_terms: List[str],
    composed_profile: Dict[str, Any],
    exclusions: Optional[List[str]] = None,
    core_override: Optional[str] = None,
) -> str:
    """
    Compiles a valid arXiv API query using documented field prefixes (ti:, abs:, all:, cat:)
    and uppercase Boolean connectors (AND, OR, ANDNOT).
    """
    safe_core = [re.sub(r"[^a-zA-Z0-9_-]", "", t) for t in core_terms[:3] if re.sub(r"[^a-zA-Z0-9_-]", "", t)]
    if not safe_core:
        safe_core = ["model"]

    core_parts = [f"(ti:{t} OR abs:{t})" for t in safe_core[:2]]
    core_expr = core_override or " AND ".join(core_parts)

    safe_mods = []
    for m in modifier_terms[:4]:
        # Take single principal token from modifier for clean arXiv syntax
        toks = [re.sub(r"[^a-zA-Z0-9_-]", "", w) for w in m.split() if len(w) >= 3]
        if toks:
            safe_mods.append(f"abs:{toks[0]}")

    if safe_mods:
        query_str = f"{core_expr} AND ({' OR '.join(safe_mods)})"
    else:
        query_str = core_expr

    if exclusions:
        safe_ex = [re.sub(r"[^a-zA-Z0-9_-]", "", x) for x in exclusions[:2] if re.sub(r"[^a-zA-Z0-9_-]", "", x)]
        if safe_ex:
            query_str = f"{query_str} ANDNOT ({' OR '.join(f'abs:{x}' for x in safe_ex)})"

    return query_str


def compile_biorxiv_queries(
    core_terms: List[str],
    modifier_terms: List[str],
) -> List[str]:
    """
    Compiles 2 concise, high-signal multi-word phrases for preprint discovery via indexed search.
    """
    primary = " ".join(core_terms[:2]) if len(core_terms) >= 2 else (core_terms[0] if core_terms else "biology")
    phrases = []
    if modifier_terms:
        phrases.append(f"{primary} {modifier_terms[0]}".strip())
        if len(modifier_terms) > 1:
            phrases.append(f"{primary} {modifier_terms[1]}".strip())
    if len(core_terms) >= 3 and len(phrases) < 2:
        phrases.append(" ".join(core_terms[:3]))
    if not phrases:
        phrases.append(primary)
    return list(dict.fromkeys(phrases))[:2]


def compile_huggingface_query(
    core_terms: List[str],
    primary_domain: DomainCategoryType,
    facet_code: str,
) -> str:
    """
    Compiles a resource-specific Hugging Face Hub search string targeting models, datasets,
    and architectures without paper-specific filler vocabulary.
    """
    paper_words = {"review", "systematic", "meta-analysis", "study", "clinical", "trial", "cohort", "limitations", "unresolved"}
    clean_tokens = [t for t in core_terms if t.lower() not in paper_words][:3]
    if not clean_tokens:
        clean_tokens = core_terms[:2] or ["transformer"]

    if facet_code == "F4_recent_developments" and primary_domain in ("biomedical_clinical", "interdisciplinary_biomedical_ai"):
        if any(w in " ".join(clean_tokens).lower() for w in ("peptide", "protein", "antimicrobial", "sequence")):
            return "esm2 protein"
    return " ".join(clean_tokens[:2])


# ============================================================================
# STAGE B: Universal Search Strategy Planner (Phase 4)
# ============================================================================

def plan_and_compile_universal_search(
    understanding: UniversalQuestionUnderstanding,
    mode: str = "standard",
) -> Tuple[UniversalSearchPlan, List[SearchFacet], Dict[str, Any]]:
    """
    Stage B & C: Builds a source-independent UniversalSearchPlan across facets F1-F6
    connected to the question's subquestions, then compiles validated API queries.
    Avoids redundant searches and adapts the number of facets to question complexity and mode.
    """
    composed = compose_profiles(
        understanding.primary_domain,
        understanding.secondary_domains,
        understanding.active_methodological_profiles,
    )

    core_terms = understanding.concepts.normalized_terms or understanding.concepts.main_entities or ["research"]
    # Keep single-word tokens for arXiv/Boolean safety alongside multi-word normalized phrases
    atomic_terms = extract_core_scientific_terms(understanding.normalized_query, max_terms=6)
    exclusions = understanding.concepts.exclusion_criteria
    # Concept phrases and required constraints keep multi-word concepts intact and make places/populations mandatory
    phrases = understanding.concepts.concept_phrases
    constraints = understanding.concepts.required_constraints
    pubmed_core = build_required_pubmed_core(phrases, constraints)
    arxiv_core = build_required_arxiv_core(phrases, constraints)
    if phrases:
        constraint_names = [c.name for c in constraints]
        core_terms = phrases[:2] + constraint_names
        bio_terms = [" ".join(phrases[:1] + constraint_names)] + phrases[1:2]
    else:
        bio_terms = atomic_terms

    sq_ids = [sq.id for sq in understanding.subquestions] or ["SQ1"]
    all_facets: List[SearchFacet] = []

    # F1: Direct Evidence (Always active — highest priority)
    f1_mods = (phrases[2:5] if phrases else atomic_terms[2:5] if len(atomic_terms) > 2 else [])
    all_facets.append(
        SearchFacet(
            facet_code="F1_direct_evidence",
            facet_name="F1: Direct Evidence & Core Subquestions",
            facet_intent="Retrieve primary empirical, theoretical, or computational studies directly answering the core question.",
            linked_subquestion_ids=[sq_ids[0]],
            expansion_reason="Primary direct evidence retrieval",
            priority_rank=1,
            queries=FacetApiQueries(
                pubmed=compile_pubmed_query(atomic_terms, f1_mods, composed, exclusions, core_override=pubmed_core),
                semantic_scholar=compile_semantic_scholar_query(core_terms[:3], f1_mods[:2]),
                arxiv=compile_arxiv_query(atomic_terms, f1_mods, composed, exclusions, core_override=arxiv_core),
                biorxiv_medrxiv=compile_biorxiv_queries(bio_terms, f1_mods[:2]),
                hugging_face=compile_huggingface_query(atomic_terms, understanding.primary_domain, "F1_direct_evidence"),
            ),
        )
    )

    # F2: Foundational, Methodological & Benchmark Literature (Does NOT search for the word "seminal")
    if understanding.primary_domain == "mathematics":
        f2_mods = ["theorem", "proof", "convergence", "bound"]
    elif understanding.primary_domain == "physics":
        f2_mods = ["theory", "simulation", "scaling", "quantum"]
    elif understanding.primary_domain == "chemistry":
        f2_mods = ["mechanism", "catalysis", "density functional", "benchmark"]
    else:
        f2_mods = ["review", "benchmark", "comparative", "validation"]

    all_facets.append(
        SearchFacet(
            facet_code="F2_foundational_contextual",
            facet_name="F2: Foundational Frameworks, Reviews & Benchmarks",
            facet_intent="Identify foundational methodologies, systematic reviews, reference benchmarks, and theoretical baselines.",
            linked_subquestion_ids=sq_ids[:2],
            expansion_reason="Contextual & benchmark baseline coverage",
            priority_rank=2,
            queries=FacetApiQueries(
                pubmed=compile_pubmed_query(
                    atomic_terms,
                    ['"Review"[Publication Type]', "benchmark", "validation", "comparative"],
                    composed,
                    exclusions,
                    core_override=pubmed_core,
                ),
                semantic_scholar=compile_semantic_scholar_query(core_terms[:2], f2_mods[:3]),
                arxiv=compile_arxiv_query(atomic_terms, f2_mods, composed, exclusions, core_override=arxiv_core),
                biorxiv_medrxiv=compile_biorxiv_queries(bio_terms, f2_mods[:2]),
                hugging_face=compile_huggingface_query(atomic_terms, understanding.primary_domain, "F2_foundational_contextual"),
            ),
        )
    )

    # F3: Alternative, Contradictory & Robustness Evidence (Targeted at failure modes, OOD, limitations)
    if understanding.primary_domain == "mathematics":
        f3_mods = ["counterexample", "sharpness", "lower bound", "non-convex"]
    elif understanding.primary_domain in ("computer_science_ai", "interdisciplinary_biomedical_ai"):
        f3_mods = ["generalization", "out-of-distribution", "limitations", "failure", "robustness"]
    else:
        f3_mods = ["limitations", "adverse", "resistance", "challenges", "reproducibility"]

    all_facets.append(
        SearchFacet(
            facet_code="F3_alternative_contradictory",
            facet_name="F3: Contradictory Evidence, Robustness & Failure Modes",
            facet_intent="Surface competing hypotheses, negative/null findings, generalizability limits, and methodological bottlenecks.",
            linked_subquestion_ids=[sq_ids[-1]],
            expansion_reason="Evaluate opposing evidence and robustness limits",
            priority_rank=3,
            queries=FacetApiQueries(
                pubmed=compile_pubmed_query(atomic_terms, f3_mods, composed, exclusions, core_override=pubmed_core),
                semantic_scholar=compile_semantic_scholar_query(core_terms[:2], f3_mods[:3]),
                arxiv=compile_arxiv_query(atomic_terms, f3_mods, composed, exclusions, core_override=arxiv_core),
                biorxiv_medrxiv=compile_biorxiv_queries(bio_terms, f3_mods[:2]),
                hugging_face=compile_huggingface_query(atomic_terms, understanding.primary_domain, "F3_alternative_contradictory"),
            ),
        )
    )

    # F4: Recent Developments & Emerging Paradigms
    if understanding.primary_domain == "mathematics":
        f4_mods = ["optimal", "high-dimensional", "stochastic", "regularity"]
    elif understanding.primary_domain == "chemistry":
        f4_mods = ["generative", "machine learning", "interatomic", "autonomous"]
    elif understanding.primary_domain in ("computer_science_ai", "interdisciplinary_biomedical_ai"):
        f4_mods = ["foundation model", "generative", "diffusion", "de novo"]
    elif understanding.primary_domain == "biomedical_clinical":
        f4_mods = ["single-cell", "spatial transcriptomics", "multi-omics", "machine learning"]
    else:
        f4_mods = ["novel", "emerging", "recent advances", "new approach"]
    # When the question names the experimental systems to cover, the fourth search targets those systems instead
    model_systems = understanding.concepts.model_systems
    if model_systems:
        f4_mods = [sp for m in model_systems for sp in m.spellings[:2]][:6]

    all_facets.append(
        SearchFacet(
            facet_code="F4_recent_developments",
            facet_name="F4: Emerging Methods & Targeted Gap Verification",
            facet_intent="Discover recent methodological advances and verify whether candidate gaps have emerging solutions.",
            linked_subquestion_ids=sq_ids[-2:] if len(sq_ids) >= 2 else sq_ids,
            expansion_reason="Recent developments and gap verification",
            priority_rank=4,
            queries=FacetApiQueries(
                pubmed=compile_pubmed_query(atomic_terms, f4_mods, composed, exclusions, core_override=pubmed_core),
                semantic_scholar=compile_semantic_scholar_query(core_terms[:2], f4_mods[:3]),
                arxiv=compile_arxiv_query(atomic_terms, f4_mods, composed, exclusions, core_override=arxiv_core),
                biorxiv_medrxiv=compile_biorxiv_queries(bio_terms, f4_mods[:2]),
                hugging_face=compile_huggingface_query(atomic_terms, understanding.primary_domain, "F4_recent_developments"),
            ),
        )
    )

    # Select active facets based on mode and question complexity (do not force 6 queries on every narrow request)
    if mode == "deep_research":
        active_facets = all_facets[:4]
        budget = 60
    else:
        # In standard mode, use 2 facets (F1 + F2, or F1 + F3 if gap_discovery is the primary intent)
        if understanding.primary_intent == "gap_discovery" or "gap_discovery" in understanding.secondary_intents:
            active_facets = [all_facets[0], all_facets[2]]
        else:
            active_facets = all_facets[:2]
        budget = 20

    coverage_map: Dict[str, List[str]] = {sq.id: [] for sq in understanding.subquestions}
    for f in active_facets:
        for sq_id in f.linked_subquestion_ids:
            coverage_map.setdefault(sq_id, []).append(f.facet_code)

    search_plan = UniversalSearchPlan(
        retrieval_budget_papers=budget,
        active_facet_codes=[f.facet_code for f in active_facets],
        subquestion_coverage_map=coverage_map,
        preferred_sources=composed["preferred_sources"],
        adaptive_expansion_triggered=(mode == "deep_research"),
        expansion_notes=[
            f"Primary domain profile: {composed['primary_domain_display']} ({composed['implementation_tier']})",
            f"Active methodological profiles: {', '.join(composed['active_methodological_profiles'])}",
        ],
    )

    return search_plan, active_facets, composed


def compile_deterministic_decomposition_plan(
    query: str, mode: str = "standard", domain_override: Optional[DomainCategoryType] = None
) -> QueryDecompositionPlan:
    """
    Executes the full 3-stage deterministic Universal Question Understanding (Stage A),
    Universal Search Strategy Planning (Stage B), and Source-Specific Query Compilation (Stage C).
    """
    understanding = understand_research_question(query, domain_override)
    search_plan, active_facets, composed = plan_and_compile_universal_search(understanding, mode=mode)

    return QueryDecompositionPlan(
        detected_domain=composed["legacy_domain_label"],
        extracted_technical_entities=understanding.concepts.normalized_terms or understanding.concepts.main_entities,
        search_facets=active_facets,
        question_understanding=understanding,
        search_plan=search_plan,
        active_profile_summary={
            "primary_domain": composed["primary_domain"],
            "primary_domain_display": composed["primary_domain_display"],
            "secondary_domains": composed["secondary_domains"],
            "methodological_profiles": composed["active_methodological_profiles"],
            "sample_reporting_policy": composed["sample_reporting_policy"],
            "preferred_sources": composed["preferred_sources"],
            "extraction_columns": composed["extraction_columns"],
        },
    )


UNIVERSAL_DECOMPOSITION_SYSTEM_PROMPT = """# ROLE & MISSION
You are the Universal Question Understanding and Multi-Source Search Planning Engine for a domain-adaptive scientific research platform.
Separate scientific question understanding (Stage A) and research strategy planning (Stage B) from source-specific API query compilation (Stage C).

# STAGE A — UNIVERSAL QUESTION UNDERSTANDING
1. Identify primary_intent and secondary_intents from: descriptive, explanatory_mechanistic, causal_intervention, associational, predictive, comparative, methodological, theoretical, exploratory_mapping, gap_discovery.
2. Identify primary_domain from: biomedical_clinical, computer_science_ai, interdisciplinary_biomedical_ai, chemistry, physics, mathematics, engineering_materials, social_sciences, humanities_interdisciplinary. Do NOT assume non-biomedical queries are biomedical.
3. Extract normalized technical entities (fixing typos and resolving ambiguous acronyms in context).

# STAGE B & C — SEARCH FACETS & VALIDATED API COMPILATION
Generate search_facets covering:
- F1_direct_evidence: Direct evidence for the principal question and subquestions.
- F2_foundational_contextual: Foundational frameworks, benchmarks, and systematic reviews (do NOT simply search the literal word "seminal").
- F3_alternative_contradictory: Competing hypotheses, negative results, OOD generalization limits, and methodological bottlenecks.
- F4_recent_developments: Recent advances and targeted knowledge-gap verification.

Source Compilation Rules:
1. PUBMED: Formal Boolean (AND, OR) with [Title/Abstract] or valid MeSH tags. Never invent MeSH headings.
2. SEMANTIC SCHOLAR: Dense natural-language keyword phrases (5-8 words). NO Boolean operators.
3. ARXIV: Valid field prefixes (ti:, abs:, all:) with uppercase AND / OR / ANDNOT connectors.
4. BIORXIV / MEDRXIV: List of 2 short multi-word phrases (3-4 words each).
5. HUGGING FACE: Model, dataset, and architecture tokens only (strip out paper-only words like "review" or "limitations").

Return strictly valid JSON:
{
  "detected_domain": "BIOMEDICAL/AI | CHEMISTRY | PHYSICS | MATHEMATICS | COMPUTATIONAL",
  "extracted_technical_entities": ["string"],
  "search_facets": [
    {
      "facet_code": "F1_direct_evidence",
      "facet_name": "string",
      "facet_intent": "string",
      "queries": {
        "pubmed": "string",
        "semantic_scholar": "string",
        "arxiv": "string",
        "biorxiv_medrxiv": ["string"],
        "hugging_face": "string"
      }
    }
  ]
}"""


async def generate_multi_api_decomposition_plan(
    query: str,
    mode: str,
    config: ApiKeysConfig,
    domain_override: Optional[DomainCategoryType] = None,
) -> QueryDecompositionPlan:
    """
    Runs the 3-stage Universal Question Understanding, Strategy Planning, and Source-Specific
    Query Compilation pipeline, augmented by the configured LLM when available and validated
    against syntax rules. Uses bounded in-memory caching for cost/latency control.
    """
    cache_key = f"{query.strip().lower()}::{mode}::{config.llm_provider}::{config.llm_model}::{domain_override or 'auto'}"
    if cache_key in _DECOMPOSITION_CACHE:
        return _DECOMPOSITION_CACHE[cache_key]

    base_plan = compile_deterministic_decomposition_plan(query, mode=mode, domain_override=domain_override)

    # Only invoke LLM if an API key or local endpoint is configured
    has_llm = bool((config.llm_api_key or "").strip()) or config.llm_provider in ("ollama", "custom")
    if not has_llm:
        _DECOMPOSITION_CACHE[cache_key] = base_plan
        return base_plan

    raw = await call_llm(
        system_prompt=UNIVERSAL_DECOMPOSITION_SYSTEM_PROMPT,
        user_prompt=(
            f"Research Request: {query}\n"
            f"Detected Domain Profile: {base_plan.active_profile_summary.get('primary_domain_display')}\n"
            f"Target Mode: {mode} ({len(base_plan.search_facets)} facets)"
            + (
                "\nRequired constraints (every query for every source must include them): "
                + "; ".join(c.name for c in base_plan.question_understanding.concepts.required_constraints)
                if base_plan.question_understanding and base_plan.question_understanding.concepts.required_constraints
                else ""
            )
        ),
        config=config,
        temperature=0.15,
        max_tokens=1500,
    )
    parsed = extract_json_block(raw) if raw else None
    if not parsed or not isinstance(parsed.get("search_facets"), list) or len(parsed["search_facets"]) == 0:
        _DECOMPOSITION_CACHE[cache_key] = base_plan
        return base_plan

    raw_entities = parsed.get("extracted_technical_entities") or base_plan.extracted_technical_entities
    if isinstance(raw_entities, str):
        raw_entities = [raw_entities]
    elif not isinstance(raw_entities, list):
        raw_entities = base_plan.extracted_technical_entities
    entities = [
        str(e).strip()
        for e in raw_entities
        if str(e).strip() and str(e).strip().lower() not in CONVERSATIONAL_STOP_WORDS
    ]
    if not entities:
        entities = base_plan.extracted_technical_entities

    facets_out: List[SearchFacet] = []
    max_facets = len(base_plan.search_facets)

    for idx, f_obj in enumerate(parsed["search_facets"][:max_facets]):
        fb_facet = base_plan.search_facets[idx % len(base_plan.search_facets)]
        # Models sometimes return facets/queries as bare strings or lists; fall back to the deterministic facet
        if not isinstance(f_obj, dict):
            facets_out.append(fb_facet)
            continue
        q_obj = f_obj.get("queries")
        if not isinstance(q_obj, dict):
            q_obj = {}
        biorxiv_list = q_obj.get("biorxiv_medrxiv")
        if isinstance(biorxiv_list, str):
            biorxiv_list = [biorxiv_list]
        elif not isinstance(biorxiv_list, list) or not biorxiv_list:
            biorxiv_list = fb_facet.queries.biorxiv_medrxiv

        # Validate arXiv & S2 syntax from LLM; fall back to deterministic compiler if invalid
        raw_s2 = str(q_obj.get("semantic_scholar") or fb_facet.queries.semantic_scholar).strip()
        clean_s2 = re.sub(r"\b(AND|OR|NOT)\b", " ", raw_s2)
        clean_s2 = " ".join(clean_s2.split()) or fb_facet.queries.semantic_scholar

        raw_arxiv = str(q_obj.get("arxiv") or fb_facet.queries.arxiv).strip()
        if raw_arxiv.count("(") != raw_arxiv.count(")") or not any(p in raw_arxiv for p in ("ti:", "abs:", "all:")):
            raw_arxiv = fb_facet.queries.arxiv

        raw_pubmed = str(q_obj.get("pubmed") or fb_facet.queries.pubmed).strip()
        if raw_pubmed.count("(") != raw_pubmed.count(")"):
            raw_pubmed = fb_facet.queries.pubmed

        facets_out.append(
            SearchFacet(
                facet_code=str(f_obj.get("facet_code") or fb_facet.facet_code),
                facet_name=str(f_obj.get("facet_name") or fb_facet.facet_name),
                facet_intent=str(f_obj.get("facet_intent") or fb_facet.facet_intent),
                linked_subquestion_ids=fb_facet.linked_subquestion_ids,
                expansion_reason=fb_facet.expansion_reason,
                priority_rank=fb_facet.priority_rank,
                queries=FacetApiQueries(
                    pubmed=raw_pubmed,
                    semantic_scholar=clean_s2,
                    arxiv=raw_arxiv,
                    biorxiv_medrxiv=[str(x).strip() for x in biorxiv_list if str(x).strip()][:3],
                    hugging_face=str(q_obj.get("hugging_face") or fb_facet.queries.hugging_face).strip(),
                ),
            )
        )

    # The model may drop a place/population restriction; re-impose it on every LLM-written query
    required = (
        base_plan.question_understanding.concepts.required_constraints if base_plan.question_understanding else []
    )
    facets_out = [f.model_copy(update={"queries": apply_required_constraints(f.queries, required)}) for f in facets_out]

    final_plan = QueryDecompositionPlan(
        detected_domain=base_plan.detected_domain,
        extracted_technical_entities=entities,
        search_facets=facets_out,
        question_understanding=base_plan.question_understanding,
        search_plan=base_plan.search_plan,
        active_profile_summary=base_plan.active_profile_summary,
    )
    if len(_DECOMPOSITION_CACHE) > 64:
        _DECOMPOSITION_CACHE.clear()
    _DECOMPOSITION_CACHE[cache_key] = final_plan
    return final_plan
