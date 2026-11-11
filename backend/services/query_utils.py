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
from services.query_planning import (
    ASPECTS_BY_MODE,
    SEARCH_PLANNING_PROMPT,
    aspects_from_model,
    build_search_plan,
    concepts_from_model,
)
from services.question_parsing import (
    QUESTION_FRAMING_WORDS,
    GENERIC_CONCEPT_WORDS,
    concept_blocks,
    spelling_variants,
    specific_words,
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


def correct_typos_keep_punctuation(query: str) -> str:
    """Common typo fixes, keeping quotes, commas and apostrophes the concept parser relies on."""
    return re.sub(r"[A-Za-z]+(?![A-Za-z'\u2019])", lambda m: COMMON_TYPO_CORRECTIONS.get(m.group(0).lower(), m.group(0)),
                  query or "")


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
    phrases = concept_phrases(correct_typos_keep_punctuation(query), CONVERSATIONAL_STOP_WORDS, max_phrases=6)
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
    variants = [term] + [v for v in spelling_variants(term)] if " " not in term else [term]
    tagged = [f"{v}[Title/Abstract]" for v in variants] if " " not in term else [f'"{term}"[Title/Abstract]']
    return tagged[0] if len(tagged) == 1 else "(" + " OR ".join(tagged) + ")"


def build_required_pubmed_core(phrases: List[str], constraints: List[RequiredConstraint],
                               question: str = "") -> Optional[str]:
    """
    Required part of every PubMed query: the main concept blocks (each concept as a phrase or all of its words;
    concepts listed side by side in the question are alternatives) AND every place/population constraint.
    Returns None when the question has no phrases.
    """
    # A purely generic concept ("diseases") must not be required: it would exclude papers naming a specific disease
    phrases = [p for p in phrases if any(w not in GENERIC_CONCEPT_WORDS for w in p.lower().split())] or phrases
    if not phrases:
        return None
    blocks = []
    for group in concept_blocks(question, phrases) if question else [[ph] for ph in phrases[:2]]:
        alts = []
        for ph in group:
            words = specific_words(ph) if len(ph.split()) > 1 else ph.split()
            alts += [_tiab(ph)] + ([f"({' AND '.join(_tiab(w) for w in words)})" if len(words) > 1 else _tiab(words[0])]
                                   if len(ph.split()) > 1 else [])
        blocks.append("(" + " OR ".join(alts) + ")")
    for c in constraints:
        blocks.append("(" + " OR ".join(_tiab(sp) for sp in c.spellings[:4]) + ")")
    return " AND ".join(blocks)


def build_required_arxiv_core(phrases: List[str], constraints: List[RequiredConstraint],
                              question: str = "") -> Optional[str]:
    # A purely generic concept ("diseases") must not be required: it would exclude papers naming a specific disease
    phrases = [p for p in phrases if any(w not in GENERIC_CONCEPT_WORDS for w in p.lower().split())] or phrases
    if not phrases:
        return None
    blocks = []
    for group in concept_blocks(question, phrases) if question else [[ph] for ph in phrases[:2]]:
        alts = []
        for ph in group:
            words = [re.sub(r"[^a-zA-Z0-9_-]", "", w) for w in ph.split()]
            words = [w for w in words if w]
            if words:
                alts.append(f'all:"{" ".join(words)}"' if len(words) > 1 else f"(ti:{words[0]} OR abs:{words[0]})")
        if alts:
            blocks.append(alts[0] if len(alts) == 1 else "(" + " OR ".join(alts) + ")")
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
    model_plan: Optional[Dict[str, Any]] = None,
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

    # Concepts, aspects and the searches for every database (services/query_planning.py). The model's plan, when
    # given, supplies concepts with synonyms and aspects; otherwise both come from the question and its field.
    atomic_terms = extract_core_scientific_terms(understanding.normalized_query, max_terms=6)
    hf_query = compile_huggingface_query(atomic_terms, understanding.primary_domain, "F1_direct_evidence")
    model_plan = model_plan or {}
    active_facets = build_search_plan(
        understanding, mode, hf_query,
        concepts=model_plan.get("concepts"), aspects=model_plan.get("aspects"), breadth=model_plan.get("breadth"),
        planned_by="model" if model_plan.get("concepts") or model_plan.get("aspects") else "rules",
    )
    budget = 60 if mode == "deep_research" else 20

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
    query: str, mode: str = "standard", domain_override: Optional[DomainCategoryType] = None,
    model_plan: Optional[Dict[str, Any]] = None,
) -> QueryDecompositionPlan:
    """
    Question understanding, search planning and query compilation. `model_plan` (validated concepts, aspects and
    breadth from the model) refines the plan; the search strings are always compiled by rules.
    """
    understanding = understand_research_question(query, domain_override)
    search_plan, active_facets, composed = plan_and_compile_universal_search(understanding, mode=mode, model_plan=model_plan)
    concept_names = [c.name for c in understanding.concepts.search_concepts]

    return QueryDecompositionPlan(
        detected_domain=composed["legacy_domain_label"],
        extracted_technical_entities=concept_names or understanding.concepts.normalized_terms or understanding.concepts.main_entities,
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


PLANNING_BUDGET_S = 35.0


async def generate_multi_api_decomposition_plan(
    query: str,
    mode: str,
    config: ApiKeysConfig,
    domain_override: Optional[DomainCategoryType] = None,
) -> QueryDecompositionPlan:
    """
    Plans the searches for a request. With a model configured, the model names the concepts (with synonyms), the
    breadth and the aspects; these are validated and the search strings are compiled by rules. If the model is
    unavailable or its plan is unusable, the plan comes from the question's own words and a per-field template.
    """
    cache_key = f"{query.strip().lower()}::{mode}::{config.llm_provider}::{config.llm_model}::{domain_override or 'auto'}"
    if cache_key in _DECOMPOSITION_CACHE:
        return _DECOMPOSITION_CACHE[cache_key].model_copy(deep=True)

    base_plan = compile_deterministic_decomposition_plan(query, mode=mode, domain_override=domain_override)
    final_plan = base_plan
    has_llm = bool((config.llm_api_key or "").strip()) or config.llm_provider in ("ollama", "custom")
    if has_llm:
        understanding = base_plan.question_understanding
        constraints = understanding.concepts.required_constraints if understanding else []
        raw = await call_llm(
            system_prompt=SEARCH_PLANNING_PROMPT,
            user_prompt=(
                f"Research request: {query}\n"
                f"Field: {base_plan.active_profile_summary.get('primary_domain_display')}\n"
                f"Number of aspects: {ASPECTS_BY_MODE.get(mode, 4)}"
                + (f"\nPlaces / populations already handled (leave out): {'; '.join(c.name for c in constraints)}"
                   if constraints else "")
            ),
            config=config,
            temperature=0.1,
            max_tokens=1100,
            # Planning must not hold up the search: past this budget the rule-based plan is used
            budget_s=PLANNING_BUDGET_S,
        )
        parsed = extract_json_block(raw) if raw else None
        if isinstance(parsed, dict) and understanding:
            model_plan = {
                "concepts": concepts_from_model(parsed.get("concepts"), understanding),
                "aspects": aspects_from_model(parsed.get("aspects"), ASPECTS_BY_MODE.get(mode, 4)),
                "breadth": parsed.get("breadth") if parsed.get("breadth") in ("broad", "focused") else None,
            }
            if model_plan["concepts"] or model_plan["aspects"]:
                final_plan = compile_deterministic_decomposition_plan(
                    query, mode=mode, domain_override=domain_override, model_plan=model_plan)

    if len(_DECOMPOSITION_CACHE) > 64:
        _DECOMPOSITION_CACHE.clear()
    _DECOMPOSITION_CACHE[cache_key] = final_plan
    return final_plan.model_copy(deep=True)
