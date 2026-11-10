"""
Search planning: turn a research request into (1) the concepts every relevant paper must be about, with their
synonyms, and (2) a handful of aspects that together cover the request. Each aspect becomes its own search.

The model (when one is configured) only names concepts, synonyms and aspects. The search strings for every
database are then compiled here by fixed rules, so they are always valid, always keep the topic and any
place/population restriction, and can be shown to the user exactly as run. Without a model, concepts come from
the question's own words and aspects from a per-field template.

  core search      concepts only                         -> the most relevant papers overall
  reviews search   concepts AND review/meta-analysis      -> overviews and pooled evidence
  aspect searches  concepts AND the aspect's terms        -> breadth across the sub-areas of the question
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models import (
    FacetApiQueries,
    RequiredConstraint,
    ResearchSubquestion,
    SearchAspect,
    SearchConcept,
    SearchFacet,
    UniversalQuestionUnderstanding,
)
from services.question_parsing import (
    GENERIC_CONCEPT_WORDS,
    concept_blocks,
    is_method_concept,
    normalize_spelling,
    specific_words,
    spelling_variants,
)

MAX_REQUIRED_CONCEPTS = 2          # concept blocks ANDed in every search (more makes searches too narrow)
ASPECTS_BY_MODE = {"standard": 4, "deep_research": 6}
# Records requested from each database: (core search, every other search)
RESULTS_PER_SEARCH = {"standard": (30, 15), "deep_research": (50, 25)}
TECHNICAL_DOMAINS = {"computer_science_ai", "mathematics", "physics", "chemistry", "engineering_materials",
                     "interdisciplinary_biomedical_ai"}

# ---------------------------------------------------------------------------------------------------------------
# Breadth: a topic to survey, or a specific question relating concepts
# ---------------------------------------------------------------------------------------------------------------
_RELATION = re.compile(
    r"^\s*(?:how|why|does|do|did|is|are|can|could|should|will|which)\b|"
    r"\b(?:effects?|impact|affects?|causes?|prevents?|reduces?|improves?|increases?|associated|association|"
    r"relationship|relation|versus|vs\.?|compared|comparison|better than|risk of|role of|predicts?|"
    r"efficacy|effectiveness|influence)\b", re.I)


def subject_concepts(phrases: Sequence[str]) -> List[str]:
    """Concept phrases that name what the question is about (not methods, not purely generic words)."""
    return [p for p in phrases if not is_method_concept(p) and any(w not in GENERIC_CONCEPT_WORDS for w in p.split())]


def detect_breadth(question: str, phrases: Sequence[str]) -> str:
    """'broad' for a topic ("gut microbiome", "machine learning in drug discovery"), else 'focused'."""
    n = len(subject_concepts(phrases))
    if n <= 1:
        return "broad"
    if len(list(phrases)) <= 2 and not _RELATION.search(question or ""):
        return "broad"
    return "focused"


# ---------------------------------------------------------------------------------------------------------------
# Aspect templates used without a model (label, search terms, sub-question; {t} is the topic)
# ---------------------------------------------------------------------------------------------------------------
_A = lambda label, terms, question: (label, terms, question)  # noqa: E731

_NULL_ASPECT = _A("Null or conflicting findings",
                  ["no significant", "not associated", "no effect", "conflicting", "controversy", "inconsistent"],
                  "Which studies of {t} report null or conflicting results?")
_MECHANISM_ASPECT = _A("Mechanisms", ["mechanism", "pathway", "signaling", "molecular", "pathogenesis"],
                       "What mechanisms are proposed for {t}?")

ASPECT_TEMPLATES: Dict[str, List[Tuple[str, List[str], str]]] = {
    "biomedical_overview": [
        _A("Role in health and disease", ["disease", "risk", "association", "health", "pathogenesis"],
           "How is {t} involved in health and disease?"),
        _MECHANISM_ASPECT,
        _A("Diagnosis and treatment", ["treatment", "therapy", "diagnosis", "clinical", "intervention", "trial"],
           "How does {t} bear on diagnosis and treatment?"),
        _A("Outcomes and safety", ["outcomes", "prognosis", "mortality", "safety", "adverse", "efficacy"],
           "What outcomes and safety findings are reported for {t}?"),
        _A("Epidemiology and risk factors", ["prevalence", "incidence", "epidemiology", "risk factors", "burden"],
           "How common is {t} and what are its risk factors?"),
        _A("Open problems and limitations", ["limitations", "challenges", "controversy", "unclear", "future"],
           "What remains uncertain about {t}?"),
    ],
    "biomedical_place": [
        _A("Epidemiology and risk factors", ["prevalence", "incidence", "epidemiology", "risk factors", "burden"],
           "How common is {t} and what are its risk factors?"),
        _A("Diagnosis and treatment", ["treatment", "therapy", "diagnosis", "management", "intervention"],
           "How is {t} diagnosed and treated?"),
        _A("Outcomes and mortality", ["outcomes", "mortality", "prognosis", "survival", "complications"],
           "What outcomes are reported for {t}?"),
        _A("Prevention and health services", ["prevention", "screening", "policy", "guidelines", "access",
                                              "health system"],
           "What is known about prevention and health services for {t}?"),
        _MECHANISM_ASPECT,
        _A("Open problems and limitations", ["limitations", "challenges", "gaps", "future"],
           "What remains uncertain about {t}?"),
    ],
    "biomedical_effect": [
        _A("Trials and controlled studies", ["randomized", "trial", "controlled", "placebo", "cohort"],
           "What do trials and controlled studies show about {t}?"),
        _MECHANISM_ASPECT,
        _A("Safety and adverse effects", ["safety", "adverse", "side effects", "harm", "tolerability"],
           "What harms or adverse effects are reported for {t}?"),
        _NULL_ASPECT,
        _A("Who benefits: subgroups and dose", ["subgroup", "dose", "duration", "moderator", "heterogeneity"],
           "For whom and under what conditions does {t} apply?"),
        _A("Long-term outcomes", ["long-term", "follow-up", "mortality", "outcomes"],
           "What are the long-term outcomes of {t}?"),
    ],
    "biomedical_association": [
        _A("Cohort and longitudinal evidence", ["cohort", "longitudinal", "prospective", "follow-up", "incidence"],
           "What do cohort and longitudinal studies show about {t}?"),
        _MECHANISM_ASPECT,
        _A("Causality and confounding", ["causal", "mendelian randomization", "confounding", "reverse causation"],
           "Is the relationship in {t} causal?"),
        _NULL_ASPECT,
        _A("Clinical outcomes and prediction", ["outcomes", "prognosis", "prediction", "biomarker", "risk"],
           "What clinical outcomes or predictive value are reported for {t}?"),
        _A("Interventions", ["treatment", "intervention", "therapy", "trial"],
           "Do interventions on {t} change outcomes?"),
    ],
    "biomedical_mechanism": [
        _A("Mechanisms and pathways", ["mechanism", "pathway", "signaling", "molecular", "regulation"],
           "What mechanisms and pathways explain {t}?"),
        _A("Experimental models", ["mouse", "in vitro", "in vivo", "model", "knockout"],
           "What do experimental models show about {t}?"),
        _A("Human evidence", ["patients", "clinical", "cohort", "human", "tissue"],
           "What human evidence supports {t}?"),
        _A("Therapeutic targeting", ["treatment", "therapy", "inhibitor", "target", "drug"],
           "Can {t} be targeted therapeutically?"),
        _NULL_ASPECT,
        _A("Genetics and risk factors", ["genetic", "variant", "risk factors", "susceptibility"],
           "What genetic or risk factors are linked to {t}?"),
    ],
    "computing": [
        _A("Methods and models", ["method", "model", "architecture", "algorithm", "framework"],
           "What methods and models are used for {t}?"),
        _A("Applications", ["application", "case study", "deployment", "real-world"],
           "Where is {t} applied?"),
        _A("Evaluation and benchmarks", ["benchmark", "evaluation", "dataset", "comparison", "performance"],
           "How is {t} evaluated and benchmarked?"),
        _A("Limitations and robustness", ["limitations", "robustness", "generalization", "bias", "failure"],
           "What are the limitations of {t}?"),
        _A("Data and resources", ["dataset", "corpus", "open-source", "toolkit"],
           "What data and resources exist for {t}?"),
        _A("Theory and analysis", ["theory", "analysis", "bounds", "interpretability"],
           "What theory explains {t}?"),
    ],
    "chemistry": [
        _A("Synthesis and preparation", ["synthesis", "preparation", "route", "catalyst"],
           "How is {t} synthesised or prepared?"),
        _A("Mechanism and theory", ["mechanism", "kinetics", "DFT", "computational"],
           "What mechanism or theory explains {t}?"),
        _A("Properties and characterisation", ["properties", "characterization", "structure", "spectroscopy"],
           "What properties are reported for {t}?"),
        _A("Applications", ["application", "performance", "device", "catalysis"], "What is {t} used for?"),
        _A("Stability and limitations", ["stability", "degradation", "limitations", "scale-up"],
           "What limits {t}?"),
        _A("Selectivity and yield", ["selectivity", "yield", "efficiency"], "What yields and selectivity are reported for {t}?"),
    ],
    "physics": [
        _A("Theory and models", ["theory", "model", "analytical", "prediction"], "What theory describes {t}?"),
        _A("Experiments and measurements", ["experiment", "measurement", "observation", "detector"],
           "What do experiments show about {t}?"),
        _A("Simulations", ["simulation", "numerical", "lattice", "monte carlo"], "What do simulations show about {t}?"),
        _A("Open questions", ["open question", "tension", "anomaly", "limits"], "What is unresolved about {t}?"),
        _A("Applications and implications", ["application", "implications", "constraints"],
           "What are the implications of {t}?"),
        _A("Methods and instruments", ["method", "technique", "instrument", "sensitivity"],
           "What methods are used to study {t}?"),
    ],
    "mathematics": [
        _A("Main results", ["theorem", "proof", "result"], "What are the main results on {t}?"),
        _A("Bounds and estimates", ["bound", "estimate", "asymptotic", "inequality"],
           "What bounds are known for {t}?"),
        _A("Open problems", ["conjecture", "open problem", "counterexample"], "What is open about {t}?"),
        _A("Methods and techniques", ["method", "technique", "approach"], "What techniques are used for {t}?"),
        _A("Algorithms and applications", ["algorithm", "application", "computation"],
           "What algorithms or applications involve {t}?"),
        _A("Generalisations", ["generalization", "higher-dimensional", "extension"], "How has {t} been generalised?"),
    ],
    "engineering": [
        _A("Design and methods", ["design", "method", "fabrication", "process"], "How is {t} designed or made?"),
        _A("Performance and testing", ["performance", "testing", "experimental", "efficiency"],
           "How does {t} perform?"),
        _A("Modelling and simulation", ["simulation", "model", "finite element", "optimization"],
           "How is {t} modelled?"),
        _A("Applications", ["application", "case study", "deployment"], "Where is {t} applied?"),
        _A("Durability and limitations", ["durability", "failure", "fatigue", "limitations", "cost"],
           "What limits {t}?"),
        _A("Materials", ["material", "composite", "alloy", "microstructure"], "What materials matter for {t}?"),
    ],
    "social": [
        _A("Effects and outcomes", ["effect", "impact", "outcomes", "consequences"], "What are the effects of {t}?"),
        _A("Determinants and causes", ["determinants", "factors", "causes", "predictors"], "What drives {t}?"),
        _A("Policy and interventions", ["policy", "intervention", "program", "reform"],
           "What policies or interventions address {t}?"),
        _A("Critiques and debates", ["critique", "debate", "controversy", "limitations"],
           "What is debated about {t}?"),
        _A("Inequality and context", ["inequality", "gender", "low-income", "cross-country"],
           "How does {t} differ across groups and settings?"),
        _A("Measurement and methods", ["measurement", "survey", "methodology", "identification"],
           "How is {t} measured and studied?"),
    ],
    "humanities": [
        _A("Historical context", ["history", "historical", "century", "origins"], "What is the history of {t}?"),
        _A("Interpretation and theory", ["interpretation", "theory", "criticism", "reading"],
           "How is {t} interpreted?"),
        _A("Reception and influence", ["reception", "influence", "legacy", "adaptation"],
           "How was {t} received and what did it influence?"),
        _A("Debates", ["debate", "critique", "controversy"], "What is debated about {t}?"),
        _A("Sources and methods", ["archive", "sources", "methodology", "text"], "What sources are used to study {t}?"),
        _A("Comparative perspectives", ["comparative", "cross-cultural", "translation"],
           "How does {t} compare across contexts?"),
    ],
}
_DOMAIN_TEMPLATE = {
    "computer_science_ai": "computing", "interdisciplinary_biomedical_ai": "computing", "chemistry": "chemistry",
    "physics": "physics", "mathematics": "mathematics", "engineering_materials": "engineering",
    "social_sciences": "social", "humanities_interdisciplinary": "humanities",
}


def default_aspects(understanding: UniversalQuestionUnderstanding, topic: str, n: int) -> List[SearchAspect]:
    """Aspects for the question's field and type, used when no model plans the search."""
    domain, intent = understanding.primary_domain, understanding.primary_intent
    key = _DOMAIN_TEMPLATE.get(domain)
    if key is None:
        if intent in ("causal_intervention", "comparative"):
            key = "biomedical_effect"
        elif intent in ("associational", "predictive"):
            key = "biomedical_association"
        elif intent == "explanatory_mechanistic":
            key = "biomedical_mechanism"
        elif any(c.kind == "place" for c in understanding.concepts.required_constraints):
            key = "biomedical_place"
        else:
            key = "biomedical_overview"
    aspects: List[SearchAspect] = []
    systems = understanding.concepts.model_systems
    if systems:  # the question names the experimental systems it wants covered
        aspects.append(SearchAspect(
            label="Model systems", terms=[sp for m in systems for sp in m.spellings[:2]][:8],
            question=f"What do {', '.join(m.name for m in systems)} show about {topic}?"))
    for label, terms, question in ASPECT_TEMPLATES[key]:
        if len(aspects) >= n:
            break
        aspects.append(SearchAspect(label=label, terms=list(terms), question=question.format(t=topic)))
    return aspects


# ---------------------------------------------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------------------------------------------
def concepts_from_question(understanding: UniversalQuestionUnderstanding) -> List[SearchConcept]:
    """Required concepts from the question's own words; concepts listed side by side become alternatives."""
    phrases = subject_concepts(understanding.concepts.concept_phrases) or list(understanding.concepts.concept_phrases)
    blocks = concept_blocks(understanding.raw_query, phrases, max_blocks=3)
    return [SearchConcept(name=b[0], synonyms=list(b[1:])) for b in blocks if b]


def _clean_term(text: Any, max_words: int = 6) -> str:
    t = re.sub(r"\[[^\]]*\]|[\"()\[\]{}*]", " ", str(text or ""))
    t = re.sub(r"\b(?:AND|OR|NOT)\b", " ", t)
    words = t.split()
    return " ".join(words[:max_words]).strip(" ,.;:")


def concepts_from_model(raw: Any, understanding: UniversalQuestionUnderstanding) -> List[SearchConcept]:
    """Validate the model's concepts. Returns [] if they do not relate to the question (the caller falls back)."""
    if not isinstance(raw, list):
        return []
    constraint_words = {sp.lower() for c in understanding.concepts.required_constraints for sp in c.spellings}
    out: List[SearchConcept] = []
    for item in raw[:3]:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = _clean_term(item.get("name"))
        if not name or name.lower() in constraint_words:
            continue
        if all(w in GENERIC_CONCEPT_WORDS for w in name.lower().split()):
            continue
        syns = item.get("synonyms") if isinstance(item.get("synonyms"), list) else []
        clean = [s for s in dict.fromkeys(_clean_term(x) for x in syns[:6]) if s and s.lower() != name.lower()
                 and s.lower() not in constraint_words][:4]
        out.append(SearchConcept(name=name, synonyms=clean))
    if not out:
        return []
    # The leading concept must be anchored in the question's own words (guards against a swapped or invented topic)
    anchor = set(re.findall(r"[a-z0-9]{3,}", normalize_spelling(
        understanding.raw_query + " " + " ".join(understanding.concepts.concept_phrases))))
    lead = set(re.findall(r"[a-z0-9]{3,}", normalize_spelling(" ".join([out[0].name] + out[0].synonyms))))
    return out if (anchor & lead) else []


def aspects_from_model(raw: Any, n: int) -> List[SearchAspect]:
    if not isinstance(raw, list):
        return []
    out: List[SearchAspect] = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = _clean_term(item.get("label"), 7)
        terms_raw = item.get("terms") if isinstance(item.get("terms"), list) else []
        terms = [t for t in dict.fromkeys(_clean_term(x, 4) for x in terms_raw[:8]) if t][:6]
        if not label or label.lower() in seen or len(terms) < 2:
            continue
        seen.add(label.lower())
        out.append(SearchAspect(label=label, terms=terms, question=str(item.get("question") or "").strip()[:200]))
        if len(out) >= n:
            break
    return out


SEARCH_PLANNING_PROMPT = """You plan literature searches. Read the research request and return ONLY a JSON object:
{
 "concepts": [{"name": "...", "synonyms": ["..."]}],
 "breadth": "broad" or "focused",
 "aspects": [{"label": "...", "terms": ["..."], "question": "..."}]
}
concepts: 1 to 3 concepts that EVERY relevant paper must be about, most important first. Use the standard
scientific name and correct any spelling mistakes. synonyms: up to 4 exact alternative names, abbreviations or
spellings of the same thing. Never list narrower sub-topics, examples or broader fields as synonyms.
breadth: "broad" if the request is a topic to survey, "focused" if it is a specific question relating concepts.
aspects: distinct aspects that together cover the request. For a broad topic give its main sub-areas. For a focused
question give the kinds of evidence needed to answer it (for example clinical outcomes, mechanisms, conflicting
findings). label: 2 to 4 words. terms: 3 to 6 single words or short phrases likely to appear in the titles or
abstracts of papers on that aspect. question: one sub-question in plain words.
Rules: no Boolean operators, quotes or field tags anywhere. Do not repeat concept names inside aspect terms.
Do not put places or populations in concepts; they are handled separately."""


# ---------------------------------------------------------------------------------------------------------------
# Compiling the searches
# ---------------------------------------------------------------------------------------------------------------
def _tiab(term: str) -> str:
    return f'"{term}"[Title/Abstract]'


def _pubmed_concept_block(concept: SearchConcept) -> str:
    alts: List[str] = []
    for name in [concept.name] + list(concept.synonyms):
        name = name.strip()
        if not name:
            continue
        words = name.split()
        if len(words) == 1:
            alts += [f"{v}[Title/Abstract]" for v in [name.lower()] + spelling_variants(name)]
        else:
            alts.append(_tiab(name))
            spec = specific_words(name)
            # The same concept written differently: all of its specific words, each in any spelling
            parts = ["(" + " OR ".join(f"{v}[Title/Abstract]" for v in [w] + spelling_variants(w)) + ")"
                     if spelling_variants(w) else f"{w}[Title/Abstract]" for w in spec]
            alts.append(parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")")
    return "(" + " OR ".join(dict.fromkeys(alts)) + ")"


def _pubmed_constraints(constraints: Sequence[RequiredConstraint]) -> List[str]:
    return ["(" + " OR ".join((f'"{sp}"[Title/Abstract]' if " " in sp else f"{sp}[Title/Abstract]")
                              for sp in c.spellings[:4]) + ")" for c in constraints]


def _arxiv_word(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _-]", "", text).strip()


def _arxiv_concept_block(concept: SearchConcept) -> str:
    alts = []
    for name in [concept.name] + list(concept.synonyms):
        w = _arxiv_word(name)
        if not w:
            continue
        if " " not in w:
            alts.append(f"(ti:{w} OR abs:{w})")
            continue
        alts.append(f'all:"{w}"')
        spec = [x for x in specific_words(w) if len(x) > 2]
        if len(w.split()) >= 3 and len(spec) >= 2:  # a long phrase is rarely written word for word
            alts.append("(" + " AND ".join(f"all:{x}" for x in spec) + ")")
    alts = list(dict.fromkeys(alts))
    return alts[0] if len(alts) == 1 else "(" + " OR ".join(alts) + ")"


def compile_facet(
    aspect: SearchAspect,
    code: str,
    concepts: Sequence[SearchConcept],
    constraints: Sequence[RequiredConstraint],
    huggingface_query: str,
    subquestion_id: str,
    rank: int,
) -> SearchFacet:
    """One search (for every database) for an aspect of the question, built by fixed rules."""
    required = list(concepts[:MAX_REQUIRED_CONCEPTS])
    names = " ".join(c.name for c in required)
    place = " ".join(c.name for c in constraints)
    terms = [t for t in aspect.terms if t]

    pm_blocks = [_pubmed_concept_block(c) for c in required] + _pubmed_constraints(constraints)
    ax_blocks = [_arxiv_concept_block(c) for c in required] + [
        "(" + " OR ".join(f'all:"{_arxiv_word(sp)}"' if " " in sp else f"all:{_arxiv_word(sp)}"
                          for sp in c.spellings[:3] if _arxiv_word(sp)) + ")" for c in constraints]
    if aspect.kind == "reviews":
        pm_blocks.append('("systematic review"[Publication Type] OR "meta-analysis"[Publication Type] OR '
                         '"review"[Publication Type])')
        ax_blocks.append("(ti:review OR ti:survey OR abs:review OR abs:survey)")
        s2 = f"{names} {place} systematic review meta-analysis"
        bio = [f"{names} {place} review".strip(), f"{names} {place} meta-analysis".strip()]
    elif aspect.kind == "core" or not terms:
        s2 = f"{names} {place}"
        bio = [f"{names} {place}".strip()]
    else:
        pm_blocks.append("(" + " OR ".join(_tiab(t) for t in terms[:6]) + ")")
        ax_terms = [_arxiv_word(t) for t in terms[:5] if _arxiv_word(t)]
        if ax_terms:
            ax_blocks.append("(" + " OR ".join(f'abs:"{t}"' if " " in t else f"abs:{t}" for t in ax_terms) + ")")
        s2 = f"{names} {place} {' '.join(terms[:3])}"
        bio = [f"{names} {place} {t}".strip() for t in terms[:2]]

    return SearchFacet(
        facet_code=code,
        facet_name=aspect.label,
        facet_intent=aspect.question,
        linked_subquestion_ids=[subquestion_id],
        expansion_reason={"core": "Most relevant papers on the topic", "reviews": "Reviews and pooled evidence"}.get(
            aspect.kind, "Coverage of one aspect of the question"),
        priority_rank=rank,
        kind=aspect.kind,
        queries=FacetApiQueries(
            pubmed=" AND ".join(pm_blocks),
            semantic_scholar=" ".join(s2.split()),
            arxiv=" AND ".join(b for b in ax_blocks if b and b != "()"),
            biorxiv_medrxiv=[" ".join(b.split()) for b in bio if b.strip()],
            hugging_face=huggingface_query,
        ),
    )


def build_search_plan(
    understanding: UniversalQuestionUnderstanding,
    mode: str,
    huggingface_query: str,
    concepts: Optional[List[SearchConcept]] = None,
    aspects: Optional[List[SearchAspect]] = None,
    breadth: Optional[str] = None,
    planned_by: str = "rules",
) -> List[SearchFacet]:
    """
    Fills in the understanding's concepts, breadth, aspects and sub-questions, and returns the searches to run:
    the core search, the reviews search, then one per aspect.
    """
    n_aspects = ASPECTS_BY_MODE.get(mode, 4)
    concepts = concepts or concepts_from_question(understanding)
    if not concepts:
        fallback = " ".join(understanding.concepts.main_entities[:3]) or understanding.normalized_query[:80]
        concepts = [SearchConcept(name=fallback)]
    topic = " and ".join(c.name for c in concepts[:MAX_REQUIRED_CONCEPTS])
    # A question that relates concepts ("How does X affect Y?") is focused whatever the model says; otherwise the
    # model's reading is used when it gave one
    by_rules = detect_breadth(understanding.raw_query, understanding.concepts.concept_phrases)
    breadth = "focused" if by_rules == "focused" else (breadth if breadth in ("broad", "focused") else by_rules)
    aspects = (aspects or [])[:n_aspects]
    if len(aspects) < 2:
        aspects = default_aspects(understanding, topic, n_aspects)
        planned_by = "rules" if planned_by == "rules" else "model (concepts) and rules (aspects)"
    for a in aspects:
        a.question = a.question or f"What is known about {a.label.lower()} of {topic}?"

    core = SearchAspect(label="Core search", kind="core", question=understanding.raw_query.strip())
    reviews = SearchAspect(label="Reviews and meta-analyses", kind="reviews",
                           question=f"What do reviews and pooled analyses conclude about {topic}?")
    understanding.concepts.search_concepts = concepts
    understanding.breadth = breadth  # type: ignore[assignment]
    understanding.aspects = [core, reviews] + aspects
    understanding.planned_by = planned_by
    understanding.subquestions = [
        ResearchSubquestion(id="SQ1", question_text=understanding.raw_query.strip(), category="direct_evidence",
                            relationship_to_main="The question as asked", priority="high")
    ] + [
        ResearchSubquestion(
            id=f"SQ{i + 2}", question_text=a.question, relationship_to_main=f"Aspect: {a.label}",
            category="validation_robustness" if re.search(r"null|conflict|limit|debate|open", a.label, re.I)
            else "contextual_evidence",
            priority="medium", depends_on=["SQ1"])
        for i, a in enumerate(aspects)
    ]
    constraints = understanding.concepts.required_constraints
    facets = [
        compile_facet(core, "S1_core", concepts, constraints, huggingface_query, "SQ1", 1),
        compile_facet(reviews, "S2_reviews", concepts, constraints, huggingface_query, "SQ1", 2),
    ]
    for i, a in enumerate(aspects):
        facets.append(compile_facet(a, f"A{i + 1}_aspect", concepts, constraints, huggingface_query, f"SQ{i + 2}", i + 3))
    return facets


def results_per_search(mode: str, kind: str) -> int:
    core, other = RESULTS_PER_SEARCH.get(mode, RESULTS_PER_SEARCH["standard"])
    return core if kind == "core" else other


def searches_arxiv(domain: str, kind: str) -> bool:
    """arXiv is searched for every aspect in technical fields, and only in the core search otherwise."""
    return kind == "core" or domain in TECHNICAL_DOMAINS
