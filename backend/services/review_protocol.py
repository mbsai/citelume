"""
The review protocol: the decisions behind a review, recorded before any search runs.

A protocol holds the review type (and why), the parts of the question with their synonyms, the populations or
places every study must name, the aspects that get their own searches, and the eligibility criteria. It drives
both the searches and study selection, so the same protocol and the same records always give the same studies.

* `default_protocol` builds the protocol the tool would follow on its own, from the rule-based reading of the
  question. Running with it reproduces a run without a protocol.
* `propose_protocol` returns that default for the user to confirm or edit, optionally with the model's extra
  synonyms and aspects as suggestions. Suggestions are never applied automatically.
* `plan_from_protocol` compiles the searches from a (possibly edited) protocol. No model is involved.
"""
from __future__ import annotations

import re
from typing import List, Optional

from models import (
    ApiKeysConfig,
    EligibilityCriteria,
    ProtocolAspect,
    ProtocolConcept,
    ProtocolConstraint,
    ProtocolSuggestion,
    QueryDecompositionPlan,
    RequiredConstraint,
    ReviewProtocol,
    SearchAspect,
    SearchConcept,
)
from services.paper_selection import question_concepts
from services.question_parsing import is_method_concept

REVIEW_TYPE_REASONS = {
    "narrative": "A narrative review summarises and interprets what the included studies found. It suits background "
                 "and overview questions, and is what a quick answer gives.",
    "systematic_prisma": "A systematic-style review includes every study that meets pre-set criteria, judges each "
                         "one's risk of bias and reports to PRISMA 2020. It suits a specific question about an "
                         "effect or association.",
    "scoping": "A scoping review maps what research exists, how it was done and where it is thin, without judging "
               "study quality. It suits broad or emerging topics.",
}

# What the user wants the review for -> the review type that fits it
PURPOSES = {
    "effect": ("systematic_prisma", "you want to know whether something works or is linked to an outcome"),
    "map": ("scoping", "you want to map what research exists on a topic and where it is thin"),
    "overview": ("narrative", "you want an overview or background on a topic"),
}

EFFECT_INTENTS = {"causal_intervention", "associational", "comparative", "predictive"}


def recommend_review_type(purpose: Optional[str]) -> Optional[tuple]:
    """(review type, reason) for a stated purpose, or None if the purpose is not one of PURPOSES."""
    hit = PURPOSES.get((purpose or "").strip().lower())
    if not hit:
        return None
    review_type, why = hit
    return review_type, f"Recommended because {why}. {REVIEW_TYPE_REASONS[review_type]}"


def _role(concept: SearchConcept, is_main: bool, intent: str) -> str:
    if is_method_concept(concept.name):
        return "method"
    if intent in EFFECT_INTENTS:
        if is_main:
            return "intervention" if intent == "causal_intervention" else "exposure"
        return "outcome"
    return "topic"


def default_protocol(
    plan: QueryDecompositionPlan,
    question: str,
    review_type: str = "narrative",
    organization: str = "thematic",
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> ReviewProtocol:
    """The protocol the tool follows on its own: everything from the rule-based reading of the question."""
    u = plan.selection_understanding or plan.question_understanding
    concepts, n_main = question_concepts(u)
    intent = (u.primary_intent if u else "") or ""
    protocol = ReviewProtocol(
        question=question,
        review_type=review_type,
        review_type_reason=REVIEW_TYPE_REASONS.get(review_type, ""),
        organization=organization,
        breadth=(u.breadth if u and u.breadth in ("focused", "broad") else "focused"),
        domain=(u.primary_domain if u else ""),
        concepts=[
            ProtocolConcept(role=_role(c, i < n_main, intent), name=c.name, synonyms=list(c.synonyms), main=i < n_main)
            for i, c in enumerate(concepts)
        ],
        populations_and_places=[
            ProtocolConstraint(kind="place" if c.kind == "place" else "population", name=c.name,
                               spellings=list(c.spellings))
            for c in (u.concepts.required_constraints if u else []) if c.kind in ("place", "population")
        ],
        aspects=[ProtocolAspect(label=a.label, terms=list(a.terms), question=a.question)
                 for a in (u.aspects if u else []) if a.kind == "aspect"],
        eligibility=EligibilityCriteria(year_from=year_from, year_to=year_to),
    )
    if not protocol.concepts:
        protocol.notes.append("No searchable concept was found in the question; add at least one before searching.")
    return protocol


def is_default(protocol: ReviewProtocol) -> bool:
    """True when no part of the protocol was set by the user or taken from a suggestion."""
    parts = list(protocol.concepts) + list(protocol.populations_and_places) + list(protocol.aspects)
    return all(p.set_by == "default" for p in parts) and not protocol.eligibility.changed_by_user


def plan_from_protocol(protocol: ReviewProtocol, mode: str, domain_override=None) -> QueryDecompositionPlan:
    """
    Compiles the searches from a protocol, by the same rules as a run without one, and sets the reading of the
    question that selection uses. Comparisons are recorded but neither searched for nor required.
    """
    from services.query_utils import compile_deterministic_decomposition_plan, understand_research_question

    understanding = understand_research_question(protocol.question, domain_override)
    understanding.concepts.required_constraints = [
        RequiredConstraint(kind=c.kind, name=c.name,
                           spellings=[s.strip().lower() for s in (c.spellings or [c.name]) if s.strip()])
        for c in protocol.populations_and_places if c.name.strip()
    ]
    concepts = [SearchConcept(name=c.name.strip(), synonyms=[s.strip() for s in c.synonyms if s.strip()], main=c.main)
                for c in protocol.concepts if c.role != "comparison" and c.name.strip()]
    if concepts and not any(c.main for c in concepts):
        concepts[0].main = True
    concepts = [c for c in concepts if c.main] + [c for c in concepts if not c.main]
    aspects = [SearchAspect(label=a.label.strip(), terms=[t.strip() for t in a.terms if t.strip()],
                            question=a.question)
               for a in protocol.aspects if a.label.strip()]
    plan = compile_deterministic_decomposition_plan(
        protocol.question, mode=mode, domain_override=domain_override,
        model_plan={"concepts": concepts, "aspects": aspects, "breadth": protocol.breadth},
        understanding=understanding,
    )
    label = "default review protocol" if is_default(protocol) else "review protocol (user)"
    plan.question_understanding.planned_by = label
    plan.selection_understanding = plan.question_understanding.model_copy(deep=True)
    plan.selection_entities = [c.name for c in concepts]
    return plan


async def model_suggestions(protocol: ReviewProtocol, mode: str, config: ApiKeysConfig,
                            domain_override=None) -> List[ProtocolSuggestion]:
    """The model's extra synonyms, concepts and aspects, as suggestions the user may accept. [] without a model."""
    from services.query_utils import generate_multi_api_decomposition_plan

    planned = await generate_multi_api_decomposition_plan(protocol.question, mode, config,
                                                          domain_override=domain_override)
    u = planned.question_understanding
    if not u or not str(u.planned_by or "").startswith("model"):
        return []
    def key(name: str) -> str:
        # "fracture" and "fractures" are the same concept
        return re.sub(r"(?<=[a-z])(?:es|s)$", "", name.strip().lower())

    have = {key(c.name): c for c in protocol.concepts}
    out: List[ProtocolSuggestion] = []
    for mc in u.concepts.search_concepts:
        mine = have.get(key(mc.name))
        if mine is None:
            # The model named the same concept more briefly or more fully ("vitamin D" for "vitamin D
            # supplementation"): both start the same way. A model concept that is only a part of one of ours
            # ("supplementation") is a different concept, so its synonyms are not offered as synonyms of ours.
            m = mc.name.lower()
            mine = next((c for c in protocol.concepts
                         if c.name.lower().startswith(m + " ") or m.startswith(c.name.lower() + " ")), None)
        if mine is None:
            out.append(ProtocolSuggestion(kind="concept", value=mc.name, terms=list(mc.synonyms)))
            continue
        known = {s.lower() for s in mine.synonyms} | {mine.name.lower()}
        out += [ProtocolSuggestion(kind="synonym", concept=mine.name, value=s)
                for s in mc.synonyms if s.lower() not in known]
    have_aspects = {a.label.lower() for a in protocol.aspects}
    out += [ProtocolSuggestion(kind="aspect", value=a.label, terms=list(a.terms))
            for a in u.aspects if a.kind == "aspect" and a.label.lower() not in have_aspects]
    return out


async def propose_protocol(
    question: str,
    mode: str = "standard",
    review_framework: str = "narrative",
    purpose: Optional[str] = None,
    domain_override=None,
    suggest: bool = False,
    config: Optional[ApiKeysConfig] = None,
) -> ReviewProtocol:
    """The default protocol for a question, with a recommended review type when a purpose is given and, if asked,
    the model's suggestions alongside (not applied)."""
    from services.query_utils import compile_deterministic_decomposition_plan

    base = compile_deterministic_decomposition_plan(question, mode=mode, domain_override=domain_override)
    review_type = review_framework
    reason = None
    rec = recommend_review_type(purpose)
    if rec:
        review_type, reason = rec
    if mode != "deep_research" and review_type != "narrative":
        review_type = "narrative"
        reason = (reason + " " if reason else "") + ("A quick answer is always narrative; choose the guided review for a "
                                                      "systematic-style or scoping review.")
    protocol = default_protocol(base, question, review_type=review_type)
    if reason:
        protocol.review_type_reason = reason
    if suggest and config is not None:
        protocol.suggestions = await model_suggestions(protocol, mode, config, domain_override)
    return protocol
