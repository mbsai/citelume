"""
CiteLume study selection protocol: how the included studies (20 Standard / 60 Deep Research) are chosen from
everything the searches return. Every rule here is deterministic and reported back with the results, so the
choice of studies can be explained and reproduced. The plain-language version is docs/RANKING.md.

  Step 1  Merge duplicates (done by the caller).
  Step 2  Eligibility: remove records that cannot be used, each for a stated reason.
  Step 3  Score every eligible record out of 100 on five criteria (CRITERIA below).
  Step 4  Balanced selection: cover each aspect of the question, keep null results in view, limit narrative
          reviews and repeated author groups, then fill the remaining places by score.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from models import (
    PaperRecord,
    SearchAspect,
    SearchConcept,
    SelectionCriterion,
    SelectionReport,
    UniversalQuestionUnderstanding,
)
from services.question_parsing import (
    GENERIC_CONCEPT_WORDS,
    concept_blocks,
    concept_mentions,
    is_method_concept,
    meets_constraint,
)

PROTOCOL_VERSION = "citelume-selection/2.0"

# ---------------------------------------------------------------------------------------------------------------
# Step 3 criteria: points available for each (they add up to 100)
# ---------------------------------------------------------------------------------------------------------------
POINTS = {"relevance": 50.0, "evidence": 20.0, "impact": 15.0, "recency": 10.0, "agreement": 5.0}
RELEVANCE_CONCEPT_POINTS = 30.0   # the question's concepts appear in the title / abstract
RELEVANCE_TEXT_POINTS = 20.0      # wording overlap with the question, ranked against the other candidates

CRITERIA: List[SelectionCriterion] = [
    SelectionCriterion(
        key="relevance", label="Relevance to the question", max_points=POINTS["relevance"],
        rule=("Up to 30 points for the question's concepts (or their synonyms), and any place or population it "
              "names, appearing in the study: full marks for the title, 70% for two or more mentions in the "
              "abstract, 40% for one mention; the main concept counts double. Up to 20 points for overall wording "
              "overlap with the question (BM25), ranked against the other candidate studies."),
    ),
    SelectionCriterion(
        key="evidence", label="Strength of study type", max_points=POINTS["evidence"],
        rule=("Points by study type: systematic review, meta-analysis or guideline 20; randomised trial 18; cohort, "
              "case-control or other controlled study 14; cross-sectional, laboratory, computational or other "
              "primary study 10-11; narrative review 8 (16 when the question is a broad topic to survey); case "
              "report 5; commentary 2. Preprints that are not peer reviewed lose 3 points."),
    ),
    SelectionCriterion(
        key="impact", label="Citation impact", max_points=POINTS["impact"],
        rule=("Citations per year since publication, ranked against the other candidate studies. Studies from the "
              "last two calendar years, and studies with no citation data, get at least the middle value so new "
              "work is not penalised."),
    ),
    SelectionCriterion(
        key="recency", label="Recency", max_points=POINTS["recency"],
        rule="10 points for the last two years, then one point less for each year older (0 from 12 years).",
    ),
    SelectionCriterion(
        key="agreement", label="Found by several searches", max_points=POINTS["agreement"],
        rule="5 points if three or more independent searches or databases returned the study, 3 points for two.",
    ),
]

# Evidence points by study type. "empirical" applies to health, life and social sciences, where study design
# determines how far a result can be trusted; "technical" applies to computing, mathematics, physics, chemistry and
# engineering, where that hierarchy does not exist and primary work is not ranked by design.
STUDY_TYPE_LABELS: Dict[str, str] = {
    "evidence_synthesis": "Systematic review / meta-analysis / guideline",
    "rct": "Randomised controlled trial",
    "controlled_observational": "Cohort / case-control / controlled study",
    "descriptive_observational": "Cross-sectional / survey study",
    "experimental_lab": "Laboratory study (cell / animal / assay)",
    "computational": "Computational / modelling study",
    "theoretical": "Theoretical / formal study",
    "qualitative": "Qualitative study",
    "primary_other": "Primary research article",
    "review": "Narrative review",
    "case_report": "Case report / case series",
    "commentary": "Editorial / commentary / letter",
    # Not eligible for inclusion
    "correction": "Correction / retraction notice",
    "protocol": "Study protocol (no results)",
    "conference_abstract": "Meeting abstract",
}
EVIDENCE_POINTS: Dict[str, Dict[str, float]] = {
    "empirical": {
        "evidence_synthesis": 20, "rct": 18, "controlled_observational": 14, "descriptive_observational": 11,
        "experimental_lab": 11, "computational": 11, "theoretical": 10, "qualitative": 10, "primary_other": 10,
        "review": 8, "case_report": 5, "commentary": 2,
    },
    "technical": {
        "evidence_synthesis": 16, "rct": 14, "controlled_observational": 14, "descriptive_observational": 14,
        "experimental_lab": 14, "computational": 14, "theoretical": 14, "qualitative": 12, "primary_other": 14,
        "review": 10, "case_report": 6, "commentary": 2,
    },
}
REVIEW_POINTS_FOR_BROAD_TOPICS = 16.0
PREPRINT_PENALTY = 3.0
INELIGIBLE_TYPES = {"correction", "protocol", "conference_abstract"}
TECHNICAL_DOMAINS = {"computer_science_ai", "mathematics", "physics", "chemistry", "engineering_materials"}

# Step 4 limits
REVIEW_SHARE_CAP = {"focused": 0.30, "broad": 0.50}   # narrative reviews as a share of the included studies
MAX_PER_AUTHOR_YEAR = 2                               # same first author and year
MIN_CONSTRAINT_MATCHES = 5                            # apply place/population exclusion only if this many match
MIN_CONCEPT_POINTS_FOR_COVERAGE_SLOT = 12.0           # a study taking an aspect's reserved place must be on topic

SELECTION_RULES: List[str] = [
    "Coverage: each aspect of the question first receives its highest-scoring study (two in Deep Research), "
    "provided that study is on topic.",
    "Null results: for questions about an effect or association, up to two studies (four in Deep Research) that "
    "report null or contrary findings are kept in if any are eligible.",
    "Remaining places are filled in order of score.",
    "Narrative reviews are limited to 30% of the included studies (50% for broad topics), so primary evidence is "
    "not crowded out.",
    "No more than two studies with the same first author and year, to favour independent evidence.",
    "Ties are broken by newer publication year, then citations per year, then title.",
]

# ---------------------------------------------------------------------------------------------------------------
# Study type
# ---------------------------------------------------------------------------------------------------------------
_T = re.IGNORECASE
_RE_CORRECTION = re.compile(r"^\s*(?:author |publisher )?(?:correction|corrigendum|erratum|retraction|retracted|"
                            r"expression of concern|withdrawn)\b", _T)
# Meeting-abstract numbering at the start of a title ("0067 The effect of ...", "P123 ..."), but not a year
_RE_CONF_ABSTRACT = re.compile(r"^\s*(?:[A-Z]{1,4}[- ]?\d{2,5}|(?!(?:19|20)\d{2}\b)\d{3,5})[.:]?\s+[A-Z]")
_RE_PROTOCOL = re.compile(r"\b(?:study protocol|trial protocol|protocol for (?:a|an|the)|: (?:a )?protocol\b|"
                          r"protocol of (?:a|an|the))", _T)
_RE_SYNTHESIS = re.compile(r"\b(?:systematic review|meta-?analys[ie]s|umbrella review|scoping review|pooled analysis|"
                           r"network meta|cochrane review|clinical practice guideline|practice guideline|"
                           r"consensus statement|position statement|guidelines? (?:for|on)\b)", _T)
_RE_RCT = re.compile(r"\b(?:randomi[sz]ed|randomly assigned|randomly allocated|double-blind|placebo-controlled)\b", _T)
_RE_REVIEW_TITLE = re.compile(r"\b(?:review|overview|perspectives?|state of the art|a survey\b|survey (?:of|on)|"
                              r"recent advances|current (?:status|concepts|understanding)|update on|primer|"
                              r"progress in|insights? into|landscape)\b", _T)
_RE_REVIEW_TEXT = re.compile(r"\b(?:in this review|this review|we review|we summari[sz]e|this article reviews|"
                             r"we discuss recent|narrative review|literature review)\b", _T)
_RE_CASE = re.compile(r"\b(?:case reports?|case series|a case of|a rare case|report of (?:a|two|three) cases?)\b", _T)
_RE_CONTROLLED = re.compile(r"\b(?:cohort|case[- ]control|longitudinal|prospective(?:ly)?|retrospective(?:ly)?|"
                            r"follow-up|followed up|registry|mendelian randomi[sz]ation|hazard ratio|propensity|"
                            r"non-?randomi[sz]ed|quasi-experimental|difference-in-differences|natural experiment|"
                            r"clinical trial|phase (?:i{1,3}|[123])\b)", _T)
_RE_DESCRIPTIVE = re.compile(r"\b(?:cross[- ]sectional|prevalence|survey|questionnaire|community-based|"
                             r"descriptive study|ecological study)\b", _T)
_RE_QUAL = re.compile(r"\b(?:qualitative|semi-structured interviews?|focus groups?|thematic analysis|"
                      r"ethnograph\w+|grounded theory)\b", _T)
_RE_LAB = re.compile(r"\b(?:in vitro|in vivo|mice|murine|mouse|rats?|zebrafish|cell lines?|knockout|western blot|"
                     r"assays?|organoids?|xenograft|isolates?|strains?)\b", _T)
_RE_COMPUT = re.compile(r"\b(?:we propose|we present|we introduce|benchmark|neural network|deep learning|"
                        r"machine learning|algorithm|simulations?|density functional|molecular dynamics|"
                        r"monte carlo|finite element|dataset|model(?:l)?ing)\b", _T)
_RE_THEORY = re.compile(r"\b(?:theorem|lemma|we prove|proof of|conjecture|we derive|analytical solution)\b", _T)

_TYPE_FROM_SOURCE = [
    ("correction", {"retracted publication", "retraction of publication", "published erratum",
                    "expression of concern", "retraction notice"}),
    ("protocol", {"clinical trial protocol"}),
    ("evidence_synthesis", {"meta-analysis", "metaanalysis", "systematic review", "practice guideline", "guideline",
                            "consensus development conference"}),
    ("rct", {"randomized controlled trial", "controlled clinical trial", "pragmatic clinical trial",
             "equivalence trial", "clinical trial, phase iii", "clinical trial, phase iv"}),
    ("case_report", {"case reports", "casereport"}),
    ("commentary", {"editorial", "comment", "letter", "news", "lettersandcomments", "biography", "interview",
                    "newspaper article"}),
    ("review", {"review"}),
    ("controlled_observational", {"clinical trial", "clinicaltrial", "clinical trial, phase i",
                                  "clinical trial, phase ii", "observational study", "comparative study",
                                  "multicenter study", "twin study"}),
]


def classify_study_type(p: PaperRecord) -> str:
    """Study type key (see STUDY_TYPE_LABELS), from the source's publication types first, then the wording."""
    title = p.title or ""
    text = f"{title}. {(p.abstract or '')[:3000]}"
    types = {t.strip().lower() for t in (p.publication_types or [])}

    if _RE_CORRECTION.search(title) or types & _TYPE_FROM_SOURCE[0][1]:
        return "correction"
    if _RE_CONF_ABSTRACT.search(title):
        return "conference_abstract"
    if types & _TYPE_FROM_SOURCE[1][1] or (_RE_PROTOCOL.search(title) and not re.search(r"\bresults?\b", title, _T)):
        return "protocol"
    if types & _TYPE_FROM_SOURCE[2][1] or _RE_SYNTHESIS.search(title):
        return "evidence_synthesis"
    if types & _TYPE_FROM_SOURCE[3][1] or (_RE_RCT.search(text) and re.search(r"\btrial|controlled\b", text, _T)
                                           and not _RE_REVIEW_TITLE.search(title)):
        return "rct"
    if types & _TYPE_FROM_SOURCE[4][1] or _RE_CASE.search(title):
        return "case_report"
    if types & _TYPE_FROM_SOURCE[5][1] and len(p.abstract or "") < 600:
        return "commentary"
    if types & _TYPE_FROM_SOURCE[6][1] or _RE_REVIEW_TITLE.search(title) or _RE_REVIEW_TEXT.search(text):
        return "review"
    if _RE_THEORY.search(text):
        return "theoretical"
    if types & _TYPE_FROM_SOURCE[7][1] or _RE_CONTROLLED.search(text):
        return "controlled_observational"
    if _RE_QUAL.search(text):
        return "qualitative"
    if _RE_DESCRIPTIVE.search(text):
        return "descriptive_observational"
    if _RE_LAB.search(text):
        return "experimental_lab"
    if _RE_COMPUT.search(text):
        return "computational"
    return "primary_other"


def is_preprint(p: PaperRecord) -> bool:
    return p.version_status == "preprint" or (
        p.version_status == "unknown" and (p.source or "").lower() in ("arxiv", "biorxiv", "medrxiv"))


# ---------------------------------------------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------------------------------------------
_WORD = re.compile(r"[^\W_][\w-]*", re.UNICODE)
_BM25_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have", "had", "not", "but",
    "its", "their", "our", "can", "may", "into", "than", "then", "also", "such", "these", "those", "been", "being",
    "which", "what", "how", "does", "did", "who", "why", "when", "where", "about", "between", "among", "within",
    "using", "used", "use", "study", "studies", "results", "methods", "conclusion", "background", "objective",
    "of", "in", "on", "to", "a", "an", "is", "it", "as", "by", "at", "or", "be", "we", "you", "your", "me", "my",
    "please", "provide", "information", "tell", "give", "find", "papers", "research", "literature", "review",
}


def _tokens(text: str) -> List[str]:
    return [w for w in (m.group(0).lower() for m in _WORD.finditer(text or "")) if len(w) > 1 and w not in _BM25_STOP]


def bm25_scores(query_text: str, docs: Sequence[str], k1: float = 1.4, b: float = 0.75) -> List[float]:
    """Okapi BM25 of each document against the query (standard formula; no external index)."""
    q_terms = list(dict.fromkeys(_tokens(query_text)))
    tokenised = [_tokens(d) for d in docs]
    n = len(tokenised)
    if not n or not q_terms:
        return [0.0] * n
    avg_len = sum(len(t) for t in tokenised) / n or 1.0
    df = Counter(term for toks in tokenised for term in set(toks) if term in set(q_terms))
    scores = []
    for toks in tokenised:
        tf = Counter(toks)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks) / avg_len))
        scores.append(s)
    return scores


def percentile_ranks(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Share of the known values that are lower than or equal to each value (ties share the average rank)."""
    known = sorted(v for v in values if v is not None)
    if not known:
        return [None] * len(values)
    n = len(known)
    first: Dict[float, int] = {}
    last: Dict[float, int] = {}
    for i, v in enumerate(known):
        first.setdefault(v, i)
        last[v] = i
    return [None if v is None else ((first[v] + last[v]) / 2 + 1) / n for v in values]


def _concept_level(concept: SearchConcept, title: str, abstract: str) -> float:
    names = [concept.name] + list(concept.synonyms)
    if any(concept_mentions(n, title) for n in names):
        return 1.0
    hits = max((concept_mentions(n, abstract) for n in names), default=0)
    return 0.7 if hits >= 2 else (0.4 if hits == 1 else 0.0)


def question_concepts(understanding: Optional[UniversalQuestionUnderstanding]) -> Tuple[List[SearchConcept], int]:
    """
    The concepts a study is checked against, and how many of the leading ones count as the main topic
    (concepts the question lists side by side, e.g. "X and Y metabolism", are all main).
    """
    if not understanding:
        return [], 0
    c = understanding.concepts
    if c.search_concepts:
        concepts = list(c.search_concepts)
    else:
        phrases = [p for p in (c.concept_phrases or [])
                   if not is_method_concept(p) and any(w not in GENERIC_CONCEPT_WORDS for w in p.split())]
        concepts = [SearchConcept(name=p) for p in (phrases or list(c.concept_phrases or []))[:3]]
    if not concepts:
        return [], 0
    first_block = (concept_blocks(understanding.normalized_query, [x.name for x in concepts]) or [[concepts[0].name]])[0]
    return concepts, max(1, len(first_block))


_NULL_RESULT = re.compile(
    r"\b(?:no (?:significant|statistically significant|evidence of|association|effect|difference|benefit|"
    r"improvement|reduction)|not (?:significantly |statistically )?associated|did not (?:significantly )?"
    r"(?:improve|reduce|differ|increase|decrease|affect|alter)|was not (?:significantly )?(?:associated|different)|"
    r"were not (?:significantly )?(?:associated|different)|failed to (?:show|demonstrate|replicate|confirm)|"
    r"null (?:results?|findings?|effect)|non-?significant|contrary to|conflicting (?:results|evidence|findings)|"
    r"inconsistent (?:results|evidence|findings))\b", _T)


def reports_null_result(p: PaperRecord) -> bool:
    return bool(_NULL_RESULT.search(p.abstract or ""))


def _aspect_members(aspect: SearchAspect, p: PaperRecord) -> bool:
    if aspect.label in (p.retrieved_by or []):
        return True
    title, text = (p.title or "").lower(), f"{p.title} {p.abstract or ''}".lower()
    terms = [t.lower() for t in aspect.terms if t]
    return any(t in title for t in terms) or sum(1 for t in terms if t in text) >= 2


# ---------------------------------------------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------------------------------------------
def select_studies(
    unique_papers: List[PaperRecord],
    query: str,
    max_total: int,
    understanding: Optional[UniversalQuestionUnderstanding] = None,
    stats: Optional[Dict[str, int]] = None,
    records_found: Optional[int] = None,
) -> Tuple[List[PaperRecord], SelectionReport]:
    """Steps 2-4 on de-duplicated records. Returns the included studies (numbered in rank order) and the report."""
    now_year = datetime.now(timezone.utc).year
    breadth = understanding.breadth if understanding else "focused"
    domain = understanding.primary_domain if understanding else "biomedical_clinical"
    profile = "technical" if domain in TECHNICAL_DOMAINS else "empirical"
    intent = understanding.primary_intent if understanding else "descriptive"
    constraints = understanding.concepts.required_constraints if understanding else []
    concepts, n_main = question_concepts(understanding)
    aspects = [a for a in (understanding.aspects if understanding else []) if a.kind == "aspect"]

    report = SelectionReport(
        protocol_version=PROTOCOL_VERSION, target_count=max_total,
        records_found=records_found if records_found is not None else len(unique_papers),
        unique_records=len(unique_papers), criteria=CRITERIA, selection_rules=SELECTION_RULES,
        study_type_labels=STUDY_TYPE_LABELS,
    )
    report.duplicates_merged = max(0, report.records_found - report.unique_records)

    # ---- Step 2: eligibility -----------------------------------------------------------------------------------
    for p in unique_papers:
        p.study_type = classify_study_type(p)
    concept_levels: Dict[str, List[float]] = {
        p.id: [_concept_level(c, p.title or "", p.abstract or "") for c in concepts] for p in unique_papers
    }

    def misses_constraint(p: PaperRecord) -> List[str]:
        text = f"{p.title} {p.abstract or ''}"
        return [c.name for c in constraints if not meets_constraint(c.spellings, text)]

    def constraint_levels(p: PaperRecord) -> List[float]:
        # A place or population named in the title counts fully; a passing mention in the abstract counts less
        out = []
        for c in constraints:
            if meets_constraint(c.spellings, p.title or ""):
                out.append(1.0)
            else:
                n = meets_constraint(c.spellings, p.abstract or "")
                out.append(0.7 if n >= 2 else (0.4 if n == 1 else 0.0))
        return out

    def on_topic(p: PaperRecord) -> bool:
        levels = concept_levels[p.id]
        return not levels or any(l > 0 for l in levels[:n_main])

    for p in unique_papers:
        missing = misses_constraint(p)
        levels = concept_levels[p.id]
        if missing:
            p.relevance, p.relevance_note = "off_topic", f"Title/abstract does not mention {', '.join(missing)}"
        elif not on_topic(p):
            p.relevance = "off_topic"
            p.relevance_note = f"Not about {' / '.join(c.name for c in concepts[:n_main])}"
        else:
            main_best = max(levels[:n_main], default=1.0)
            others = levels[n_main:]
            if others:
                full = any(l > 0 for l in others)
            else:
                full = main_best >= 0.7 or not (p.abstract or "").strip()
                # A metadata-only record is judged on its title alone
                if not (p.abstract or "").strip() and main_best < 1.0:
                    full = False
            p.relevance = "addresses" if full else "partly"
            covered = [c.name for c, l in zip(concepts, levels) if l > 0]
            p.relevance_note = (f"Matches {', '.join(c.name for c in constraints)}; " if constraints else "") + (
                f"covers {', '.join(covered)}" if covered else "on topic")

    matching = [p for p in unique_papers if p.relevance != "off_topic"]
    excluded: Dict[str, int] = {}
    pool = list(unique_papers)
    apply_screen = len(matching) >= MIN_CONSTRAINT_MATCHES
    if stats is not None:
        stats["unique_records"] = len(unique_papers)
        stats["constraint_excluded"] = (len(unique_papers) - len(matching)) if apply_screen else 0
    if apply_screen:
        n_constraint = sum(1 for p in pool if misses_constraint(p))
        if n_constraint:
            excluded["Outside the place or population asked about"] = n_constraint
        n_topic = len(pool) - len(matching) - n_constraint
        if n_topic > 0:
            excluded["Not about the topic of the question"] = n_topic
        pool = matching
    elif len(matching) < len(unique_papers):
        report.notes.append(
            f"Only {len(matching)} records matched the topic and constraints, so the others were kept but flagged "
            "as off topic and left out of the synthesis.")

    typed = [p for p in pool if p.study_type not in INELIGIBLE_TYPES]
    if len(typed) < len(pool):
        excluded["Not a research report (correction, retraction, protocol or meeting abstract)"] = len(pool) - len(typed)
    pool = typed
    with_abstract = [p for p in pool if len((p.abstract or "").strip()) >= 20]
    if len(with_abstract) >= max_total and len(with_abstract) < len(pool):
        excluded["No abstract available to assess"] = len(pool) - len(with_abstract)
        pool = with_abstract
    report.excluded = excluded
    report.eligible = len(pool)
    if not pool:
        return [], report

    # ---- Step 3: score -----------------------------------------------------------------------------------------
    query_text = " ".join([query] + [c.name for c in concepts] + [s for c in concepts for s in c.synonyms])
    bm25 = bm25_scores(query_text, [f"{p.title}. {p.title}. {p.abstract or ''}" for p in pool])
    text_pct = percentile_ranks(bm25 if any(bm25) else [None] * len(pool))

    def cites_per_year(p: PaperRecord) -> Optional[float]:
        if not p.citation_source and p.citation_count <= 0:
            return None
        age = max(1, now_year - (p.year or now_year) + 1)
        return p.citation_count / age

    impact_pct = percentile_ranks([cites_per_year(p) for p in pool])
    known_citations = sum(1 for p in pool if cites_per_year(p) is not None)
    sources_used = Counter(p.citation_source for p in pool if p.citation_source)
    report.citation_data = (
        f"Citation counts were available for {known_citations} of {len(pool)} eligible studies"
        + (f" ({', '.join(f'{k}: {v}' for k, v in sources_used.most_common())})" if sources_used else "") + ".")

    weights = [2.0 if i < n_main else 1.0 for i in range(len(concepts))] + [1.0] * len(constraints)
    scores: Dict[str, float] = {}
    for i, p in enumerate(pool):
        levels = concept_levels[p.id] + constraint_levels(p)
        concept_part = (sum(w * l for w, l in zip(weights, levels)) / sum(weights)) if levels else 0.5
        text_part = text_pct[i] if text_pct[i] is not None else 0.5
        relevance = RELEVANCE_CONCEPT_POINTS * concept_part + RELEVANCE_TEXT_POINTS * text_part

        evidence = EVIDENCE_POINTS[profile].get(p.study_type, 10.0)
        if p.study_type == "review" and breadth == "broad":
            evidence = REVIEW_POINTS_FOR_BROAD_TOPICS
        if is_preprint(p):
            evidence = max(0.0, evidence - PREPRINT_PENALTY)

        pct = impact_pct[i]
        recent = p.year is not None and now_year - p.year <= 1
        if pct is None or (recent and pct < 0.5):
            pct = 0.5
        impact = POINTS["impact"] * pct

        if p.year is None:
            recency = POINTS["recency"] / 2
        else:
            recency = max(0.0, min(POINTS["recency"], POINTS["recency"] - max(0, (now_year - p.year) - 2)))

        found_by = len(set(p.retrieved_by or [])) + max(0, len(set(p.all_sources or [])) - 1)
        agreement = 5.0 if found_by >= 3 else (3.0 if found_by == 2 else 0.0)

        total = relevance + evidence + impact + recency + agreement
        scores[p.id] = total
        p.relevance_score = round(total / 100.0, 4)
        p.selection_breakdown = {
            "relevance": round(relevance, 1), "evidence": round(evidence, 1), "impact": round(impact, 1),
            "recency": round(recency, 1), "agreement": round(agreement, 1), "total": round(total, 1),
            "concept_match": round(RELEVANCE_CONCEPT_POINTS * concept_part, 1),
        }

    def sort_key(p: PaperRecord):
        return (p.relevance != "off_topic", round(scores[p.id], 3), p.year or 0, cites_per_year(p) or 0.0,
                [-ord(ch) for ch in (p.title or "")[:40].lower()])

    ranked = sorted(pool, key=sort_key, reverse=True)
    for rank, p in enumerate(ranked, start=1):
        p.selection_rank = rank
        p.matched_subquestion_ids = [f"SQ{j + 2}" for j, a in enumerate(aspects) if _aspect_members(a, p)] or ["SQ1"]

    # ---- Step 4: balanced selection ----------------------------------------------------------------------------
    chosen: List[PaperRecord] = []
    chosen_ids: set = set()
    author_year: Counter = Counter()
    review_cap = max(1, int(round(REVIEW_SHARE_CAP[breadth] * max_total)))

    def key_author_year(p: PaperRecord) -> str:
        return f"{(p.authors[0].split()[-1].lower() if p.authors else '?')}_{p.year or 0}"

    def take(p: PaperRecord, reason: str) -> None:
        chosen.append(p)
        chosen_ids.add(p.id)
        author_year[key_author_year(p)] += 1
        p.selection_reason = reason

    def allowed(p: PaperRecord, enforce_caps: bool = True) -> bool:
        if p.id in chosen_ids:
            return False
        if not enforce_caps:
            return True
        if p.authors and author_year[key_author_year(p)] >= MAX_PER_AUTHOR_YEAR:
            return False
        if p.study_type == "review" and sum(1 for c in chosen if c.study_type == "review") >= review_cap:
            return False
        return True

    per_aspect = 1 if max_total <= 20 else 2
    if len(ranked) > max_total:
        for a in aspects:
            got = 0
            for p in ranked:
                if got >= per_aspect or len(chosen) >= max_total:
                    break
                if (allowed(p) and p.relevance != "off_topic" and _aspect_members(a, p)
                        and p.selection_breakdown["concept_match"] >= MIN_CONCEPT_POINTS_FOR_COVERAGE_SLOT):
                    take(p, f"Best match for the aspect: {a.label}")
                    got += 1
        if breadth == "focused" and intent in ("causal_intervention", "associational", "comparative", "predictive"):
            want = 2 if max_total <= 20 else 4
            have = sum(1 for p in chosen if reports_null_result(p))
            for p in ranked:
                if have >= want or len(chosen) >= max_total:
                    break
                if allowed(p) and p.relevance == "addresses" and reports_null_result(p):
                    take(p, "Kept as a null or contrary result")
                    have += 1
    for enforce in (True, False):
        for p in ranked:
            if len(chosen) >= max_total:
                break
            if allowed(p, enforce):
                take(p, "Ranked by score" if enforce else "Ranked by score (limits relaxed to fill the set)")

    chosen.sort(key=sort_key, reverse=True)
    for i, p in enumerate(chosen, start=1):
        p.citation_index = i

    report.included = len(chosen)
    report.included_by_study_type = dict(Counter(STUDY_TYPE_LABELS.get(p.study_type, p.study_type) for p in chosen).most_common())
    report.included_by_reason = dict(Counter(re.sub(r":.*$", "", p.selection_reason) for p in chosen).most_common())
    report.included_by_aspect = {a.label: sum(1 for p in chosen if _aspect_members(a, p)) for a in aspects}
    if len(chosen) < max_total:
        report.notes.append(f"Only {len(chosen)} eligible studies were found, fewer than the {max_total} allowed.")
    return chosen, report
