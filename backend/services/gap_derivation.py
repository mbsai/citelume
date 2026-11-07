"""
Evidence-derived knowledge gaps: what the retrieved studies themselves show or say is unresolved.

Gaps come from three places, all specific to the query's own corpus:
  1. Open questions the authors state ("further prospective studies are needed to ...", "... remains unclear").
     These are read from the abstract and, when available, the full text's discussion / limitations / conclusion.
  2. The design profile of the included studies (e.g. only cross-sectional data, only cell/animal models,
     prediction models with no external validation, an intervention question with no randomised evidence).
  3. Cells of the model-built coverage grid (handled in gap_matrix_engine).

Nothing here uses fixed per-domain wording that could appear unchanged for an unrelated question.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from models import PaperRecord

# ---------------------------------------------------------------------------
# Study design families
# ---------------------------------------------------------------------------

DESIGN_LABELS: Dict[str, str] = {
    "review": "review / meta-analysis",
    "theory": "formal / theoretical analysis",
    "trial": "randomised or controlled trial",
    "prediction_model": "prediction / machine-learning model",
    "case_control": "case-control study",
    "cohort": "cohort / longitudinal study",
    "cross_sectional": "cross-sectional / prevalence study",
    "qualitative": "qualitative study",
    "lab": "laboratory (cell / animal / assay) study",
    "computational": "computational / simulation study",
    "other": "other / unclassified design",
}

_REVIEW_RE = re.compile(r"\b(systematic review|meta-analys[ie]s|scoping review|narrative review|umbrella review|"
                        r"literature review|pooled analysis)\b", re.I)
_DESIGN_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("theory", re.compile(r"\b(theorem|lemma|we prove|proof of|conjecture)\b", re.I)),
    ("trial", re.compile(r"\b(randomi[sz]ed|clinical trial|placebo|double-blind|phase (?:i{1,3}|[123])\b)", re.I)),
    ("prediction_model", re.compile(
        r"\b(auroc|auc\b|c-statistic|prediction model|predictive model|machine learning|deep learning|neural network|"
        r"transformer|classifier|cross-validation|benchmark|diagnostic accuracy|sensitivity and specificity|risk score)",
        re.I)),
    ("case_control", re.compile(r"\bcase[- ]control\b", re.I)),
    ("cross_sectional", re.compile(r"\bcross[- ]sectional\b", re.I)),
    ("cohort", re.compile(r"\b(cohort|prospective(?:ly)?|longitudinal|follow-up|followed up|incidence|hazard ratio|"
                          r"registry)\b", re.I)),
    ("cross_sectional", re.compile(r"\b(prevalence|survey|questionnaire|community-based|"
                                   r"population-based screening)\b", re.I)),
    ("qualitative", re.compile(r"\b(qualitative|semi-structured interviews?|focus groups?|thematic analysis)\b", re.I)),
    ("lab", re.compile(r"\b(in vitro|in vivo|mice|murine|mouse model|rats?|cell lines?|fibroblasts?|knockout|"
                       r"western blot|assays?|organoids?|xenograft)\b", re.I)),
    ("computational", re.compile(r"\b(simulations?|density functional|dft|molecular dynamics|monte carlo|"
                                 r"finite element|ab initio)\b", re.I)),
]


def design_family(p: PaperRecord) -> str:
    """Coarse design family of a study, read from its title, extracted design and abstract."""
    title = p.title or ""
    if _REVIEW_RE.search(title):
        return "review"
    elicit = p.elicit
    # Sentences about what future studies should do describe designs the study did NOT use
    abstract = " ".join(s for s in _sentences((p.abstract or "")[:2500]) if not _OPEN_QUESTION_RE.search(s))
    text = f"{title}. {elicit.study_design if elicit else ''}. {elicit.methodology if elicit else ''}. {abstract}"
    if re.search(r"\bwe (conducted|performed|undertook) an? (systematic review|meta-analysis|scoping review)", text, re.I):
        return "review"
    for family, pattern in _DESIGN_PATTERNS:
        if pattern.search(text):
            return family
    return "other"


# ---------------------------------------------------------------------------
# Author-stated open questions
# ---------------------------------------------------------------------------

_OPEN_QUESTION_RE = re.compile(
    r"\b(?:further|future|more|additional|larger|longitudinal|prospective|well-designed|multicent(?:er|re)|"
    r"randomi[sz]ed|high-quality)\s+(?:[a-z-]+\s+){0,3}(?:research|studies|study|work|trials?|investigations?|data|"
    r"evidence|validation)\b[^.;]{0,60}?\b(?:is|are|will be)?\s*(?:needed|required|warranted|necessary)\b"
    r"|\bfuture (?:research|studies|work) should\b"
    r"|\bremains?\s+(?:largely\s+|still\s+|poorly\s+)?(?:unclear|unknown|uncertain|understudied|unexplored|"
    r"poorly understood|to be (?:determined|elucidated|established|clarified)|an open (?:problem|question)|open|"
    r"controversial|elusive|unresolved)\b"
    r"|\b(?:little|less) is known\b|\b(?:limited|scarce|sparse|insufficient) (?:data|evidence|information)\b"
    r"|\bhas not (?:yet )?been (?:studied|investigated|explored|established|evaluated|determined)\b"
    r"|\b(?:were|was) not (?:evaluated|assessed|examined|investigated|measured)\b"
    r"|\bnot yet (?:known|established|clear|understood)\b|\bopen (?:problem|question|conjecture)s?\b"
    r"|\bknowledge gap\b|\bwarrants? further\b"
    # Formal results that break outside their assumptions leave the general case open
    r"|\bcounterexamples?\b|\b(?:bounds?|estimates?|results?|theorems?)\s+(?:fails?|do not hold|does not hold)\b",
    re.I,
)
# Clauses that are about the study's own analysis plan rather than the field ("data were not available")
_NOT_A_FIELD_GAP_RE = re.compile(
    r"\b(?:informed consent|ethic|funding|conflict of interest|this article|copyright)\b"
    # the study's own scope ("we did not include ...") is a limitation of that study, not an open question
    r"|\bwe (?:did|could) not\b|\bour (?:study|analysis|review|sample|data)\b"
    r"|\bthis (?:study|review|analysis|trial|work) (?:did|does|could|was) not\b"
    r"|\bnot (?:examined|investigated|assessed|evaluated|tested|measured|addressed) (?:here|in this)\b"
    r"|\b(?:here|in this (?:study|work|paper)),? we\b", re.I)
_FULLTEXT_GAP_SECTIONS = ("limitations", "discussion", "conclusion")

_STOP = {
    "the", "and", "for", "with", "that", "this", "these", "those", "from", "into", "their", "there", "which", "while",
    "were", "was", "are", "is", "been", "have", "has", "had", "not", "but", "also", "such", "more", "further",
    "future", "studies", "study", "research", "needed", "required", "warranted", "necessary", "remains", "remain",
    "unclear", "unknown", "should", "would", "could", "may", "might", "will", "our", "its", "than", "between",
    "among", "within", "about", "however", "including", "well", "need", "data", "evidence", "work", "larger",
    "additional", "investigate", "investigation", "evaluate", "determine", "explore", "understand", "role",
}


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9-]{2,}", (text or "").lower()) if w not in _STOP]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", re.sub(r"\s+", " ", text or "")) if s.strip()]


@dataclass
class OpenQuestion:
    sentence: str
    citation_indices: List[int] = field(default_factory=list)
    from_full_text: bool = False
    score: float = 0.0


def find_author_open_questions(
    papers: Sequence[PaperRecord],
    query: str,
    max_questions: int = 3,
) -> List[OpenQuestion]:
    """
    Sentences in which study authors say something is still unknown or needs further study, ranked by how
    closely they relate to the query, with near-duplicates across studies merged (keeping all citations).
    """
    q_words = set(_words(query))
    candidates: List[OpenQuestion] = []
    for p in papers:
        if p.citation_index is None:
            continue
        sources: List[Tuple[str, bool]] = [(p.abstract or "", False)]
        sections = p.full_text_sections or {}
        sources += [(sections.get(name, ""), True) for name in _FULLTEXT_GAP_SECTIONS]
        best: Optional[OpenQuestion] = None
        for text, is_full in sources:
            for s in _sentences(text):
                if not (40 <= len(s) <= 420) or not _OPEN_QUESTION_RE.search(s) or _NOT_A_FIELD_GAP_RE.search(s):
                    continue
                overlap = len(q_words & set(_words(s)))
                score = overlap + (0.5 if not is_full else 0.0) + min(len(_words(s)), 25) / 50
                if best is None or score > best.score:
                    best = OpenQuestion(s.rstrip(), [p.citation_index], is_full, score)
        if best is not None:
            candidates.append(best)

    candidates.sort(key=lambda c: -c.score)
    merged: List[OpenQuestion] = []
    for c in candidates:
        cw = set(_words(c.sentence))
        for m in merged:
            mw = set(_words(m.sentence))
            if cw and mw and len(cw & mw) / len(cw | mw) >= 0.4:
                m.citation_indices.extend(i for i in c.citation_indices if i not in m.citation_indices)
                break
        else:
            merged.append(c)
    return merged[:max_questions]


_LEAD_IN_RE = re.compile(
    r"^.*?\b(?:needed|required|warranted|necessary)\s+(?:in order\s+)?(?:to|for|on|regarding|in)\s+", re.I)
_REMAINS_RE = re.compile(
    r"^(?:.*?(?:,|;|\bwhile|\bwhereas|\bbut|\balthough|\bhowever)\s+)?(.{8,}?)\s+(remains?\s+(?:largely\s+|still\s+|poorly\s+)?"
    r"(?:unclear|unknown|uncertain|understudied|unexplored|poorly understood|to be \w+|an open (?:problem|question)|open|"
    r"controversial|elusive|unresolved))", re.I)


def open_question_title(sentence: str, max_words: int = 14) -> str:
    """Short readable title for an author-stated open question."""
    s = re.sub(r"\[[\d,\s–-]+\]|\(\s*\d+\s*\)", "", sentence).strip().rstrip(".")
    # Section labels and discourse markers are not part of the question
    s = re.sub(r"^(?:[A-Z][A-Z ]{2,20}:\s*|(?:background|conclusions?|objectives?|aims?|results)\s*:\s*)", "", s, flags=re.I)
    s = re.sub(r"^(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|lastly|moreover|furthermore|in addition|"
               r"additionally|importantly|notably|however|thus|therefore|overall|together|taken together)\s*,\s*",
               "", s, flags=re.I)
    m_need = re.match(r"^(?:further|more|additional|future|larger|well-designed|prospective|longitudinal)\s+"
                      r"(?:[\w-]+\s+){0,2}?(?:research|studies|study|investigations?|work|trials?|data)\s+"
                      r"(?:on|of|into|in|about|regarding|to clarify|to determine|to establish)\s+(.{8,}?)\s+"
                      r"(?:is|are|will be)\s+(?:still\s+)?(?:needed|required|warranted|necessary)", s, re.I)
    if m_need:
        s = "Needed: further study of " + m_need.group(1)
        words = s.split()
        return " ".join(words[:max_words]).rstrip(",;:") + (" …" if len(words) > max_words else "")
    m = _REMAINS_RE.match(s)
    if m:
        subject = re.sub(r"^(?:and|but|or|yet|so|while|whereas|although)\s+", "", m.group(1).strip(), flags=re.I)
        subject = re.sub(r"^(?:its|their|his|her)\s+", "", subject, flags=re.I)
        title = f"{subject} {m.group(2).strip()}"
    else:
        lead = _LEAD_IN_RE.match(s)
        title = ("Needed: " + s[lead.end():]) if lead and len(s) - lead.end() > 15 else s
    title = re.sub(r"^(?:the|a|an|these|this)\s+", "", title.strip(), flags=re.I)
    title = re.sub(r"^Needed: (?:the|a|an)\s+", "Needed: ", title)
    words = title.split()
    if len(words) > max_words:
        title = " ".join(words[:max_words]).rstrip(",;:") + " …"
    return title[:1].upper() + title[1:] if title else "Open question raised by the study authors"


def open_question_category(sentence: str, module_key: str) -> str:
    s = sentence.lower()
    if module_key == "mathematics" or re.search(r"\b(conjecture|theorem|open problem)\b", s):
        return "Theoretical or formal unresolved problems"
    if re.search(r"\bmechanis|pathway|pathogen|aetiolog|etiolog|signall?ing|causal role|\bhow\b.*\b(drives?|causes?|interacts?|regulates?|mediates?|contributes?)\b", s):
        return "Mechanistic uncertainty"
    if re.search(r"\bvalidat|replicat|external|independent cohort|generaliz|generalis", s):
        return "Validation deficiencies"
    if re.search(r"\b(population|rural|urban|region|ethnic|children|women|elderly|older adults|setting|countr|"
                 r"low-income|community)\b", s):
        return "Population or contextual limitations"
    if re.search(r"\b(clinical practice|translat|implementation|patients?\b.*\btreat|therap)", s):
        return "Translational deficiencies"
    if re.search(r"\b(longitudinal|prospective|long-term|larger|sample size|randomi[sz]ed|controlled|"
                 r"methodolog|confound|bias)\b", s):
        return "Methodological limitations"
    if re.search(r"\b(standardi[sz]|definition|measure|criteria|protocol|harmoni[sz])", s):
        return "Measurement/standardization limitations"
    if re.search(r"\b(conflicting|inconsistent|controversial|contradictory)\b", s):
        return "Inconsistent findings"
    return "Evidence absence"


def key_terms(text: str, n: int = 4) -> List[str]:
    """Most informative words of a sentence, in order of appearance, for verification searches."""
    seen: List[str] = []
    for w in _words(text):
        if len(w) > 3 and w not in seen:
            seen.append(w)
    return seen[:n]


# ---------------------------------------------------------------------------
# Design-profile gaps
# ---------------------------------------------------------------------------

_EXTERNAL_VALIDATION_RE = re.compile(
    r"\b(external(?:ly)? validat\w*|independent (?:test|validation) (?:set|cohort)|prospective(?:ly)? validat\w*|"
    r"multi-?cent(?:er|re) validat\w*|temporal validation|out-of-distribution test)", re.I)
_SINGLE_SITE_RE = re.compile(
    r"\b(single[- ]cent(?:er|re)|single[- ]institution|tertiary care|tertiary hospital|hospital-based|"
    r"one hospital|a teaching hospital|outpatient clinic of)\b", re.I)
_INTERVENTION_QUESTION_RE = re.compile(
    r"\b(effect(?:iveness)?|efficacy|treatments?|therap(?:y|ies)|interventions?|drugs?|reduce|improve|prevent(?:ion)?|"
    r"management|outcomes? of)\b", re.I)


@dataclass
class DesignGap:
    kind: str
    category: str
    title: str
    statement: str
    current_knowledge: str
    unresolved_issue: str
    why_it_matters: str
    evidence_needed: str
    recommended_approach: str
    study_indices: List[int]
    search_terms: List[str]
    origin: str = "B_limitations_in_existing_evidence"


def _cite(idxs: Sequence[int], limit: int = 8) -> str:
    shown = ", ".join(f"[{i}]" for i in list(idxs)[:limit])
    return shown + (f" and {len(idxs) - limit} more" if len(idxs) > limit else "")


def design_profile(papers: Sequence[PaperRecord]) -> Dict[str, List[int]]:
    profile: Dict[str, List[int]] = {}
    for p in papers:
        if p.citation_index is not None:
            profile.setdefault(design_family(p), []).append(p.citation_index)
    return profile


def design_profile_summary(profile: Dict[str, List[int]]) -> str:
    counts = Counter({k: len(v) for k, v in profile.items()})
    return "; ".join(f"{DESIGN_LABELS.get(k, k)} × {n}" for k, n in counts.most_common())


def derive_design_gaps(
    papers: Sequence[PaperRecord],
    query: str,
    module_key: str,
) -> List[DesignGap]:
    """Gaps that follow from which kinds of study the corpus contains and lacks."""
    if module_key == "mathematics" or not papers:
        return []
    profile = design_profile(papers)
    by_idx = {p.citation_index: p for p in papers}
    primary = [i for fam, idxs in profile.items() if fam not in ("review",) for i in idxs]
    n_primary = len(primary)
    if n_primary == 0:
        return []
    summary = design_profile_summary(profile)
    gaps: List[DesignGap] = []

    def text_of(i: int) -> str:
        p = by_idx[i]
        sections = p.full_text_sections or {}
        return f"{p.title} {p.abstract or ''} {sections.get('methods', '')} {sections.get('results', '')}"

    # Prediction / ML models without external validation
    models = profile.get("prediction_model", [])
    unvalidated = [i for i in models if not _EXTERNAL_VALIDATION_RE.search(text_of(i))]
    if unvalidated:
        lab_context = module_key in ("molecular_biology_omics",) or bool(profile.get("lab"))
        gaps.append(DesignGap(
            kind="external_validation",
            category="Validation deficiencies",
            title="Prediction models lack external validation",
            statement=(f"{len(unvalidated)} of {len(models)} model-development studies ({_cite(unvalidated)}) report "
                       "performance only on their own development data, with no independent external test."),
            current_knowledge="",  # filled by the caller from the extracted results
            unresolved_issue=("Whether the reported performance holds in independent cohorts or datasets"
                              + (", and in wet-lab or in vivo assays" if lab_context else "") + "."),
            why_it_matters=("Performance measured on the development data (internal splits or cross-validation) is "
                            "usually optimistic; without an external test it is unknown how well these models generalise."),
            evidence_needed="Evaluation of the published models on independent, held-out cohorts or datasets.",
            recommended_approach="External validation of the existing models on a new population or dataset, "
                                 "with leakage controls and reported calibration.",
            study_indices=unvalidated,
            search_terms=["external validation"],
            origin="D_insufficient_validation",
        ))

    # Laboratory-only evidence for a question about people
    lab = profile.get("lab", [])
    human = profile.get("trial", []) + profile.get("cohort", []) + profile.get("cross_sectional", []) + \
        profile.get("case_control", [])
    if lab and len(lab) >= max(2, n_primary // 2) and not human and module_key in (
            "biomedical_clinical", "molecular_biology_omics", "universal_fallback"):
        gaps.append(DesignGap(
            kind="translation",
            category="Translational deficiencies",
            title="Findings rest on cell or animal models",
            statement=(f"{len(lab)} of {n_primary} primary studies ({_cite(lab)}) are laboratory studies, and none "
                       "of the included studies tests these findings in patients."),
            current_knowledge=f"Design profile of the included studies: {summary}.",
            unresolved_issue="Whether the laboratory findings hold in human tissue, patient cohorts or clinical trials.",
            why_it_matters="Effects seen in cell lines or animal models often fail to translate to people.",
            evidence_needed="Studies in patient-derived material or clinical cohorts that test the same targets.",
            recommended_approach="Validate the key targets in patient samples, then in an observational cohort or early-phase trial.",
            study_indices=lab,
            search_terms=["patients", "clinical"],
        ))

    # Only cross-sectional evidence: no data over time
    xs = profile.get("cross_sectional", []) + profile.get("case_control", [])
    cohorts = profile.get("cohort", []) + profile.get("trial", [])
    if len(xs) >= 2 and len(xs) >= 0.6 * n_primary and not cohorts:
        gaps.append(DesignGap(
            kind="no_longitudinal",
            category="Methodological limitations",
            title="No longitudinal data: evidence is cross-sectional",
            statement=(f"{len(xs)} of {n_primary} primary studies ({_cite(xs)}) are cross-sectional or case-control, "
                       "and none follows people over time."),
            current_knowledge=f"Design profile of the included studies: {summary}.",
            unresolved_issue="Incidence, progression, outcomes and the time order of exposure and disease cannot be "
                             "established from snapshot data.",
            why_it_matters="Cross-sectional associations cannot show what comes first, so they cannot separate cause "
                           "from consequence or estimate risk over time.",
            evidence_needed="Prospective cohort or registry studies with follow-up.",
            recommended_approach="A prospective cohort with repeated measurements, or linkage of existing registries.",
            study_indices=xs,
            search_terms=["cohort", "incidence", "longitudinal"],
        ))

    # An intervention / effect question answered without randomised evidence
    trials = profile.get("trial", [])
    if _INTERVENTION_QUESTION_RE.search(query or "") and not trials and n_primary >= 3 and \
            module_key in ("biomedical_clinical", "social_sciences_humanities", "universal_fallback"):
        gaps.append(DesignGap(
            kind="no_trials",
            category="Evidence absence",
            title="No randomised or controlled evidence on effects",
            statement=f"None of the {n_primary} primary studies is a randomised or controlled trial.",
            current_knowledge=f"Design profile of the included studies: {summary}.",
            unresolved_issue="Whether the interventions or exposures studied actually cause the reported outcomes.",
            why_it_matters="Without a control group, observed differences may reflect confounding or selection rather "
                           "than an effect.",
            evidence_needed="Randomised or well-controlled comparative studies.",
            recommended_approach="A randomised controlled trial, or a quasi-experimental design where randomisation is not feasible.",
            study_indices=primary[:8],
            search_terms=["randomized", "trial"],
            origin="A_missing_direct_evidence",
        ))

    # Mostly single-centre / hospital samples
    single = [i for i in primary if _SINGLE_SITE_RE.search(text_of(i))]
    if len(single) >= 2 and len(single) >= 0.5 * n_primary:
        gaps.append(DesignGap(
            kind="single_site",
            category="Population or contextual limitations",
            title="Samples are mostly single-centre or hospital-based",
            statement=(f"{len(single)} of {n_primary} primary studies ({_cite(single)}) recruited from a single centre "
                       "or hospital."),
            current_knowledge=f"Design profile of the included studies: {summary}.",
            unresolved_issue="How far the findings apply to the wider community or to other regions and settings.",
            why_it_matters="Hospital and single-centre samples over-represent more severe or more accessible cases.",
            evidence_needed="Community-based or multi-centre studies with representative sampling.",
            recommended_approach="A multi-centre or population-based study with probability sampling.",
            study_indices=single,
            search_terms=["community-based", "multicentre"],
            origin="E_unexplored_relevant_conditions",
        ))
    return gaps
