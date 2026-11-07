"""
Turns a natural-language research question into search building blocks:

* concept phrases - consecutive content words kept together ("coronary artery disease", not three loose words);
* required constraints - places and populations the question is restricted to ("in India", "in children"),
  each with the spellings a paper might use (India / Indian / Indians);
* question-framing words ("what is the state of", "status", "landscape") dropped from search terms.

Constraints are applied as mandatory clauses in every source query and checked in titles and abstracts during
ranking, so a question about one country does not return studies from others.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Words that frame a question rather than name what it is about
QUESTION_FRAMING_WORDS = {
    "state", "status", "situation", "scenario", "landscape", "picture", "overall", "general", "known", "regarding",
    "concerning", "around", "trends", "trend", "today", "nowadays", "currently", "scope", "extent", "role",
    "evidence", "context", "aspects", "aspect", "things", "thing", "topic", "area", "field", "lot", "kind", "type",
    "types", "level", "levels",
    # connectives not in the general stop-word list
    "among", "amongst", "across", "within", "versus", "vs", "towards", "toward", "among", "per",
    # request vocabulary in conversational questions ("I want to do an academic grade study on the following topic")
    "academic", "grade", "rigorous", "comprehensive", "detailed", "thorough", "following", "topic", "topics",
    "subject", "plan", "planning", "planned", "involve", "involves", "involving", "include", "includes", "including",
    "interested", "interest", "aim", "aims", "goal", "goals", "well", "similar", "similarity", "similarities",
    "pattern", "patterns", "compare", "compared", "comparing", "comparison", "differ", "differs", "difference",
    "differences", "multiple", "other", "others", "whether", "way", "ways", "learn", "focus", "focusing",
    "regard", "respect", "respective", "able", "possible", "possibly", "basically", "etc",
    # linking verbs that join two concepts ("systemic sclerosis drives fibrosis")
    "drive", "drives", "driving", "driven", "cause", "causes", "causing", "caused", "lead", "leads", "leading",
    "play", "plays", "playing", "contribute", "contributes", "contributing", "mediate", "mediates", "mediated",
    "regulate", "regulates", "regulated", "underlie", "underlies", "underlying", "result", "results", "induce",
    "induces", "induced", "promote", "promotes", "influence", "influences", "relate", "related", "relates",
    # reporting and reasoning vocabulary in questions about someone's study ("Your study identified changes in X
    # associated with Y. How would you distinguish whether these are causes or consequences?")
    "your", "my", "our", "their", "study", "studies", "identified", "identify", "identifies", "identifying",
    "observed", "observe", "found", "find", "finding", "findings", "showed", "shows", "show", "revealed", "reveal",
    "reveals", "suggest", "suggests", "suggested", "change", "changes", "changed", "altered", "alteration",
    "alterations", "associated", "association", "associations", "linked", "link", "links", "distinguish",
    "distinguishing", "determine", "explain", "explains", "understand", "interpret", "might", "may", "would",
    "could", "should", "these", "those", "this", "that", "consequence", "consequences", "mechanism", "mechanisms",
    "contribution", "approach", "approaches", "address", "assess", "evaluate", "investigate", "examine",
    # effect verbs and comparatives that join an exposure to an outcome ("does X prevent Y", "higher X in Y")
    "prevent", "prevents", "preventing", "reduce", "reduces", "reducing", "lower", "lowers", "lowering", "improve",
    "improves", "improving", "increase", "increases", "increasing", "decrease", "decreases", "affect", "affects",
    "affecting", "work", "works", "working", "help", "helps", "higher", "high", "low", "elevated", "increased",
    "decreased", "reduced", "better", "worse", "best", "versus", "compared", "against", "usual",
    # research-request vocabulary ("I'm writing my thesis and need everything on X")
    "writing", "write", "thesis", "dissertation", "paper", "papers", "article", "articles", "literature", "need",
    "needs", "needed", "everything", "anything", "something", "information", "info", "please", "kindly", "help",
    "sources", "references", "summary", "summarize", "summarise", "overview", "update", "insights", "insight",
    "advances", "advance", "progress", "recent", "latest", "new", "novel", "current", "emerging", "future",
    "since", "till", "until", "during", "last", "past", "decade", "decades", "year", "years", "day", "days",
    "im", "i'm", "ive", "i've", "id", "i'd", "lets", "let's",
    # pasted-abstract section labels and reporting verbs
    "background", "introduction", "objective", "objectives", "methods", "method", "results", "conclusion",
    "conclusions", "purpose", "design", "setting", "conducted", "performed", "common", "commonly", "main",
    # generic people words (specific populations are constraints)
    "adults", "adult", "people", "persons", "individuals", "participants", "subjects", "patients", "patient",
    "population", "populations", "humans",
    # common Spanish/Portuguese/French function words, so non-English questions still yield concepts
    "de", "del", "la", "las", "el", "los", "en", "y", "e", "o", "para", "por", "con", "sobre", "que", "un", "una",
    "da", "do", "das", "dos", "em", "no", "na", "le", "les", "des", "du", "et", "pour", "avec", "sur", "au", "aux",
}
# Multi-word framing removed before parsing
_FRAMING_PHRASES = re.compile(
    r"\b(?:state[- ]of[- ]the[- ]art|state of knowledge|current state|what is known|an overview of|"
    r"the latest on|recent advances in|literature review on|systematic review of|review of)\b", re.I)
_NEGATION_RE = re.compile(r"\b(?:NOT|not|without|excluding|exclude|except)\b\s+([^,;:()]+?)(?=\s+(?:AND|OR|and|or)\b|[,;:()]|$)")


def strip_negated(question: str) -> str:
    """The question without its excluded parts ("asthma NOT children" -> "asthma")."""
    return _NEGATION_RE.sub(" ", question or "")

# Methods and data types: kept as concepts, but ranked after what the question is about, so "metabolomics" in
# "your metabolomics study identified changes in branched-chain amino acids" does not become the main topic
METHOD_WORDS = {
    "metabolomics", "metabolomic", "proteomics", "proteomic", "transcriptomics", "transcriptomic", "genomics",
    "genomic", "lipidomics", "lipidomic", "multi-omics", "multiomics", "omics", "epigenomics", "metagenomics",
    "machine", "learning", "deep", "neural", "network", "networks", "meta-analysis", "review", "systematic",
    "mendelian", "randomization", "randomisation", "sequencing", "rna-seq", "single-cell", "mass", "spectrometry",
    "lc-ms", "gc-ms", "nmr", "imaging", "cohort", "trial", "randomized", "randomised", "survey", "analysis",
    "profiling", "screening", "modelling", "modeling", "simulation", "data", "dataset", "datasets",
    "experiments", "experiment", "experimental", "prevalence", "incidence", "epidemiology", "epidemiological",
    "burden", "rates", "rate", "trends", "trend", "statistics", "reception", "applications", "application",
}
_LINK_SUFFIX = re.compile(r"-(?:related|associated|based|dependent|mediated|induced|driven|derived|linked)$", re.I)


def is_method_concept(phrase: str) -> bool:
    words = phrase.lower().split()
    return bool(words) and all(w in METHOD_WORDS for w in words)

# Experimental systems a question asks to cover; used to target searches, never required in every query
_MODEL_SYSTEMS: Dict[str, Tuple[str, ...]] = {
    "mouse models": ("mouse", "mice", "murine", "mouse model", "mouse models"),
    "animal models": ("animal model", "animal models", "rat", "rats", "in vivo"),
    "organoids": ("organoid", "organoids", "3d culture", "spheroids"),
    "human clinical samples": ("human clinical samples", "clinical samples", "patient samples", "human samples",
                               "human tissue", "biopsies", "biopsy", "patient-derived", "human cohorts"),
    "cell lines": ("cell line", "cell lines", "in vitro", "primary cells", "fibroblasts in culture"),
}

# Place constraints: canonical name -> spellings found in titles and abstracts (demonyms included)
_PLACES: Dict[str, Tuple[str, ...]] = {
    "India": ("india", "indian", "indians"),
    "China": ("china", "chinese"),
    "Pakistan": ("pakistan", "pakistani"),
    "Bangladesh": ("bangladesh", "bangladeshi"),
    "Sri Lanka": ("sri lanka", "sri lankan"),
    "Nepal": ("nepal", "nepalese", "nepali"),
    "Japan": ("japan", "japanese"),
    "Korea": ("korea", "korean"),
    "Indonesia": ("indonesia", "indonesian"),
    "Vietnam": ("vietnam", "vietnamese"),
    "Thailand": ("thailand", "thai"),
    "Philippines": ("philippines", "filipino"),
    "Malaysia": ("malaysia", "malaysian"),
    "Singapore": ("singapore", "singaporean"),
    "Iran": ("iran", "iranian"),
    "Saudi Arabia": ("saudi arabia", "saudi"),
    "Turkey": ("turkey", "turkish"),
    "Israel": ("israel", "israeli"),
    "Egypt": ("egypt", "egyptian"),
    "Nigeria": ("nigeria", "nigerian"),
    "Kenya": ("kenya", "kenyan"),
    "Ethiopia": ("ethiopia", "ethiopian"),
    "South Africa": ("south africa", "south african"),
    "Ghana": ("ghana", "ghanaian"),
    "Brazil": ("brazil", "brazilian"),
    "Mexico": ("mexico", "mexican"),
    "Argentina": ("argentina", "argentinian", "argentine"),
    "Chile": ("chile", "chilean"),
    "Colombia": ("colombia", "colombian"),
    "Canada": ("canada", "canadian"),
    "United States": ("united states", "usa", "u.s.", "american", "americans"),
    "United Kingdom": ("united kingdom", "uk", "britain", "british", "england", "english", "scotland", "wales"),
    "Germany": ("germany", "german"),
    "France": ("france", "french"),
    "Italy": ("italy", "italian"),
    "Spain": ("spain", "spanish"),
    "Netherlands": ("netherlands", "dutch"),
    "Sweden": ("sweden", "swedish"),
    "Norway": ("norway", "norwegian"),
    "Denmark": ("denmark", "danish"),
    "Finland": ("finland", "finnish"),
    "Poland": ("poland", "polish"),
    "Russia": ("russia", "russian"),
    "Australia": ("australia", "australian"),
    "New Zealand": ("new zealand",),
    "South Asia": ("south asia", "south asian", "south asians", "indian subcontinent"),
    "Southeast Asia": ("southeast asia", "south-east asia", "southeast asian"),
    "East Asia": ("east asia", "east asian"),
    "Middle East": ("middle east", "middle eastern", "gulf countries"),
    "Sub-Saharan Africa": ("sub-saharan africa", "sub-saharan"),
    "Africa": ("africa", "african"),
    "Europe": ("europe", "european"),
    "Latin America": ("latin america", "latin american", "south america"),
    "Asia": ("asia", "asian"),
    "low- and middle-income countries": ("low- and middle-income", "low and middle income", "lmic", "lmics",
                                         "developing countries"),
}

# Population constraints: canonical name -> spellings
_POPULATIONS: Dict[str, Tuple[str, ...]] = {
    "children": ("children", "child", "pediatric", "paediatric", "infants", "infant", "adolescents", "adolescent"),
    "neonates": ("neonates", "neonatal", "newborns", "newborn", "preterm"),
    "older adults": ("older adults", "elderly", "aged", "geriatric", "older people"),
    "women": ("women", "female", "females", "maternal", "pregnant", "pregnancy"),
    "pregnant women": ("pregnant women", "pregnant", "pregnancy", "maternal", "antenatal"),
}
# Question words that only count as a population constraint when they appear in the question itself
_POPULATION_TRIGGERS = {
    "children": ("children", "child", "pediatric", "paediatric", "adolescents", "adolescent", "kids"),
    "neonates": ("neonates", "neonatal", "newborns", "newborn", "preterm"),
    "older adults": ("older adults", "elderly", "geriatric", "older people"),
    "pregnant women": ("pregnant women", "pregnant", "pregnancy", "antenatal"),
    "women": ("women", "female"),
}


@dataclass(frozen=True)
class QuestionConstraint:
    kind: str  # "place" | "population"
    name: str
    spellings: Tuple[str, ...]

    def matches(self, text: str) -> bool:
        low = (text or "").lower()
        return any(re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", low) for s in self.spellings)


def detect_constraints(question: str) -> List[QuestionConstraint]:
    """Places and populations the question is explicitly restricted to (never ones it excludes)."""
    low = f" {strip_negated(question).lower()} "
    found: List[QuestionConstraint] = []
    # Longest names first so "South Asia" wins over "Asia", "South Africa" over "Africa"
    for name, spellings in sorted(_PLACES.items(), key=lambda kv: -max(len(s) for s in kv[1])):
        if any(re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", low) for s in spellings):
            if not any(name in c.name or any(s in " ".join(c.spellings) for s in spellings) for c in found):
                found.append(QuestionConstraint("place", name, spellings))
    for name, triggers in _POPULATION_TRIGGERS.items():
        if any(re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", low) for t in triggers):
            if name == "women" and any(c.name == "pregnant women" for c in found):
                continue
            found.append(QuestionConstraint("population", name, _POPULATIONS[name]))
    return found


def detect_model_systems(question: str) -> List[QuestionConstraint]:
    """Experimental systems the question wants covered (mouse models, organoids, human samples, ...)."""
    low = f" {question.lower()} "
    return [
        QuestionConstraint("model_system", name, spellings)
        for name, spellings in _MODEL_SYSTEMS.items()
        if any(re.search(rf"(?<![a-z]){re.escape(sp)}(?![a-z])", low) for sp in spellings)
    ]


def constraint_words(constraints: Sequence[QuestionConstraint]) -> set:
    """Every word of every spelling (and question trigger) of the detected constraints."""
    words = {w for c in constraints for s in c.spellings for w in s.split()}
    for c in constraints:
        words |= {w for t in _POPULATION_TRIGGERS.get(c.name, ()) for w in t.split()}
    return words


def concept_phrases(question: str, stop_words: set, max_phrases: int = 4, max_words: int = 4) -> List[str]:
    """
    Runs of consecutive content words, in question order. "What is the state of coronary artery disease in India"
    -> ["coronary artery disease"]; constraint words and framing words split runs and are not included.
    """
    question = strip_negated(_FRAMING_PHRASES.sub(" , ", question or ""))
    question = re.sub(r"(\d),(\d{3})\b", r"\1\2", question)  # 10,000 -> 10000
    constraints = detect_constraints(question)
    model_words = {w for c in detect_model_systems(question) for sp in c.spellings for w in sp.split()}
    model_words |= {"models", "model", "animal", "human", "clinical", "samples", "sample", "patients", "patient"} \
        if model_words else set()
    excluded = stop_words | QUESTION_FRAMING_WORDS | constraint_words(constraints) | model_words
    phrases: List[str] = []
    run: List[str] = []

    def flush():
        if run:
            # Long runs are usually two concepts; keep the first max_words words together
            phrases.append(" ".join(run[:max_words]))
            if len(run) > max_words:
                phrases.append(" ".join(run[max_words:max_words * 2]))
        run.clear()

    # A quoted phrase is one concept; punctuation, "vs" and boolean operators end a concept
    token_re = re.compile(r'"([^"]+)"|\u201c([^\u201d]+)\u201d|(\s[-\u2013\u2014]\s|[,;:()?!.]|\bAND\b|\bOR\b|\bvs\.?)'
                          r"|([^\W_][\w+/'\u2019-]*)", re.UNICODE)
    for m in token_re.finditer(question):
        quoted = m.group(1) or m.group(2)
        if quoted:
            flush()
            words = [w for w in re.findall(r"[^\W_][\w+/-]*", quoted) if w.lower() not in excluded or len(quoted.split()) > 1]
            if words:
                phrases.append(" ".join(w if (w.isupper() and len(w) <= 6) else w.lower() for w in words))
            continue
        if m.group(3):
            flush()
            continue
        tok = re.sub(r"['\u2019]s$", "", m.group(4) or "").strip("'\u2019-")  # Alzheimer's -> Alzheimer
        low = tok.lower()
        if not tok:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}s?", tok):  # years are a date filter, not a concept
            flush()
        elif len(low) == 1 and run and (tok.isupper() or tok.isdigit()):
            run.append(tok)  # vitamin D, hepatitis B, type 2
        elif low in excluded or len(low) < 2:
            flush()
        else:
            # "arginine-related metabolism" -> "arginine metabolism"
            tok = _LINK_SUFFIX.sub("", tok)
            run.append(tok if (tok.isupper() and len(tok) <= 6) else tok.lower())
    flush()
    unique = list(dict.fromkeys(p for p in phrases if p))
    # What the question is about first, methods after (stable within each group)
    ordered = [p for p in unique if not is_method_concept(p)] + [p for p in unique if is_method_concept(p)]
    return ordered[:max_phrases]


def concept_blocks(question: str, phrases: Sequence[str], max_blocks: int = 2) -> List[List[str]]:
    """
    Groups the leading concepts into search blocks that are ANDed together. Concepts the question lists side by
    side ("branched-chain amino acid and arginine metabolism") are alternatives within one block, so a study about
    either one is found; the next distinct concept forms the next block.
    """
    # Upper-case AND is a Boolean operator (both required), not a list of alternatives
    question = re.sub(r"\bAND\b", " ; ", question or "")
    text = _LINK_SUFFIX.sub("", " ".join(_LINK_SUFFIX.sub("", w) for w in question.lower().split()))
    text = re.sub(r"\s+", " ", text)

    def span(ph: str):
        m = re.search(re.escape(ph.lower()), text)
        return (m.start(), m.end()) if m else None

    blocks: List[List[str]] = []
    i = 0
    while i < len(phrases) and len(blocks) < max_blocks:
        block = [phrases[i]]
        while i + 1 < len(phrases):
            a, b = span(block[-1]), span(phrases[i + 1])
            if a and b and a[1] <= b[0] and re.fullmatch(r"[\s,]*(?:and|or|and/or|,)[\s,]*", text[a[1]:b[0]]):
                block.append(phrases[i + 1])
                i += 1
            else:
                break
        blocks.append(block)
        i += 1
    return blocks


# Words that qualify a concept rather than name it; a concept is matched by its other (specific) words
GENERIC_CONCEPT_WORDS = {
    "adverse", "outcome", "outcomes", "metabolism", "metabolic", "pathway", "pathways", "dysfunction",
    "progression", "levels", "level", "activity", "function", "functions", "risk", "risks", "event", "events",
    "effect", "effects", "impact", "response", "responses", "status", "burden", "disease", "diseases", "disorder",
    "disorders", "health", "factor", "factors",
}


def specific_words(phrase: str) -> List[str]:
    words = phrase.lower().split()
    spec = [w for w in words if w not in GENERIC_CONCEPT_WORDS]
    return spec or words


def initialism(phrase: str) -> str:
    """'branched-chain amino acid' -> 'BCAA', 'coronary artery disease' -> 'CAD' (only for 3+ letter results)."""
    parts = [w for w in re.split(r"[\s-]+", phrase) if w]
    abbr = "".join(w[0] for w in parts).upper()
    return abbr if len(abbr) >= 3 and len(parts) >= 3 else ""


def concept_mentions(phrase: str, text: str) -> int:
    """
    How often a concept appears in text: the phrase itself (plural, hyphen or space variants), its initialism
    (BCAA), or - for multi-word concepts written differently - all of its specific words (counted once).
    """
    if not phrase or not text:
        return 0
    low = text.lower()
    pattern = r"[\s-]?".join(re.escape(w) for w in re.split(r"[\s-]+", phrase.lower()) if w)
    n = len(re.findall(rf"(?<![a-z0-9]){pattern}(?:s|es)?(?![a-z])", low))
    abbr = initialism(phrase)
    if abbr:
        n += len(re.findall(rf"(?<![A-Za-z0-9]){abbr}s?(?![A-Za-z0-9])", text))
    if n:
        return n
    spec = specific_words(phrase)
    if len(phrase.split()) > 1 and all(re.search(rf"(?<![a-z0-9]){re.escape(w)}", low) for w in spec):
        return 1
    return 0
