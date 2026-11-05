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
}

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
    """Places and populations the question is explicitly restricted to."""
    low = f" {question.lower()} "
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
    constraints = detect_constraints(question)
    model_words = {w for c in detect_model_systems(question) for sp in c.spellings for w in sp.split()}
    model_words |= {"models", "model", "animal", "human", "clinical", "samples", "sample", "patients", "patient"} \
        if model_words else set()
    excluded = stop_words | QUESTION_FRAMING_WORDS | constraint_words(constraints) | model_words
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+/-]*", question)
    phrases: List[str] = []
    run: List[str] = []

    def flush():
        if run:
            # Long runs are usually two concepts; keep the first max_words words together
            phrases.append(" ".join(run[:max_words]))
            if len(run) > max_words:
                phrases.append(" ".join(run[max_words:max_words * 2]))
        run.clear()

    for tok in tokens:
        low = tok.lower()
        if low in excluded or len(low) < 2:
            flush()
        else:
            run.append(tok if (tok.isupper() and len(tok) <= 6) else low)
    flush()
    return list(dict.fromkeys(p for p in phrases if p))[:max_phrases]
