"""
Shared patterns for reading study facts out of abstracts and open-access full text, used by both the rule-based
study extraction and the evidence table so the two never disagree about whether a study reported its sample size.
"""
import re
from typing import Iterable, Optional

from models import PaperRecord

# What a study can be a sample of, across clinical, laboratory, computational and bibliometric research
_SAMPLE_NOUNS = (
    "patients|participants|subjects|adults|individuals|children|adolescents|infants|neonates|women|men|persons|people|"
    "cases|controls|respondents|volunteers|donors|households|families|students|workers|employees|residents|"
    "samples|specimens|biopsies|cells|organoids|mice|rats|animals|zebrafish|eyes|hospitals|centres|centers|sites|"
    "villages|districts|sequences|peptides|proteins|compounds|molecules|structures|images|scans|records|"
    "articles|publications|papers|studies|trials|datasets|reviews|cohorts|genes|variants|snps"
)
# "n = 5167", "997 young patients", "1,024 adults aged 20-79", "a total of 52 studies"
SAMPLE_SIZE_PATTERN = re.compile(
    rf"\bn\s*=\s*\d[\d,]*|\b\d{{1,3}}(?:,\d{{3}})+(?:\s+[a-z][a-z-]*){{0,3}}?\s+(?:{_SAMPLE_NOUNS})\b|"
    rf"\b\d+(?:\s+[a-z][a-z-]*){{0,3}}?\s+(?:{_SAMPLE_NOUNS})\b",
    re.IGNORECASE,
)
# Numbers that are years, ages or durations rather than counts
_NOT_A_COUNT = re.compile(r"\b(?:19|20)\d{2}\b(?!\s*,\d)|\byears?\b|\bmonths?\b|\bweeks?\b|\bdays?\b|\baged?\b", re.I)


def find_sample_size(*texts: Iterable[str]) -> Optional[str]:
    """First credible sample-size phrase in the texts, in the order given (e.g. sentence, abstract, methods)."""
    for text in texts:
        for m in SAMPLE_SIZE_PATTERN.finditer(text or ""):
            phrase = m.group(0).strip()
            if not _NOT_A_COUNT.search(phrase):
                return phrase
    return None


def study_sample_size(p: PaperRecord, sentence: str = "") -> Optional[str]:
    """Sample size for a study: the sentence itself, then the abstract, then the full text's methods and results."""
    sections = p.full_text_sections or {}
    return find_sample_size(sentence, p.abstract or "", sections.get("methods", ""), sections.get("results", ""))
