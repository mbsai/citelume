"""
Offers to broaden a guided review's protocol when its search found too little (reported by a demo user, 26 Nov
2026: "atrazine biotransformation by Aspergillus hortai" has no studies, and the review should have looked at
Aspergillus instead of only reporting the absence).

Only offered, never applied: the user decides, and the change is recorded in the protocol like any other edit. The
rule is narrow on purpose: a species name (genus + epithet) that few records mention can be widened to its genus,
which still includes the species. Nothing here is part of the measured selection protocol.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from pydantic import BaseModel

FEW_ELIGIBLE = 5        # offer when fewer studies than this met the criteria
FEW_MENTIONS = 2        # ... and a species is mentioned by at most this many records
# Endings of Latin species epithets (hortai, fumigatus, oryzae, niger, versicolor, coli, aureus, sydowii, ...)
_EPITHET = re.compile(r"^[a-z]{3,}(?:i|ii|ae|us|um|is|a|es|er|or|ans|ens|ix|ax|ensis|oides|icola|ium)$")
# First words that are ordinary English, not genera ("study area", "control group")
_NOT_GENUS = {"study", "control", "data", "case", "risk", "health", "care", "drug", "cell", "gene", "soil", "water",
              "plant", "animal", "human", "model", "field", "food", "acute", "chronic", "early", "late", "high", "low",
              "total", "mean", "body", "blood", "brain", "heart", "liver", "lung", "skin", "bone", "oral", "nasal",
              # Latin disease names, which look like species names
              "diabetes", "lupus", "status", "situs", "corpus", "pemphigus", "herpes", "lichen", "tinea", "acne",
              "erythema", "molluscum", "pityriasis", "spina", "cor", "dementia", "angina", "vasa"}
# Genera end like this (Aspergillus, Escherichia, Penicillium, Candida, Pseudomonas, Streptomyces, Salmonella ...)
_GENUS = re.compile(r"(?:us|ia|ium|a|as|es|is|um|ma|myces|cus|ter|ella|ina|on|ax|ex|ix|yx)$")


class BroadenSuggestion(BaseModel):
    concept: str
    broader: str
    mentions: int
    reason: str


def species_genus(name: str) -> Optional[str]:
    """The genus of a species name ("aspergillus hortai" -> "aspergillus"), or None if it does not look like one."""
    words = re.sub(r"\s+", " ", (name or "").strip()).split(" ")
    if len(words) == 2 and words[1].lower() in ("sp", "sp.", "spp", "spp."):
        return None  # already the genus
    if len(words) != 2:
        return None
    genus, epithet = words[0].lower(), words[1].lower()
    if (len(genus) < 4 or not genus.isalpha() or genus in _NOT_GENUS or not _GENUS.search(genus)
            or not _EPITHET.match(epithet)):
        return None
    return genus


def mentions(records: Sequence, term: str) -> int:
    pat = re.compile(r"\b" + re.escape(term.lower()).replace(r"\ ", r"\s+") + r"\b")
    return sum(1 for r in records if pat.search(f"{getattr(r, 'title', '') or ''} {getattr(r, 'abstract', '') or ''}".lower()))


def broaden_suggestions(protocol, eligible: int, records: Sequence) -> List[BroadenSuggestion]:
    if protocol is None or eligible >= FEW_ELIGIBLE:
        return []
    out: List[BroadenSuggestion] = []
    for c in getattr(protocol, "concepts", []) or []:
        if getattr(c, "role", "") == "comparison":
            continue
        genus = species_genus(c.name)
        if not genus:
            continue
        n = mentions(records, c.name)
        if n > FEW_MENTIONS:
            continue
        out.append(BroadenSuggestion(
            concept=c.name, broader=genus, mentions=n,
            reason=(f"{n} of the {len(records)} records found mention {c.name}. Searching for the genus "
                    f"{genus.capitalize()} includes this species and finds studies of related species, which the "
                    "review will report as indirect evidence.")))
    return out
