"""
Matching a benchmark review's included studies against a CiteLume corpus, as eval/PROTOCOL.md section 5 defines it:
a study counts as found when a record has its PubMed ID or DOI (case-insensitive) or, for a record with neither, the
same title after removing case, punctuation and spaces.

    from eval.scoring import load_reviews, review_split, match, recall

Used by Sessions 5b (loss by stage, development set only) and 6 (test set). `review_split` refuses test reviews
unless the caller says it is the Session 6 analysis, so a test review is not looked at by accident.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

BENCHMARK = Path(__file__).resolve().parent / "benchmark"
REVIEWS_FILE = BENCHMARK / "clef2019_intervention.jsonl"
SPLIT_FILE = BENCHMARK / "split.json"


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _norm_doi(d: str) -> str:
    d = (d or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)


def _get(rec: Any, name: str) -> str:
    v = rec.get(name) if isinstance(rec, dict) else getattr(rec, name, None)
    return str(v or "").strip()


def load_reviews(path: Path = REVIEWS_FILE) -> Dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {r["review_id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def load_split(path: Path = SPLIT_FILE) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"development": list(data["development"]), "test": list(data["test"])}


def check_split(reviews: Dict[str, dict], split: Dict[str, List[str]], seed: int) -> List[str]:
    """Problems with the committed split ([] when it is the seeded draw over exactly these reviews)."""
    problems = []
    dev, test = set(split["development"]), set(split["test"])
    if dev & test:
        problems.append(f"in both sets: {sorted(dev & test)}")
    if dev | test != set(reviews):
        problems.append("the split does not cover exactly the benchmark reviews")
    ordered = sorted(reviews)
    random.Random(seed).shuffle(ordered)
    n_dev = len(ordered) // 3
    if sorted(ordered[:n_dev]) != sorted(dev):
        problems.append("the development set is not the seeded draw")
    return problems


def review_split(which: str, session6_test_analysis: bool = False) -> List[dict]:
    """The reviews of one set. The test set is for the Session 6 analysis only (eval/PROTOCOL.md section 4)."""
    if which == "test" and not session6_test_analysis:
        raise PermissionError("Test reviews are held out until Session 6 (eval/PROTOCOL.md, section 4).")
    reviews = load_reviews()
    return [reviews[r] for r in load_split()[which]]


@dataclass
class Match:
    found: List[dict] = field(default_factory=list)       # included studies found, each with "record_id" and "how"
    missing: List[dict] = field(default_factory=list)
    total: int = 0

    @property
    def recall(self) -> float:
        return len(self.found) / self.total if self.total else 0.0


def match(included: Sequence[dict], records: Iterable[Any]) -> Match:
    """Which of a review's included studies appear among `records` (PaperRecord objects or dicts)."""
    by_pmid: Dict[str, str] = {}
    by_doi: Dict[str, str] = {}
    by_title: Dict[str, str] = {}
    for r in records:
        rid = _get(r, "id")
        pmid, doi = _get(r, "pmid"), _norm_doi(_get(r, "doi"))
        if pmid:
            by_pmid.setdefault(pmid, rid)
        if doi:
            by_doi.setdefault(doi, rid)
        if not pmid and not doi:
            t = _norm_title(_get(r, "title"))
            if len(t) >= 20:
                by_title.setdefault(t, rid)
    out = Match(total=len(included))
    for s in included:
        hit: Optional[tuple] = None
        if s.get("pmid") and s["pmid"] in by_pmid:
            hit = (by_pmid[s["pmid"]], "pmid")
        elif s.get("doi") and _norm_doi(s["doi"]) in by_doi:
            hit = (by_doi[_norm_doi(s["doi"])], "doi")
        elif _norm_title(s.get("title", "")) in by_title:
            hit = (by_title[_norm_title(s["title"])], "title")
        if hit:
            out.found.append({**s, "record_id": hit[0], "how": hit[1]})
        else:
            out.missing.append(s)
    return out


def recall(included: Sequence[dict], records: Iterable[Any]) -> float:
    return match(included, records).recall


def questions(which: str, session6_test_analysis: bool = False) -> List[dict]:
    """Batch-runner lines (services/batch.py) for one set: the review title as the question, limited to the
    review's year."""
    return [{"id": r["review_id"], "query": r["question"], "year_to": r["year_to"]}
            for r in review_split(which, session6_test_analysis)]


if __name__ == "__main__":
    import sys

    # python -m eval.scoring questions development > runs/dev_questions.jsonl
    if len(sys.argv) == 3 and sys.argv[1] == "questions" and sys.argv[2] in ("development", "test"):
        for q in questions(sys.argv[2]):
            print(json.dumps(q, ensure_ascii=False))
    else:
        print("usage: python -m eval.scoring questions development", file=sys.stderr)
        sys.exit(2)
