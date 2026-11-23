"""
Session 6 error analysis: classifies every included study of the 22 test reviews that CiteLume's systematic-style
run did not select, by where and why it was lost. Reads eval/results/6_test_loss_by_stage.json (from
eval/loss_by_stage.py) and the stored corpora; no model, no network; changes nothing in the measured pipeline.

    python -m eval.error_analysis runs/6_test_prisma

Categories:
    search: depth              no search returned it, but one of CiteLume's own PubMed queries matches it beyond
                               the records fetched
    search: query sensitivity  no CiteLume query matches it
    eligibility: topic words   found, but judged not about the question by the concept matching
    eligibility: typed as review   found, but excluded as a review article (publication type)
    eligibility: other         not a research report, place or population
    screening: criterion       the model answered "no" to a screening criterion (the criterion is named)
    screening: cap or limit    eligible but never screened, or beyond the study limit
Writes eval/results/6_error_analysis.json and prints the tables used in 6_error_analysis.md.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from eval.scoring import match, review_split  # noqa: E402
from models import Corpus  # noqa: E402

LOSS = ROOT / "results" / "6_test_loss_by_stage.json"


def category(row: dict) -> str:
    reason = row["reason"]
    if row["stage"] == "missed":
        return "search: depth" if row.get("matched_by_pubmed_query") else "search: query sensitivity"
    if row["stage"] == "found":
        return "duplicate merging"
    if row["stage"] == "merged":
        if reason.startswith("Not about the topic"):
            return "eligibility: topic words"
        if reason.startswith("Review article"):
            return "eligibility: typed as review"
        return "eligibility: other"
    if reason.startswith("screened out"):
        return "screening: criterion"
    return "screening: cap or limit"


def main(argv) -> int:
    runs = Path(argv[1]) if len(argv) > 1 else ROOT.parent / "runs" / "6_test_prisma"
    loss = json.loads(LOSS.read_text())
    reviews = {r["review_id"]: r for r in review_split("test", session6_test_analysis=True)}
    per_review = defaultdict(Counter)
    totals = Counter()
    criteria = Counter()
    for r in loss["reviews"]:
        if r.get("status") != "done":
            continue
        for row in r["studies"]:
            if row["stage"] == "included":
                continue
            cat = category(row)
            per_review[r["review_id"]][cat] += 1
            totals[cat] += 1
            if cat == "screening: criterion":
                criteria[row["reason"].split(":", 1)[1].strip()] += 1
    # Publication types of the included studies excluded as reviews
    typed = []
    for rid, rv in reviews.items():
        folder = runs / rid
        if not (folder / "corpus.json.gz").exists():
            continue
        with gzip.open(folder / "corpus.json.gz", "rt", encoding="utf-8") as fh:
            c = Corpus.model_validate_json(fh.read())
        recs = {p.id: p for p in c.records}
        for s in match(rv["included"], c.records).found:
            if c.excluded.get(s["record_id"], "").startswith("Review article"):
                typed.append({"review_id": rid, "title": recs[s["record_id"]].title,
                              "publication_types": recs[s["record_id"]].publication_types})
    lost = sum(totals.values())
    out = {"lost": lost, "totals": dict(totals.most_common()), "screening_criteria": dict(criteria.most_common()),
           "per_review": {k: dict(v) for k, v in sorted(per_review.items())}, "typed_as_review": typed}
    (ROOT / "results" / "6_error_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"Included studies not selected: {lost} (of {loss['included_studies']})\n")
    print("| Where and why | Studies | Share of those lost |\n|---|---|---|")
    for k, v in totals.most_common():
        print(f"| {k} | {v} | {100 * v / lost:.0f}% |")
    print("\n| Screening criterion answered 'no' | Studies |\n|---|---|")
    for k, v in criteria.most_common():
        print(f"| {k} | {v} |")
    print(f"\nTyped as review: {len(typed)} included studies, publication types "
          f"{Counter(t for x in typed for t in x['publication_types']).most_common(4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
