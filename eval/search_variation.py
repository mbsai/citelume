"""
Session 7, E2 (records found): how much the records found for the same question vary between runs.

    python -m eval.search_variation --runs runs/7_search_r1 runs/7_search_r2 --stored runs/6_test_prisma --set test

The repeats are built with no model (rule-based planning), so two repeats differ only by what the databases
return; the stored Session 6 run was planned by the model, so comparing a repeat with it adds the planning
difference. Records are identified by PubMed ID, DOI or normalised title (record IDs can differ between runs for
the same paper). Reports, per pair, the overlap (Jaccard) of the record sets after duplicate merging and the
included studies found by each run. Writes eval/results/7_search_variation.md and .json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from eval.scoring import match, review_split  # noqa: E402
from models import Corpus  # noqa: E402


def load(folder: Path) -> Optional[Corpus]:
    path = folder / "corpus.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return Corpus.model_validate_json(fh.read())


def key(p) -> str:
    if p.pmid:
        return f"pmid:{p.pmid.strip()}"
    if p.doi:
        return f"doi:{p.doi.strip().lower()}"
    return "title:" + re.sub(r"[^a-z0-9]", "", (p.title or "").lower())


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.search_variation")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--stored", type=Path, required=True)
    p.add_argument("--set", choices=("development", "test"), required=True)
    a = p.parse_args(argv)
    names = [r.name for r in a.runs] + ["stored (model-planned)"]
    folders = list(a.runs) + [a.stored]
    rows = []
    for r in review_split(a.set, session6_test_analysis=(a.set == "test")):
        cs = [load(f / r["review_id"]) for f in folders]
        if any(c is None for c in cs):
            continue
        sets = [{key(x) for x in c.records} for c in cs]
        found = [len(match(r["included"], c.records).found) / len(r["included"]) for c in cs]
        row = {"review_id": r["review_id"], "records": [len(s) for s in sets], "pool_recall": found}
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                row[f"{names[i]} vs {names[j]}"] = jaccard(sets[i], sets[j])
        rows.append(row)
    if not rows:
        print("no complete set of runs yet")
        return 1
    pair_keys = [k for k in rows[0] if " vs " in k]
    lines = [f"# Session 7, E2: records found on repeated searches ({a.set} set)", "",
             f"{len(rows)} questions. Runs: {', '.join(names)}. The repeats used no model (rule-based planning); the "
             "stored run was planned by the model. Overlap is Jaccard of the record sets after duplicate merging "
             "(records keyed by PubMed ID, DOI or title).", "", "| Pair | Median overlap | Lowest |", "|---|---|---|"]
    for k in pair_keys:
        vals = [x[k] for x in rows]
        lines.append(f"| {k} | {statistics.median(vals):.2f} | {min(vals):.2f} |")
    lines += ["", "| Run | Mean pool recall | Median records |", "|---|---|---|"]
    for i, n in enumerate(names):
        lines.append(f"| {n} | {100 * statistics.fmean(x['pool_recall'][i] for x in rows):.0f}% | "
                     f"{statistics.median(x['records'][i] for x in rows):.0f} |")
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "7_search_variation.md").write_text(md)
    (ROOT / "results" / "7_search_variation.json").write_text(json.dumps(rows, indent=1))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
