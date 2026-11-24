"""
Session 9: precision of duplicate merging (services/corpus.merge_duplicates, frozen), on a hand-checked sample.

    python -m eval.merge_check build runs/6_test_prisma --n 100
    python -m eval.merge_check score

The frozen merge is replayed on the saved raw search output of each review (every raw record tagged, so the
records merged into each study can be read back). From the merged groups, pairs of records that were merged are
sampled (seed 20261118), half from merges with no shared identifier (matched on the normalised title alone), half
from merges on a shared identifier (DOI, PubMed ID, arXiv ID or Semantic Scholar ID) where the two titles differ.
Merges of records with the same identifier and the same title are not sampled: they are the same record by
construction. The sheet (eval/annotation/merge_sheet.csv, kept local) shows both records; fill "same_study" with
yes, no or unsure. score reports the share judged the same study, per kind of merge, weighted to all such merges.
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from models import PaperRecord  # noqa: E402
from services.corpus import merge_duplicates  # noqa: E402

SEED = 20261118
ANN = ROOT / "annotation"


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _ids(p: PaperRecord) -> set:
    out = set()
    if p.doi:
        out.add("doi:" + p.doi.strip().lower())
    if p.pmid:
        out.add("pmid:" + p.pmid.strip())
    if p.arxiv_id:
        out.add("arxiv:" + re.sub(r"v\d+$", "", p.arxiv_id.strip().lower()))
    if p.s2_id:
        out.add("s2:" + p.s2_id.strip().lower())
    return out


def merged_pairs(raw: List[PaperRecord]) -> List[tuple]:
    tagged = []
    for i, p in enumerate(raw):
        q = p.model_copy(deep=True)
        q.retrieved_by = [f"raw:{i}"]
        tagged.append(q)
    pairs = []
    for m in merge_duplicates(tagged):
        members = [int(t[4:]) for t in m.retrieved_by if t.startswith("raw:")]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pa, pb = raw[members[a]], raw[members[b]]
                same_ids = bool(_ids(pa) & _ids(pb))
                same_title = _norm(pa.title) == _norm(pb.title)
                if same_ids and same_title:
                    continue
                pairs.append(("identifier, titles differ" if same_ids else "title only", pa, pb))
    return pairs


def describe(p: PaperRecord) -> str:
    authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
    ids = " ".join(sorted(_ids(p)))
    return f"{p.title} | {authors} | {p.year or 'n.d.'} | {p.venue or ''} | {p.source} | {ids}"


def build(folders: List[Path], n: int) -> int:
    by_kind = defaultdict(list)
    for folder in folders:
        for raw_path in sorted(folder.glob("*/raw.json.gz")):
            with gzip.open(raw_path, "rt", encoding="utf-8") as fh:
                raw = [PaperRecord.model_validate(r) for r in json.load(fh)]
            for kind, pa, pb in merged_pairs(raw):
                by_kind[kind].append((raw_path.parent.name, pa, pb))
    rng = random.Random(SEED)
    sample = []
    for kind, items in sorted(by_kind.items()):
        rng.shuffle(items)
        sample += [(kind, *x) for x in items[:n // 2]]
    rng.shuffle(sample)
    ANN.mkdir(parents=True, exist_ok=True)
    key = {"seed": SEED, "population": {k: len(v) for k, v in by_kind.items()}, "units": {}}
    with (ANN / "merge_sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "record_a", "record_b", "same_study", "notes"])
        for i, (kind, qid, pa, pb) in enumerate(sample, start=1):
            pid = f"M{i:03d}"
            key["units"][pid] = {"kind": kind, "question_id": qid}
            w.writerow([pid, describe(pa), describe(pb), "", ""])
    (ANN / "merge_key.json").write_text(json.dumps(key, indent=1))
    print(f"merged pairs by kind: {key['population']}; sampled {len(sample)} into {ANN / 'merge_sheet.csv'}")
    return 0


def score() -> int:
    key = json.loads((ANN / "merge_key.json").read_text())
    with (ANN / "merge_sheet.csv").open(encoding="utf-8") as fh:
        judged = {r["pair_id"]: r["same_study"].strip().lower() for r in csv.DictReader(fh) if r["same_study"].strip()}
    bad = sorted({j for j in judged.values() if j not in ("yes", "no", "unsure")})
    if bad:
        print(f"unrecognised values {bad}; use yes, no or unsure")
        return 1
    lines = ["# Session 9: precision of duplicate merging", "",
             f"Hand-checked pairs of merged records ({len(judged)} judged; 'unsure' left out). Population of merged "
             f"pairs by kind: {key['population']}. Identical records (same identifier and title) are not counted.", "",
             "| Kind of merge | Pairs judged | Same study | Precision |", "|---|---|---|---|"]
    out = {}
    total_pop = sum(key["population"].values())
    weighted = 0.0
    for kind in sorted(key["population"]):
        js = [j for pid, j in judged.items() if key["units"][pid]["kind"] == kind and j != "unsure"]
        prec = sum(j == "yes" for j in js) / len(js) if js else None
        out[kind] = {"judged": len(js), "precision": prec}
        if prec is not None:
            weighted += prec * key["population"][kind] / total_pop
        lines.append(f"| {kind} | {len(js)} | {sum(j == 'yes' for j in js)} | "
                     + ("" if prec is None else f"{100 * prec:.0f}%") + " |")
    lines += ["", f"Weighted to all non-identical merges: {100 * weighted:.0f}% correct."]
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "9_merging.md").write_text(md)
    (ROOT / "results" / "9_merging.json").write_text(json.dumps({"kinds": out, "weighted": weighted}, indent=1))
    print(md)
    return 0


def main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[1] == "score":
        return score()
    if len(argv) >= 3 and argv[1] == "build":
        n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 100
        return build([Path(a) for a in argv[2:] if not a.startswith("--") and not a.isdigit()], n)
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
