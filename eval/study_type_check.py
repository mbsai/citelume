"""
Session 9: how well the study-type classifier (services/paper_selection.classify_study_type, frozen) recognises a
study's design.

    python -m eval.study_type_check runs/5b_dev runs/5c_dev runs/5c_dev_full runs/5d_dev runs/6_test_prisma \\
        runs/7_search_r1 runs/7_search_r2

Gold labels: the publication types PubMed indexers assigned (records returned by PubMed in the evaluation runs'
raw output, one per PubMed ID), mapped to coarse classes:
    correction, protocol, synthesis (systematic review, meta-analysis, guideline), rct, case report,
    commentary (editorial, comment, letter), review, other primary (a journal article with none of these types).
The classifier reads publication types first, so it is tested in two ways:
    wording only        the record with its publication types removed (as for most arXiv, bioRxiv and some
                        Semantic Scholar records)
    Semantic Scholar    the same paper as Semantic Scholar returned it, with its own types (where both sources
                        returned it)
Reports per-class precision, recall and F1, overall accuracy, and the confusion table. No model, no network.
Writes eval/results/9_study_types.md and .json.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from models import PaperRecord  # noqa: E402
from services.paper_selection import classify_study_type  # noqa: E402

CLASSES = ["correction", "protocol", "synthesis", "rct", "case report", "commentary", "review", "other primary"]
# Highest priority first, as an indexer would read a record with several types
GOLD = [
    ("correction", {"published erratum", "retracted publication", "retraction of publication",
                    "expression of concern"}),
    ("protocol", {"clinical trial protocol"}),
    ("synthesis", {"systematic review", "meta-analysis", "practice guideline", "guideline"}),
    ("rct", {"randomized controlled trial"}),
    ("case report", {"case reports"}),
    ("commentary", {"editorial", "comment", "letter", "news"}),
    ("review", {"review"}),
]
PREDICTED = {"correction": "correction", "protocol": "protocol", "evidence_synthesis": "synthesis", "rct": "rct",
             "case_report": "case report", "commentary": "commentary", "review": "review"}


def gold_class(types: List[str]) -> Optional[str]:
    t = {x.strip().lower() for x in types}
    if not t:
        return None
    for cls, names in GOLD:
        if t & names:
            return cls
    return "other primary" if "journal article" in t else None


def predicted_class(p: PaperRecord) -> str:
    return PREDICTED.get(classify_study_type(p), "other primary")


def table(pairs: List[tuple]) -> dict:
    conf: Dict[str, Counter] = defaultdict(Counter)
    for g, pr in pairs:
        conf[g][pr] += 1
    rows = {}
    for c in CLASSES:
        tp = conf[c][c]
        fp = sum(conf[g][c] for g in CLASSES if g != c)
        fn = sum(conf[c][x] for x in CLASSES if x != c)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        rows[c] = {"n": tp + fn, "precision": prec, "recall": rec, "f1": f1}
    acc = sum(conf[c][c] for c in CLASSES) / len(pairs) if pairs else None
    return {"rows": rows, "accuracy": acc, "n": len(pairs),
            "confusion": {g: dict(conf[g]) for g in CLASSES}}


def main(argv: List[str]) -> int:
    folders = [Path(a) for a in argv[1:]] or [ROOT.parent / "runs" / "6_test_prisma"]
    pubmed: Dict[str, PaperRecord] = {}
    s2: Dict[str, PaperRecord] = {}
    for folder in folders:
        for raw in sorted(folder.glob("*/raw.json.gz")):
            with gzip.open(raw, "rt", encoding="utf-8") as fh:
                for r in json.load(fh):
                    p = PaperRecord.model_validate(r)
                    if not p.pmid:
                        continue
                    if p.source == "PubMed" and p.publication_types and p.pmid not in pubmed:
                        pubmed[p.pmid] = p
                    elif p.source == "Semantic Scholar" and p.pmid not in s2:
                        s2[p.pmid] = p
    wording, via_s2 = [], []
    for pmid, p in pubmed.items():
        g = gold_class(p.publication_types)
        if g is None:
            continue
        wording.append((g, predicted_class(p.model_copy(update={"publication_types": []}))))
        if pmid in s2:
            via_s2.append((g, predicted_class(s2[pmid])))
    res = {"records": len(pubmed), "gold_distribution": dict(Counter(g for g, _ in wording)),
           "wording_only": table(wording), "semantic_scholar": table(via_s2)}
    f = lambda v: "" if v is None else f"{v:.2f}"
    lines = ["# Session 9: study-type classifier against PubMed publication types", "",
             f"{len(wording)} PubMed records with a usable publication type (from {len(folders)} evaluation runs' raw "
             f"search output, one per PubMed ID); {len(via_s2)} of them also returned by Semantic Scholar. Gold: the "
             "indexers' publication types. The classifier is frozen (selection protocol 2.5).", ""]
    for name, key in (("Wording only (publication types removed)", "wording_only"),
                      ("As Semantic Scholar returned the record (its own types)", "semantic_scholar")):
        t = res[key]
        lines += [f"## {name}", "", f"Accuracy {f(t['accuracy'])} on {t['n']} records.", "",
                  "| Class | Records | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        lines += [f"| {c} | {t['rows'][c]['n']} | {f(t['rows'][c]['precision'])} | {f(t['rows'][c]['recall'])} | "
                  f"{f(t['rows'][c]['f1'])} |" for c in CLASSES if t["rows"][c]["n"]]
        lines += ["", "Confusion (rows: gold, columns: predicted):", "",
                  "| Gold \\ predicted | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        lines += [f"| {g} | " + " | ".join(str(t["confusion"][g].get(c, "")) for c in CLASSES) + " |"
                  for g in CLASSES if t["rows"][g]["n"]]
        lines.append("")
    md = "\n".join(lines)
    (ROOT / "results" / "9_study_types.md").write_text(md, encoding="utf-8")
    (ROOT / "results" / "9_study_types.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
