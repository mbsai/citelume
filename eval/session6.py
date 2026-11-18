"""
Session 6 (E1, recall): the measures of eval/PROTOCOL.md section 5, computed from batch-runner output.

    python -m eval.session6 --prisma runs/6_test_prisma --standard runs/6_test_standard --set test
    python -m eval.session6 --prisma runs/5d_dev --set development --name dev_check     (a dry run)

Per review, from the stored corpora (no model, no network):
    pool recall            included studies among the records after duplicate merging (primary measure)
    pool recall, databases only   the same, counting only records some database search returned (without
                                  citation chasing and the reference lists of reviews)
    top 20 / top 60        PRISMA run: included studies among the first 20 / 60 eligible records in screening order
    selected               PRISMA run: included studies in the final selection (after model screening and the limit)
    standard selected      Standard run: included studies among its 20 selected studies
    screening sensitivity  of the included studies the model screened, the share it kept
    screening precision    of the studies the model kept, the share the review included
    baselines              on the same pool (PRISMA run): BM25 of the question over title and abstract, citation
                           count, and random order (seed 20261118); included studies in the top 20 and top 60

Summaries across reviews: median, mean and 95% bootstrap confidence interval of the mean (10,000 resamples of
reviews, seed 20261118), over all reviews and over reviews with at most 60 included studies. Writes
eval/results/<name>.md, .json and a figure (.png).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from eval.scoring import match, review_split  # noqa: E402
from models import Corpus  # noqa: E402

SEED = 20261118
RESAMPLES = 10_000
OTHER_METHODS = {"Citation searching", "Reference lists of reviews"}
RESULTS = ROOT / "results"


def load_corpus(folder: Path) -> Optional[Corpus]:
    path = folder / "corpus.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return Corpus.model_validate_json(fh.read())


def _share(found: int, total: int) -> Optional[float]:
    return found / total if total else None


def baseline_orders(corpus: Corpus, question: str) -> Dict[str, List[str]]:
    from services.paper_selection import bm25_scores

    recs = list(corpus.records)
    bm25 = bm25_scores(question, [f"{r.title}. {r.title}. {r.abstract or ''}" for r in recs])
    by_bm25 = [r.id for _, r in sorted(zip(bm25, recs), key=lambda x: (-x[0], x[1].id))]
    by_cites = [r.id for r in sorted(recs, key=lambda r: (-(r.citation_count or 0), r.id))]
    rnd = [r.id for r in sorted(recs, key=lambda r: r.id)]
    random.Random(SEED).shuffle(rnd)
    return {"BM25": by_bm25, "Citations": by_cites, "Random": rnd}


def review_measures(review: dict, prisma: Optional[Corpus], standard: Optional[Corpus]) -> Dict[str, Optional[float]]:
    gold = review["included"]
    n = len(gold)
    m: Dict[str, Optional[float]] = {"included": n}
    if prisma is not None:
        in_pool = match(gold, prisma.records).found
        m["pool recall"] = _share(len(in_pool), n)
        db_records = [r for r in prisma.records if set([r.source, *(r.all_sources or [])]) - OTHER_METHODS]
        m["pool recall, databases only"] = _share(len(match(gold, db_records).found), n)
        gold_ids = {s["record_id"] for s in in_pool}
        order = [c.record_id for c in sorted(prisma.candidates, key=lambda c: c.rank)]
        m["top 20"] = _share(sum(1 for i in order[:20] if i in gold_ids), n)
        m["top 60"] = _share(sum(1 for i in order[:60] if i in gold_ids), n)
        m["selected"] = _share(sum(1 for i in prisma.included_ids if i in gold_ids), n)
        decisions = prisma.selection_report.screening_decisions if prisma.selection_report else []
        screened_gold = [d for d in decisions if d.screened and d.record_id in gold_ids]
        kept = [d for d in decisions if d.screened and d.effective_include]
        m["screening sensitivity"] = _share(sum(1 for d in screened_gold if d.effective_include), len(screened_gold))
        m["screening precision"] = _share(sum(1 for d in kept if d.record_id in gold_ids), len(kept))
        for name, ids in baseline_orders(prisma, review["question"]).items():
            m[f"{name} top 20"] = _share(sum(1 for i in ids[:20] if i in gold_ids), n)
            m[f"{name} top 60"] = _share(sum(1 for i in ids[:60] if i in gold_ids), n)
    if standard is not None:
        std_gold = {s["record_id"] for s in match(gold, standard.records).found}
        m["standard pool recall"] = _share(len(std_gold), n)
        m["standard selected"] = _share(sum(1 for i in standard.included_ids if i in std_gold), n)
    return m


def bootstrap_ci(values: Sequence[float]) -> tuple:
    rng = random.Random(SEED)
    k = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=k)) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def summarise(rows: List[dict], keys: Sequence[str]) -> Dict[str, dict]:
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        out[key] = {"n": len(vals), "median": statistics.median(vals), "mean": statistics.fmean(vals),
                    "ci_low": lo, "ci_high": hi}
    return out


MEASURES = ["pool recall", "pool recall, databases only", "top 20", "top 60", "selected", "screening sensitivity",
            "screening precision", "standard pool recall", "standard selected",
            "BM25 top 20", "BM25 top 60", "Citations top 20", "Citations top 60", "Random top 20", "Random top 60"]


def pct(x: Optional[float]) -> str:
    return "" if x is None else f"{100 * x:.0f}%"


def markdown(name: str, which: str, rows: List[dict], s_all: dict, s_small: dict, missing: List[str]) -> str:
    lines = [f"# Session 6, recall ({which} set): {name}", "",
             f"{len(rows)} reviews analysed" + (f"; not run or failed: {', '.join(missing)}" if missing else "")
             + ". Measures and summaries as pre-registered in eval/PROTOCOL.md section 5; 95% CIs are bootstrap "
             f"intervals of the mean ({RESAMPLES:,} resamples of reviews, seed {SEED}).", "",
             "| Measure | Reviews | Median | Mean (95% CI) | Mean, reviews with <= 60 included (95% CI) |",
             "|---|---|---|---|---|"]
    for k in MEASURES:
        a, b = s_all.get(k), s_small.get(k)
        if not a:
            continue
        lines.append(f"| {k} | {a['n']} | {pct(a['median'])} | {pct(a['mean'])} ({pct(a['ci_low'])}-{pct(a['ci_high'])}) | "
                     + (f"{pct(b['mean'])} ({pct(b['ci_low'])}-{pct(b['ci_high'])})" if b else "") + " |")
    lines += ["", "## Per review", "", "| Review | Included | " + " | ".join(MEASURES[:9]) + " |",
              "|---|---|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['review_id']} | {r['included']} | " + " | ".join(pct(r.get(k)) for k in MEASURES[:9]) + " |")
    return "\n".join(lines) + "\n"


def figure(path: Path, s_all: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("CiteLume", ["pool recall", "top 20", "top 60", "selected"]),
              ("BM25", ["BM25 top 20", "BM25 top 60"]), ("Citations", ["Citations top 20", "Citations top 60"]),
              ("Random", ["Random top 20", "Random top 60"])]
    labels, means, lows, highs, colors = [], [], [], [], []
    palette = {"CiteLume": "#3b5bdb", "BM25": "#868e96", "Citations": "#adb5bd", "Random": "#ced4da"}
    for g, keys in groups:
        for k in keys:
            if k in s_all:
                labels.append(k if g == "CiteLume" else k)
                means.append(100 * s_all[k]["mean"])
                lows.append(100 * (s_all[k]["mean"] - s_all[k]["ci_low"]))
                highs.append(100 * (s_all[k]["ci_high"] - s_all[k]["mean"]))
                colors.append(palette[g])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(means)), means, yerr=[lows, highs], color=colors, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Included studies recalled (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Recall of the reviews' included studies (mean, 95% bootstrap CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.session6")
    p.add_argument("--prisma", type=Path, required=True)
    p.add_argument("--standard", type=Path)
    p.add_argument("--set", choices=("development", "test"), required=True)
    p.add_argument("--name", default="")
    a = p.parse_args(argv)
    reviews = review_split(a.set, session6_test_analysis=(a.set == "test"))
    rows, missing = [], []
    for r in reviews:
        prisma = load_corpus(a.prisma / r["review_id"])
        standard = load_corpus(a.standard / r["review_id"]) if a.standard else None
        if prisma is None:
            missing.append(r["review_id"])
            continue
        rows.append({"review_id": r["review_id"], **review_measures(r, prisma, standard)})
    s_all = summarise(rows, MEASURES)
    s_small = summarise([r for r in rows if r["included"] <= 60], MEASURES)
    name = a.name or f"6_{a.set}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps({"rows": rows, "all": s_all, "at_most_60": s_small,
                                                      "missing": missing}, indent=1), encoding="utf-8")
    md = markdown(name, a.set, rows, s_all, s_small, missing)
    (RESULTS / f"{name}.md").write_text(md, encoding="utf-8")
    figure(RESULTS / f"{name}.png", s_all)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
