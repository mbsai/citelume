"""
Session 7, E3: how much the selection depends on each weight and cap.

    python -m eval.sensitivity --standard runs/6_test_standard --prisma runs/6_test_prisma --set test

Each weight and cap of services/paper_selection.py is set to half and to one and a half times its value, one at a
time, in memory only (the frozen code is not changed), and selection is re-run on every stored corpus. Against the
frozen selection it reports:
    Standard corpora   overlap (Jaccard) of the 20 selected studies, and Kendall's tau of the ranks of the studies
                       selected both times
    PRISMA corpora     overlap (Jaccard) of the first 300 eligible records in screening order (the records the
                       model would screen; the screening itself is not re-run)
and the change in the share of the reviews' included studies among them. No model, no network.
Writes eval/results/<name>.md and .json.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from eval.scoring import match, review_split  # noqa: E402
from models import Corpus  # noqa: E402
from services import paper_selection as ps  # noqa: E402
from services.paper_selection import SelectionDetails, select_studies  # noqa: E402

SCREEN_CAP = 300
FACTORS = (0.5, 1.5)


def _scaled_dict(d: dict, factor: float, keys=None) -> dict:
    return {k: (v * factor if (keys is None or k in keys) and isinstance(v, (int, float)) else v) for k, v in d.items()}


def variants() -> List[tuple]:
    """(label, context manager factory) for every weight and cap, at each factor."""
    out = []
    for f in FACTORS:
        out += [
            (f"concept-match points x{f}", lambda f=f: mock.patch.object(ps, "RELEVANCE_CONCEPT_POINTS", ps.RELEVANCE_CONCEPT_POINTS * f)),
            (f"wording-overlap points x{f}", lambda f=f: mock.patch.object(ps, "RELEVANCE_TEXT_POINTS", ps.RELEVANCE_TEXT_POINTS * f)),
            (f"study-type points x{f}", lambda f=f: mock.patch.object(
                ps, "EVIDENCE_POINTS", {k: _scaled_dict(v, f) for k, v in ps.EVIDENCE_POINTS.items()})),
            (f"citation-impact points x{f}", lambda f=f: mock.patch.object(ps, "POINTS", _scaled_dict(ps.POINTS, f, {"impact"}))),
            (f"recency points x{f}", lambda f=f: mock.patch.object(ps, "POINTS", _scaled_dict(ps.POINTS, f, {"recency"}))),
            (f"preprint penalty x{f}", lambda f=f: mock.patch.object(ps, "PREPRINT_PENALTY", ps.PREPRINT_PENALTY * f)),
            (f"review share caps x{f}", lambda f=f: contextlib.ExitStack()),  # filled below
            (f"same author-year cap x{f}", lambda f=f: mock.patch.object(
                ps, "MAX_PER_AUTHOR_YEAR", max(1, round(ps.MAX_PER_AUTHOR_YEAR * f)))),
        ]
    fixed = []
    for label, factory in out:
        if label.startswith("review share caps"):
            f = float(label.split("x")[-1])

            def caps(f=f):
                stack = contextlib.ExitStack()
                stack.enter_context(mock.patch.object(ps, "REVIEW_SHARE_CAP", _scaled_dict(ps.REVIEW_SHARE_CAP, f)))
                stack.enter_context(mock.patch.object(ps, "SECONDARY_SHARE_CAP", _scaled_dict(ps.SECONDARY_SHARE_CAP, f)))
                return stack
            fixed.append((label, caps))
        else:
            fixed.append((label, factory))
    return fixed


def load(folder: Path) -> Optional[Corpus]:
    path = folder / "corpus.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return Corpus.model_validate_json(fh.read())


def run_selection(c: Corpus) -> tuple:
    """(selected ids in order, eligible ids in screening order) as selection produces them from the stored records."""
    u = c.plan.selection_understanding or c.plan.question_understanding if c.plan else None
    details = SelectionDetails()
    records = [p.model_copy(deep=True) for p in c.records]
    chosen, _ = select_studies(records, c.ranking_query, c.pool_limit, u, records_found=c.records_found,
                               method=c.selection_method, details=details,
                               reference_year=c.reference_year or None,
                               eligibility=c.protocol.eligibility if c.protocol else None)
    return [p.id for p in chosen], [p.id for p in details.candidates]


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def kendall(a: List[str], b: List[str]) -> Optional[float]:
    from scipy.stats import kendalltau
    common = [x for x in a if x in set(b)]
    if len(common) < 3:
        return None
    pos_b = {x: i for i, x in enumerate(b)}
    tau = kendalltau(list(range(len(common))), [pos_b[x] for x in common]).statistic
    return None if tau != tau else float(tau)


def figure(path: Path, rows: List[dict]) -> None:
    """Median overlap of the Standard selection (and the PRISMA screening set) with the frozen one, per change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["variant"] for r in rows]
    std = [r["standard_jaccard_median"] or 0 for r in rows]
    std_min = [r["standard_jaccard_min"] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = range(len(rows))
    ax.barh(list(y), std, color="#3b5bdb", label="Standard: median overlap of the 20 selected")
    ax.scatter(std_min, list(y), color="#e8590c", zorder=3, s=14, label="Standard: lowest overlap of any review")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Overlap with the frozen selection (Jaccard)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.sensitivity")
    p.add_argument("--standard", type=Path)
    p.add_argument("--prisma", type=Path)
    p.add_argument("--set", choices=("development", "test"), required=True)
    p.add_argument("--name", default="7_sensitivity")
    a = p.parse_args(argv)
    reviews = review_split(a.set, session6_test_analysis=(a.set == "test"))
    corpora = {}
    for r in reviews:
        for mode, folder in (("standard", a.standard), ("prisma", a.prisma)):
            if folder:
                c = load(folder / r["review_id"])
                if c is not None:
                    gold = {s["record_id"] for s in match(r["included"], c.records).found}
                    corpora[(mode, r["review_id"])] = (c, gold, len(r["included"]))
    base = {k: run_selection(c) for k, (c, _, _) in corpora.items()}

    def recall(ids, gold, n):
        return sum(1 for i in ids if i in gold) / n if n else 0.0

    rows = []
    for label, factory in variants():
        std_j, std_tau, pr_j, d_std, d_pr = [], [], [], [], []
        with factory():
            for k, (c, gold, n) in corpora.items():
                sel, elig = run_selection(c)
                bsel, belig = base[k]
                if k[0] == "standard":
                    std_j.append(jaccard(sel, bsel))
                    t = kendall(bsel, sel)
                    if t is not None:
                        std_tau.append(t)
                    d_std.append(recall(sel, gold, n) - recall(bsel, gold, n))
                else:
                    pr_j.append(jaccard(elig[:SCREEN_CAP], belig[:SCREEN_CAP]))
                    d_pr.append(recall(elig[:SCREEN_CAP], gold, n) - recall(belig[:SCREEN_CAP], gold, n))
        med = lambda xs: statistics.median(xs) if xs else None
        mean = lambda xs: statistics.fmean(xs) if xs else None
        rows.append({"variant": label, "standard_jaccard_median": med(std_j), "standard_jaccard_min": min(std_j) if std_j else None,
                     "standard_tau_median": med(std_tau), "standard_recall_change_mean": mean(d_std),
                     "prisma_top300_jaccard_median": med(pr_j), "prisma_top300_jaccard_min": min(pr_j) if pr_j else None,
                     "prisma_recall_change_mean": mean(d_pr)})
    f = lambda x, pct=False: "" if x is None else (f"{100 * x:+.1f} pts" if pct else f"{x:.2f}")
    lines = [f"# Session 7, E3: sensitivity of the selection to its weights and caps ({a.set} set)", "",
             f"{sum(1 for k in corpora if k[0] == 'standard')} Standard and {sum(1 for k in corpora if k[0] == 'prisma')} "
             "PRISMA corpora; each weight or cap set to half and one and a half times its value, one at a time, in "
             "memory; compared with the frozen selection (protocol "
             f"{ps.PROTOCOL_VERSION}). Overlap is Jaccard (1 = identical); tau is Kendall's rank correlation of the "
             "studies selected both times; recall change is the mean change in the share of the reviews' included "
             "studies, in percentage points.", "",
             "| Change | Standard: overlap of the 20 (median, min) | Standard: rank tau (median) | Standard: recall change | "
             "PRISMA: overlap of the first 300 (median, min) | PRISMA: recall change in the first 300 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {f(r['standard_jaccard_median'])}, {f(r['standard_jaccard_min'])} | "
                     f"{f(r['standard_tau_median'])} | {f(r['standard_recall_change_mean'], True)} | "
                     f"{f(r['prisma_top300_jaccard_median'])}, {f(r['prisma_top300_jaccard_min'])} | "
                     f"{f(r['prisma_recall_change_mean'], True)} |")
    md = "\n".join(lines) + "\n"
    out = ROOT / "results"
    (out / f"{a.name}.json").write_text(json.dumps(rows, indent=1))
    (out / f"{a.name}.md").write_text(md)
    figure(out / f"{a.name}.png", rows)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
