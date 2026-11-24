"""
Session 7, E2 (planning): how much the model's search plan for the same question varies between runs.

    CITELUME_LLM_API_KEY=... python -m eval.replan run --prisma runs/6_test_prisma --set test \\
        --model nvidia/nemotron-3-ultra-550b-a55b:free --repeats 3
    python -m eval.replan report --prisma runs/6_test_prisma --set test

Each test question is planned again `--repeats` times with the model (the planner's cache is cleared between
repeats; only planning calls, no searches). Each plan's concept names and PubMed strings are stored in
runs/7_replan/<review>.json (resumable) and compared with the plan stored in the Session 6 run: overlap (Jaccard)
of the concept names (case-folded) and of the PubMed search strings, and whether the plan fell back to rules.
Writes eval/results/7_planning_variation.md and .json.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import statistics
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from eval.scoring import review_split  # noqa: E402
from models import ApiKeysConfig, Corpus  # noqa: E402

OUT = ROOT.parent / "runs" / "7_replan"


def load(folder: Path) -> Optional[Corpus]:
    path = folder / "corpus.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return Corpus.model_validate_json(fh.read())


def summary(plan) -> dict:
    u = plan.question_understanding
    return {"planned_by": (u.planned_by if u else "") or "",
            "concepts": [c.name for c in (u.concepts.search_concepts if u else [])],
            "pubmed": [f.queries.pubmed for f in plan.search_facets if f.queries.pubmed]}


def run(a) -> int:
    from services import query_utils
    cfg = ApiKeysConfig(llm_provider="openrouter", llm_model=a.model,
                        llm_api_key=os.environ.get("CITELUME_LLM_API_KEY", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    for r in review_split(a.set, session6_test_analysis=(a.set == "test")):
        dest = OUT / f"{r['review_id']}.json"
        done = json.loads(dest.read_text())["plans"] if dest.exists() else []
        c = load(a.prisma / r["review_id"])
        if c is None:
            continue
        while len(done) < a.repeats:
            query_utils._DECOMPOSITION_CACHE.clear()
            plan = asyncio.run(query_utils.generate_multi_api_decomposition_plan(c.query, c.mode, cfg))
            s = summary(plan)
            if "model" not in s["planned_by"]:
                print(f"{r['review_id']}: the model did not plan (quota?); stopping here, the next run resumes")
                return 1
            done.append(s)
            dest.write_text(json.dumps({"model": a.model, "plans": done}, indent=0))
        print(f"{r['review_id']}: {len(done)} plans")
    return 0


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def report(a) -> int:
    rows = []
    for r in review_split(a.set, session6_test_analysis=(a.set == "test")):
        dest = OUT / f"{r['review_id']}.json"
        c = load(a.prisma / r["review_id"])
        if c is None or not dest.exists():
            continue
        stored = summary(c.plan)
        # The stored strings carry MeSH alternatives added after planning (services/mesh_terms.py); new plans do not
        alts = [m for x in c.plan.question_understanding.concepts.search_concepts for m in x.mesh]

        def strip(q: str) -> str:
            for m in alts:
                q = q.replace(" OR " + m, "")
            return q
        stored["pubmed"] = [strip(q) for q in stored["pubmed"]]
        plans = json.loads(dest.read_text())["plans"]
        norm = lambda xs: [x.strip().lower() for x in xs]
        cj = [jaccard(norm(p["concepts"]), norm(stored["concepts"])) for p in plans]
        qj = [jaccard(p["pubmed"], stored["pubmed"]) for p in plans]
        pair_c = [jaccard(norm(plans[i]["concepts"]), norm(plans[j]["concepts"]))
                  for i in range(len(plans)) for j in range(i + 1, len(plans))]
        core_same = sum(1 for p in plans if p["pubmed"] and stored["pubmed"] and p["pubmed"][0] == stored["pubmed"][0])
        rows.append({"review_id": r["review_id"], "plans": len(plans), "concepts_vs_stored": statistics.fmean(cj),
                     "core_search_identical": core_same / len(plans),
                     "pubmed_vs_stored": statistics.fmean(qj),
                     "concepts_between_repeats": statistics.fmean(pair_c) if pair_c else None,
                     "identical_concepts_all": all(norm(p["concepts"]) == norm(stored["concepts"]) for p in plans)})
    if not rows:
        print("no plans stored yet")
        return 1
    med = lambda k: statistics.median([x[k] for x in rows if x[k] is not None])
    lines = [f"# Session 7, E2: variation of the model's search plan ({a.set} set)", "",
             f"{len(rows)} questions, {sum(x['plans'] for x in rows)} new plans. Overlap is Jaccard (1 = identical) "
             "of the concept names (case-folded) and of the PubMed search strings, against the plan stored in the "
             "Session 6 run.", "",
             f"- Concept names, new plan vs stored: median overlap {med('concepts_vs_stored'):.2f}",
             f"- Concept names, between new plans: median overlap {med('concepts_between_repeats'):.2f}",
             f"- Core PubMed search identical to the stored one (MeSH alternatives removed): "
             f"{sum(x['core_search_identical'] * x['plans'] for x in rows):.0f} of {sum(x['plans'] for x in rows)} plans",
             f"- All PubMed strings (core, reviews and aspect searches), new plan vs stored: median overlap "
             f"{med('pubmed_vs_stored'):.2f}",
             f"- Questions whose every new plan named exactly the stored concepts: "
             f"{sum(x['identical_concepts_all'] for x in rows)} of {len(rows)}", "",
             "The concept names say what the model takes the question to be about; the PubMed strings also carry the "
             "synonyms it adds and the aspects it chooses, which vary more.", "",
             "| Review | Plans | Concepts vs stored | Concepts between repeats | Core search identical | All PubMed strings vs stored |",
             "|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['review_id']} | {x['plans']} | {x['concepts_vs_stored']:.2f} | "
                     + ("" if x["concepts_between_repeats"] is None else f"{x['concepts_between_repeats']:.2f}")
                     + f" | {100 * x['core_search_identical']:.0f}% | {x['pubmed_vs_stored']:.2f} |")
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "7_planning_variation.md").write_text(md)
    (ROOT / "results" / "7_planning_variation.json").write_text(json.dumps(rows, indent=1))
    print(md)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.replan")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--prisma", type=Path, required=True)
        s.add_argument("--set", choices=("development", "test"), required=True)
        if name == "run":
            s.add_argument("--model", required=True)
            s.add_argument("--repeats", type=int, default=3)
    a = p.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    sys.exit(main())
