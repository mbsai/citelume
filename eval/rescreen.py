"""
Session 7, E2 (screening): the same records screened again, by the same model (repeats) and by other models.

    python -m eval.rescreen run --prisma runs/6_test_prisma --set test --label qwen-local \\
        --provider ollama --model qwen3.5:9b
    python -m eval.rescreen run ... --label nemotron-r1 --provider openrouter \\
        --model nvidia/nemotron-3-ultra-550b-a55b:free          (key from CITELUME_LLM_API_KEY)
    python -m eval.rescreen report --prisma runs/6_test_prisma --set test

For each review, a fixed sample of the records the stored run's model screened: every screened record that the
review included, then others drawn at random (seed 20261118) up to 96. They are screened again with the stored
question and criteria (services.ai_screening, the frozen screening prompt), and the decisions are written to
runs/7_rescreen/<label>/<review>.json. Resumable: a review with a stored file is skipped.

The report compares every label with the stored decisions and with each other: agreement (share of the same
include / exclude decision), Cohen's kappa, and sensitivity on the reviews' included studies. Writes
eval/results/7_screening_agreement.md and .json.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from eval.scoring import match, review_split  # noqa: E402
from models import ApiKeysConfig, Corpus  # noqa: E402
from services.ai_screening import SCREEN_BATCH, _screen_batch, screening_criteria  # noqa: E402

SEED = 20261118
SAMPLE = 96
OUT = ROOT.parent / "runs" / "7_rescreen"


def load(folder: Path) -> Optional[Corpus]:
    path = folder / "corpus.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return Corpus.model_validate_json(fh.read())


def sample(c: Corpus, gold_ids: set) -> List[str]:
    screened = [d.record_id for d in c.selection_report.screening_decisions if d.screened]
    first = [r for r in screened if r in gold_ids]
    rest = sorted(r for r in screened if r not in gold_ids)
    random.Random(SEED).shuffle(rest)
    return first + rest[:max(0, SAMPLE - len(first))]


async def rescreen_review(c: Corpus, ids: List[str], cfg: ApiKeysConfig) -> Dict[str, dict]:
    recs = {p.id: p for p in c.records}
    batch_records = [recs[i] for i in ids if i in recs]
    criteria = screening_criteria(c.protocol, c.selection_method)
    out: Dict[str, dict] = {}
    for i in range(0, len(batch_records), SCREEN_BATCH):
        batch = batch_records[i:i + SCREEN_BATCH]
        decisions = await _screen_batch(c.query, batch, cfg, criteria)
        for p in batch:
            d = (decisions or {}).get(p.id)
            out[p.id] = {"screened": bool(d and d.screened), "include": bool(d.include) if d else None,
                         "answers": list(d.answers) if d else []}
    return out


def run(a) -> int:
    cfg = ApiKeysConfig(llm_provider=a.provider, llm_model=a.model,
                        llm_api_key=os.environ.get("CITELUME_LLM_API_KEY", "") if a.provider != "ollama" else "",
                        llm_base_url=a.base_url or "")
    folder = OUT / a.label
    folder.mkdir(parents=True, exist_ok=True)
    for r in review_split(a.set, session6_test_analysis=(a.set == "test")):
        dest = folder / f"{r['review_id']}.json"
        if dest.exists():
            continue
        c = load(a.prisma / r["review_id"])
        if c is None or not c.selection_report or not c.selection_report.screening_decisions:
            continue
        gold = {s["record_id"] for s in match(r["included"], c.records).found}
        ids = sample(c, gold)
        if not ids:
            continue
        decisions = asyncio.run(rescreen_review(c, ids, cfg))
        unscreened = sum(1 for d in decisions.values() if not d["screened"])
        if unscreened > 0.05 * len(decisions):
            print(f"{r['review_id']}: {unscreened} of {len(decisions)} not screened; not saved (retry later)")
            continue
        dest.write_text(json.dumps({"model": f"{a.provider}:{a.model}", "decisions": decisions}, indent=0))
        print(f"{r['review_id']}: {len(decisions)} records screened")
    return 0


def kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def report(a) -> int:
    reviews = review_split(a.set, session6_test_analysis=(a.set == "test"))
    labels = sorted(p.name for p in OUT.iterdir() if p.is_dir() and not p.name.startswith(".")) if OUT.exists() else []
    decisions: Dict[str, Dict[str, Dict[str, bool]]] = {"stored": {}}
    gold_all: Dict[str, set] = {}
    for r in reviews:
        c = load(a.prisma / r["review_id"])
        if c is None or not c.selection_report:
            continue
        gold_all[r["review_id"]] = {s["record_id"] for s in match(r["included"], c.records).found}
        decisions["stored"][r["review_id"]] = {d.record_id: d.include for d in c.selection_report.screening_decisions
                                               if d.screened}
    for lab in labels:
        decisions[lab] = {}
        for f in (OUT / lab).glob("CD*.json"):  # not the drive's ._ metadata files
            d = json.loads(f.read_text())["decisions"]
            decisions[lab][f.stem] = {k: v["include"] for k, v in d.items() if v["screened"]}
    names = ["stored"] + labels
    flat = {n: {(rid, rec): inc for rid, dec in decisions[n].items() for rec, inc in dec.items()} for n in names}
    pairs = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            common = sorted(set(flat[x]) & set(flat[y]))
            if common:
                xs, ys = [flat[x][k] for k in common], [flat[y][k] for k in common]
                pairs.append({"a": x, "b": y, "records": len(common),
                              "agreement": sum(p == q for p, q in zip(xs, ys)) / len(common), "kappa": kappa(xs, ys)})
    sens = []
    for lab in names:
        kept = total = 0
        for rid, dec in decisions[lab].items():
            for rec, inc in dec.items():
                if rec in gold_all.get(rid, set()) and (lab == "stored" or rec in decisions["stored"].get(rid, {})):
                    total += 1
                    kept += inc
        sens.append({"label": lab, "included_studies_screened": total, "sensitivity": kept / total if total else None})
    lines = [f"# Session 7, E2: screening agreement on the same records ({a.set} set)", "",
             f"Same records (a fixed sample of up to {SAMPLE} screened records per review, every screened included "
             "study first), same question and criteria, frozen screening prompt. 'stored' is the decision in the "
             "Session 6 run.", "", "| A | B | Records | Agreement | Cohen's kappa |", "|---|---|---|---|---|"]
    for p in pairs:
        lines.append(f"| {p['a']} | {p['b']} | {p['records']} | {100 * p['agreement']:.0f}% | "
                     + ("" if p["kappa"] is None else f"{p['kappa']:.2f}") + " |")
    lines += ["", "| Screened by | Included studies screened | Sensitivity |", "|---|---|---|"]
    for s in sens:
        lines.append(f"| {s['label']} | {s['included_studies_screened']} | "
                     + ("" if s["sensitivity"] is None else f"{100 * s['sensitivity']:.0f}%") + " |")
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "7_screening_agreement.json").write_text(json.dumps({"pairs": pairs, "sensitivity": sens}, indent=1))
    (ROOT / "results" / "7_screening_agreement.md").write_text(md)
    print(md)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.rescreen")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--prisma", type=Path, required=True)
        s.add_argument("--set", choices=("development", "test"), required=True)
        if name == "run":
            s.add_argument("--label", required=True)
            s.add_argument("--provider", required=True)
            s.add_argument("--model", required=True)
            s.add_argument("--base-url", default="")
    a = p.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    sys.exit(main())
