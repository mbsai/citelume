"""
Session 8 (E4): build the blinded annotation sheet of claims, and score it once filled in.

    python -m eval.annotation_sheet build --results runs/8_gates runs/8_dev_standard --n 200 --second 50
    python -m eval.annotation_sheet score

A claim is a sentence of the narrative (executive answer and sections) that cites at least one study. For each
question two narratives are compared: the one shown to the user (after the quality gate, "gated") and the one the
model wrote before the gate acted on it ("ungated", SynthesisEvidencePackage.ungated_narrative). Claims fall into
three strata: only in the ungated narrative (removed by the gate), only in the gated one (added or rewritten by the
gate) and in both. Every claim of the first two strata is sampled, then claims in both are drawn at random (seed
20261118) to reach --n. Each sampled claim is weighted by its stratum's sampling fraction, so error rates can be
estimated for each narrative as a whole.

The sheet (eval/annotation/sheet.csv) shows, for each claim, the question, the claim, and for each cited study its
title, year and the two abstract sentences that share most words with the claim. It does not show which narrative
the claim came from; that is in eval/annotation/key.json. --second claims (drawn at random from the sample) are also
written to sheet_second.csv for a second annotator. Annotators fill in "judgement" with one of: supported, partly,
not supported, cannot tell (see eval/annotation/GUIDE.md).

score: reads the filled sheets and the key and reports, for each narrative, the weighted share of claims judged not
supported (and partly or not supported), with 95% bootstrap intervals over questions, and Cohen's kappa between
annotators. Writes eval/results/8_gates.md and .json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
ANN = ROOT / "annotation"
SEED = 20261118
JUDGEMENTS = ("supported", "partly", "not supported", "cannot tell")
_CITE = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", re.sub(r"[#*_>`]", " ", text or "")).strip()
    # A citation after the full stop belongs to that sentence: "X. [3]" -> "X [3]."
    text = re.sub(r"([.!?])\s*(\[\d+(?:\s*[-–,]\s*\d+)*\])", r" \2\1", text)
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def cited(sentence: str) -> List[int]:
    out: List[int] = []
    for group in _CITE.findall(sentence):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[-–]", part):
                a, b = [int(x) for x in re.split(r"\s*[-–]\s*", part)[:2]]
                out += list(range(a, b + 1)) if 0 < b - a < 30 else [a, b]
            elif part.strip().isdigit():
                out.append(int(part))
    return list(dict.fromkeys(out))


def claims_of(executive: str, sections: List[dict]) -> List[str]:
    texts = [executive] + [s.get("content", "") for s in sections]
    return list(dict.fromkeys(s for t in texts for s in sentences(t) if cited(s)))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_for(claim: str, paper: dict, k: int = 2) -> str:
    words = set(re.findall(r"[a-z0-9]{3,}", claim.lower()))
    sents = sentences(paper.get("abstract") or "")
    best = sorted(sents, key=lambda s: -len(words & set(re.findall(r"[a-z0-9]{3,}", s.lower()))))[:k]
    keep = [s for s in sents if s in best]  # in the abstract's own order
    return " ... ".join(keep) if keep else "(no abstract available)"


def build(a) -> int:
    rng = random.Random(SEED)
    units, strata = [], defaultdict(list)
    for folder in a.results:
        for res_path in sorted(folder.glob("*/result.json")):
            r = json.loads(res_path.read_text())
            u = (r.get("synthesis_evidence") or {}).get("ungated_narrative") or {}
            if not u.get("model_written"):
                continue
            gated = claims_of(r.get("executive_answer", ""), r.get("sections") or [])
            ungated = claims_of(u.get("executive_answer", ""), u.get("sections") or [])
            g, n = {_norm(c): c for c in gated}, {_norm(c): c for c in ungated}
            papers = {p.get("citation_index"): p for p in r.get("papers") or []}
            qid = res_path.parent.name
            for key, text in {**n, **g}.items():
                stratum = "both" if (key in g and key in n) else ("ungated_only" if key in n else "gated_only")
                unit = {"question_id": qid, "question": r.get("query", ""), "claim": text, "stratum": stratum,
                        "in_gated": key in g, "in_ungated": key in n,
                        "refs": cited(text), "papers": {i: papers[i] for i in cited(text) if i in papers}}
                strata[stratum].append(unit)
    if not any(strata.values()):
        print("no results with a model-written narrative yet")
        return 1
    sample = list(strata["ungated_only"]) + list(strata["gated_only"])
    both = list(strata["both"])
    rng.shuffle(both)
    take = max(0, a.n - len(sample))
    sample += both[:take]
    weights = {"ungated_only": 1.0, "gated_only": 1.0,
               "both": (len(strata["both"]) / min(take, len(strata["both"]))) if strata["both"] and take else 0.0}
    rng.shuffle(sample)
    ANN.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for i, u in enumerate(sample, start=1):
        uid = f"C{i:03d}"
        key[uid] = {"question_id": u["question_id"], "stratum": u["stratum"], "in_gated": u["in_gated"],
                    "in_ungated": u["in_ungated"], "weight": weights[u["stratum"]]}
        studies = " || ".join(f"[{ref}] {p.get('title', '')} ({p.get('year') or 'n.d.'}): {evidence_for(u['claim'], p)}"
                              for ref, p in u["papers"].items()) or "(cited number not found among the studies)"
        rows.append({"unit_id": uid, "question": u["question"], "claim": u["claim"],
                     "cited_studies_and_evidence": studies, "judgement": "", "notes": ""})
    fields = list(rows[0])
    for name, subset in (("sheet.csv", rows), ("sheet_second.csv", rng.sample(rows, min(a.second, len(rows))))):
        with (ANN / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(subset, key=lambda x: x["unit_id"]))
    (ANN / "key.json").write_text(json.dumps({"seed": SEED, "strata_sizes": {k: len(v) for k, v in strata.items()},
                                              "sampled": {k: sum(1 for x in key.values() if x["stratum"] == k)
                                                          for k in strata}, "units": key}, indent=1))
    print(f"strata: { {k: len(v) for k, v in strata.items()} }; sampled {len(rows)}; "
          f"second annotator {min(a.second, len(rows))}; written to {ANN}")
    return 0


def read_sheet(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["unit_id"]: r["judgement"].strip().lower() for r in csv.DictReader(fh) if r["judgement"].strip()}


def kappa(x: List[str], y: List[str]) -> Optional[float]:
    n = len(x)
    if not n:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    cats = set(x) | set(y)
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(a) -> int:
    key = json.loads((ANN / "key.json").read_text())["units"]
    first, second = read_sheet(ANN / "sheet.csv"), read_sheet(ANN / "sheet_second.csv")
    bad = [j for j in list(first.values()) + list(second.values()) if j not in JUDGEMENTS]
    if bad:
        print(f"unrecognised judgements: {sorted(set(bad))}; use one of {JUDGEMENTS}")
        return 1

    from eval.scoring import load_split
    split = load_split()
    which_set = {q: s for s in ("development", "test") for q in split[s]}

    def rate(arm: str, judged: Dict[str, str], harsh: bool, only_set: Optional[str] = None) -> tuple:
        by_q: Dict[str, List[tuple]] = defaultdict(list)
        for uid, j in judged.items():
            k = key[uid]
            if not k[f"in_{arm}"] or j == "cannot tell":
                continue
            if only_set and which_set.get(k["question_id"]) != only_set:
                continue
            wrong = j == "not supported" or (harsh and j == "partly")
            by_q[k["question_id"]].append((k["weight"], wrong))

        def est(groups):
            w = sum(wt for g in groups for wt, _ in g)
            return sum(wt for g in groups for wt, bad_ in g if bad_) / w if w else None
        qs = list(by_q.values())
        point = est(qs)
        rng = random.Random(SEED)
        boots = sorted(x for x in (est(rng.choices(qs, k=len(qs))) for _ in range(5000)) if x is not None) if qs else []
        ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (None, None)
        return point, ci, sum(len(g) for g in qs)

    common = sorted(set(first) & set(second))
    kap = kappa([first[u] for u in common], [second[u] for u in common])
    gate_changed = any(k["in_gated"] != k["in_ungated"] for k in key.values())
    lines = ["# Session 8, E4: are the claims supported by the studies they cite?", "",
             f"{len(first)} claims judged by the first annotator, {len(second)} by the second ({len(common)} by both). "
             "Rates are weighted by each stratum's sampling fraction; 95% intervals are bootstrap intervals over "
             "questions. 'cannot tell' is left out.", "",
             "| Claims | Judged | Not supported (95% CI) | Partly or not supported (95% CI) |",
             "|---|---|---|---|"]
    out = {}
    f = lambda v: "" if v is None else f"{100 * v:.1f}%"
    rows = [("gated", "As shown to the reader", None), ("gated", "  test reviews", "test"),
            ("gated", "  development reviews", "development")]
    if gate_changed:  # the gates changed some narratives: report the model's own narrative too
        rows.insert(0, ("ungated", "As the model wrote it (before the gate)", None))
    for arm, label, only in rows:
        p1, c1, n = rate(arm, first, False, only)
        p2, c2, _ = rate(arm, first, True, only)
        out[label.strip()] = {"not_supported": p1, "ci": c1, "partly_or_not": p2, "ci2": c2, "claims": n}
        lines.append(f"| {label} | {n} | {f(p1)} ({f(c1[0])}-{f(c1[1])}) | {f(p2)} ({f(c2[0])}-{f(c2[1])}) |")
    if not gate_changed:
        lines += ["", "The verification gates changed none of the sampled narratives, so the model's own narrative and "
                      "the one shown are the same claims; only the shown narrative is reported."]
    lines += ["", f"Agreement between annotators on {len(common)} claims: "
              + ("not available" if kap is None else f"Cohen's kappa {kap:.2f}, raw agreement "
                 f"{100 * sum(first[u] == second[u] for u in common) / len(common):.0f}%") + "."]
    md = "\n".join(lines) + "\n"
    # The judgements alone (no abstract text), kept with the results
    with (ANN / "judgements.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["unit_id", "first_annotator", "second_annotator"])
        for uid in sorted(set(first) | set(second)):
            w.writerow([uid, first.get(uid, ""), second.get(uid, "")])
    (ROOT / "results" / "8_gates.md").write_text(md)
    (ROOT / "results" / "8_gates.json").write_text(json.dumps({"arms": out, "kappa": kap, "double_coded": len(common)},
                                                              indent=1))
    print(md)
    return 0


def activity(a) -> int:
    """How often each verification gate changed what the reader sees, over the analysis results."""
    counts = defaultdict(int)
    notes = defaultdict(int)
    for folder in a.results:
        for res_path in sorted(folder.glob("*/result.json")):
            r = json.loads(res_path.read_text())
            se = r.get("synthesis_evidence") or {}
            u = se.get("ungated_narrative") or {}
            if not u.get("model_written"):
                continue
            counts["questions"] += 1
            shown = claims_of(r.get("executive_answer", ""), r.get("sections") or [])
            before = claims_of(u.get("executive_answer", ""), u.get("sections") or [])
            counts["claims shown"] += len(shown)
            counts["narratives changed by the section gate"] += shown != before
            counts["structured claims kept"] += len(se.get("claims") or [])
            counts["structured claims excluded by the claim gate"] += len(se.get("excluded_claims") or [])
            for w in r.get("warnings") or []:
                if "corrective retr" in w:
                    counts["narratives that needed a corrective retry"] += 1
                if w.startswith("Quality notes on the synthesis"):
                    counts["narratives with soft quality notes (kept)"] += 1
                    notes[re.sub(r"\[\d+(?:,\s*\d+)*\]", "[n]", w.split(":", 1)[1].strip())[:90]] += 1
                if "replaced by the rule-based synthesis" in w:
                    counts["narratives replaced by the rule-based one"] += 1
                if "could not be traced to a study record were removed" in w:
                    counts["narratives with untraceable figures removed"] += 1
    lines = ["# Session 8: how often the verification gates acted", "",
             f"Over {counts['questions']} model-written analyses ({', '.join(str(f) for f in a.results)}); counted from "
             "the stored results (`python -m eval.annotation_sheet activity`).", "", "| | Count |", "|---|---|"]
    for k in ("questions", "claims shown", "narratives changed by the section gate",
              "narratives with untraceable figures removed", "narratives replaced by the rule-based one",
              "structured claims kept", "structured claims excluded by the claim gate",
              "narratives that needed a corrective retry", "narratives with soft quality notes (kept)"):
        lines.append(f"| {k} | {counts[k]} |")
    if notes:
        lines += ["", "Soft quality notes:", ""] + [f"- {v} x {k}" for k, v in sorted(notes.items(), key=lambda x: -x[1])]
    md = "\n".join(lines) + "\n"
    (ROOT / "results" / "8_gate_activity.md").write_text(md)
    (ROOT / "results" / "8_gate_activity.json").write_text(json.dumps({"counts": counts, "notes": notes}, indent=1))
    print(md)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.annotation_sheet")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--results", type=Path, nargs="+", required=True)
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--second", type=int, default=50)
    sub.add_parser("score")
    act = sub.add_parser("activity")
    act.add_argument("--results", type=Path, nargs="+", required=True)
    a = p.parse_args(argv)
    return {"build": build, "score": score, "activity": activity}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
