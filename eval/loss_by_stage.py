"""
Session 5b: where each development review's included studies are lost in the CiteLume pipeline.

    python -m eval.loss_by_stage runs/5b_dev [--name=5b_loss_by_stage] [--no-pubmed]
    python -m eval.loss_by_stage runs/6_test_prisma --set=test --session6 --name=6_test_loss_by_stage
    (from the repository root; adds backend/ to the path; --name sets the eval/results/ file names)

Reads, for each development review, the batch runner's raw.json.gz (records before duplicates were merged) and
corpus.json.gz, and follows every included study through the stages, in order:

    found         in any record the searches returned (database searches; citation chasing and review reference
                  lists, "other methods", counted apart)
    merged        still present after duplicates were merged (a wrong merge can drop a study's identifiers)
    eligible      passed the eligibility rules (otherwise: the rule that excluded it)
    screened in   kept by the model's title-and-abstract screening (or never screened because the limit filled)
    included      in the final included set

and, alongside, whether it was ranked in the top 20 and top 60 of the eligible records. For studies the searches
missed, it asks PubMed whether any of CiteLume's own PubMed queries matches the study at all (the query run without
a result limit): "matched by a query" means the study was beyond the number of records fetched (search depth);
"matched by no query" means the queries were not sensitive enough. All included studies are in PubMed by
construction (eval/build_benchmark.py), so PubMed coverage itself is not the ceiling.

Writes eval/results/<name>.md and .json. Development reviews unless --set=test --session6 is given (eval/PROTOCOL.md
section 4: test reviews are for the Session 6 analysis only).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from eval.scoring import match, review_split  # noqa: E402
from models import Corpus, PaperRecord  # noqa: E402

CHASED = {"Citation searching", "Reference lists of reviews"}   # identified by other methods
STAGES = ("found", "merged", "eligible", "screened_in", "included")
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache" / "pubmed_queries"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
RETMAX = 9999


def _load(folder: Path):
    with gzip.open(folder / "corpus.json.gz", "rt", encoding="utf-8") as fh:
        corpus = Corpus.model_validate_json(fh.read())
    with gzip.open(folder / "raw.json.gz", "rt", encoding="utf-8") as fh:
        raw = [PaperRecord.model_validate(r) for r in json.load(fh)]
    return corpus, raw


def follow(included: List[dict], corpus: Corpus, raw: List[PaperRecord]) -> List[dict]:
    """One row per included study: the last stage it reached and why it went no further."""
    db_raw = [r for r in raw if r.source not in CHASED]
    chased_raw = [r for r in raw if r.source in CHASED]
    found_db = {s["pmid"] for s in match(included, db_raw).found}
    found_chase = {s["pmid"] for s in match(included, chased_raw).found}
    in_records = {s["pmid"]: s["record_id"] for s in match(included, corpus.records).found}
    ranks = {c.record_id: c.rank for c in corpus.candidates}
    report = corpus.selection_report
    decisions = {d.record_id: d for d in (report.screening_decisions if report else [])}
    screening_ran = bool(decisions)
    included_ids = set(corpus.included_ids)
    rows = []
    for s in included:
        row = {"pmid": s["pmid"], "title": s.get("title", ""), "year": s.get("year"),
               "found_by_search": s["pmid"] in found_db, "found_by_chasing": s["pmid"] in found_chase,
               "rank": None, "stage": "missed", "reason": ""}
        rid = in_records.get(s["pmid"])
        if not (row["found_by_search"] or row["found_by_chasing"]):
            row["reason"] = "not returned by any search"
        elif rid is None:
            row.update(stage="found", reason="lost when duplicates were merged")
        elif rid not in ranks:
            row.update(stage="merged", reason=corpus.excluded.get(rid, "not eligible (no reason recorded)"))
        else:
            row["rank"] = ranks[rid]
            d = decisions.get(rid)
            if screening_ran and d is None and rid not in included_ids:
                row.update(stage="eligible", reason="not screened: the study limit was filled first")
            elif d is not None and not d.effective_include:
                row.update(stage="eligible", reason=f"screened out: {d.failed_criterion or d.reason}")
            elif rid not in included_ids:
                row.update(stage="screened_in", reason="beyond the study limit")
            else:
                row["stage"] = "included"
        rows.append(row)
    return rows


def reached(rows: List[dict]) -> Dict[str, int]:
    order = ("missed",) + STAGES
    return {st: sum(1 for r in rows if order.index(r["stage"]) >= order.index(st)) for st in STAGES}


# ---------------------------------------------------------------------------------------------------------------
# The PubMed query check for studies the searches missed
# ---------------------------------------------------------------------------------------------------------------
def pubmed_queries(corpus: Corpus, year_to: Optional[int]) -> List[str]:
    from services.pubmed_client import _clean_search_query

    out: List[str] = []
    for facet in (corpus.plan.search_facets if corpus.plan else []):
        q = getattr(facet.queries, "pubmed", None)
        for item in (q if isinstance(q, list) else [q]):
            if item:
                term = _clean_search_query(item, max_terms=5)
                if year_to:
                    term = f"({term}) AND (1900:{year_to}[dp])"
                if term not in out:
                    out.append(term)
    return out


_last = 0.0


def esearch_all(term: str, client: httpx.Client) -> Dict[str, object]:
    """Every PubMed ID the term matches (up to 9,999) and the full count; cached by term."""
    global _last
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (hashlib.sha1(term.encode("utf-8")).hexdigest() + ".json")
    if key.exists():
        cached = json.loads(key.read_text())
        if cached.get("term") == term:
            return cached
    wait = _last + 0.4 - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        r = client.post(EUTILS, data={"db": "pubmed", "term": term, "retmax": str(RETMAX), "retmode": "json"})
        _last = time.monotonic()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        res = r.json().get("esearchresult", {})
        out = {"term": term, "count": int(res.get("count", 0)), "ids": res.get("idlist", [])}
        key.write_text(json.dumps(out))
        return out
    return {"term": term, "count": -1, "ids": []}


def query_check(rows: List[dict], corpus: Corpus, year_to: Optional[int], client: httpx.Client) -> Dict[str, object]:
    missed = [r for r in rows if r["stage"] == "missed"]
    terms = pubmed_queries(corpus, year_to)
    matched: Set[str] = set()
    truncated = 0
    for t in terms:
        res = esearch_all(t, client)
        matched |= set(res["ids"])
        truncated += int(res["count"] > RETMAX)
    for r in missed:
        r["matched_by_pubmed_query"] = r["pmid"] in matched
    return {"pubmed_queries": len(terms), "queries_over_9999": truncated,
            "missed": len(missed), "missed_matched_by_a_query": sum(1 for r in missed if r["matched_by_pubmed_query"])}


# ---------------------------------------------------------------------------------------------------------------
def analyse(runs: Path, check_queries: bool = True, which: str = "development", session6: bool = False) -> dict:
    reviews = []
    client = httpx.Client(timeout=60.0, headers={"User-Agent": "CiteLume-eval/1.0"}) if check_queries else None
    for review in review_split(which, session6_test_analysis=session6):
        folder = runs / review["review_id"]
        if not (folder / "raw.json.gz").exists():
            reviews.append({"review_id": review["review_id"], "status": "not run"})
            continue
        corpus, raw = _load(folder)
        rows = follow(review["included"], corpus, raw)
        entry = {"review_id": review["review_id"], "status": "done", "title": review["title"],
                 "included": len(rows), "records_found": len(raw), "records_merged": len(corpus.records),
                 "eligible": len(corpus.candidates), "selected": len(corpus.included_ids),
                 "reached": reached(rows), "top20": sum(1 for r in rows if r["rank"] and r["rank"] <= 20),
                 "top60": sum(1 for r in rows if r["rank"] and r["rank"] <= 60),
                 "found_only_by_chasing": sum(1 for r in rows if r["found_by_chasing"] and not r["found_by_search"]),
                 "loss_reasons": dict(Counter(r["reason"].split(":")[0] if r["reason"].startswith("screened out")
                                              else r["reason"] for r in rows if r["stage"] != "included")),
                 "studies": rows}
        if client is not None:
            entry["query_check"] = query_check(rows, corpus, review["year_to"], client)
        reviews.append(entry)
    done = [r for r in reviews if r["status"] == "done"]
    total = {st: sum(r["reached"][st] for r in done) for st in STAGES}
    n = sum(r["included"] for r in done)
    losses = {"search": n - total["found"], "duplicate merging": total["found"] - total["merged"],
              "eligibility": total["merged"] - total["eligible"],
              "screening and limit": total["eligible"] - total["included"]}
    return {"set": which, "reviews": reviews, "included_studies": n, "reached": total, "losses": losses,
            "largest_loss": max(losses, key=losses.get) if done else "",
            "top20": sum(r["top20"] for r in done), "top60": sum(r["top60"] for r in done),
            "found_only_by_chasing": sum(r["found_only_by_chasing"] for r in done),
            "missed_matched_by_a_query": sum(r.get("query_check", {}).get("missed_matched_by_a_query", 0) for r in done)}


def pct(a: int, b: int) -> str:
    return f"{a} ({100 * a / b:.0f}%)" if b else "0"


def markdown(res: dict, runs: Path) -> str:
    done = [r for r in res["reviews"] if r["status"] == "done"]
    n = res["included_studies"]
    lines = [f"# Where included studies are lost ({res.get('set', 'development')} set)", "",
             f"Runs: `{runs}`; {len(done)} of {len(res['reviews'])} development reviews; {n} included studies. "
             "Deep Research, systematic-style (PRISMA) review type, citation chasing on. Stages are cumulative: "
             "each column counts the included studies still present after that stage.", "",
             "| Review | Included | Found | Merged | Eligible | Screened in | Selected | Top 20 | Top 60 | "
             "Missed but matched by a PubMed query |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in done:
        q = r.get("query_check", {})
        lines.append(f"| {r['review_id']} | {r['included']} | " + " | ".join(str(r["reached"][s]) for s in STAGES)
                     + f" | {r['top20']} | {r['top60']} | {q.get('missed_matched_by_a_query', '')} of "
                       f"{q.get('missed', '')} |")
    t = res["reached"]
    lines.append(f"| **All** | **{n}** | " + " | ".join(f"**{pct(t[s], n)}**" for s in STAGES)
                 + f" | {pct(res['top20'], n)} | {pct(res['top60'], n)} | |")
    total_selected = sum(r["selected"] for r in done)
    lines += ["", f"Studies selected in all: {total_selected}; of these, {t['included']} are included by the reviews "
                  f"(precision {100 * t['included'] / total_selected:.0f}%)." if total_selected else ""]
    lines += ["", "## Losses by stage", "", "| Stage | Included studies lost | Share of all included |", "|---|---|---|"]
    for stage, lost in res["losses"].items():
        lines.append(f"| {stage} | {lost} | {100 * lost / n:.0f}% |" if n else f"| {stage} | {lost} | |")
    lines += ["", f"**Largest loss: {res['largest_loss']}.**", ""]
    missed = n - t["found"]
    lines.append(f"Of the {missed} included studies no search returned, {res['missed_matched_by_a_query']} are matched "
                 "by at least one of CiteLume's own PubMed queries when it is run without a result limit (a "
                 "search-depth loss); the rest are matched by none of them (a query-sensitivity loss).")
    lines.append(f"Other methods (citation chasing, reference lists of reviews) found {res['found_only_by_chasing']} "
                 "included studies that no database search returned.")
    reasons: Counter = Counter()
    for r in done:
        reasons.update(r["loss_reasons"])
    lines += ["", "## Why studies were lost", "", "| Reason | Studies |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    runs = Path(argv[1])
    which = "test" if "--set=test" in argv else "development"
    res = analyse(runs, check_queries="--no-pubmed" not in argv, which=which, session6="--session6" in argv)
    stem = next((a.split("=", 1)[1] for a in argv if a.startswith("--name=")), "5b_loss_by_stage")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stem}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"{stem}.md").write_text(markdown(res, runs), encoding="utf-8")
    print(markdown(res, runs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
