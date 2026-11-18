"""
Builds the primary benchmark described in eval/PROTOCOL.md: the 40 Cochrane intervention reviews of CLEF eHealth TAR
2019 (Task 2), each with its question, search date limit and included studies resolved through PubMed, and the
seeded development / test split.

    cd backend && ../.venv/bin/python ../eval/build_benchmark.py      (or any Python 3.10+ with httpx)

Writes, under eval/benchmark/:
    clef2019_intervention.jsonl   one review per line (see `review_record`)
    split.json                    the development and test review IDs, and how they were drawn
    EXCLUDED.md                   reviews left out, with the reason
Downloads (CLEF topic and relevance files, PubMed summaries) are cached in eval/.cache/, which git ignores. Re-running
gives the same files unless PubMed's records change. Only public IDs are sent to PubMed; nothing about the user.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
OUT = ROOT / "benchmark"
CLEF_RAW = "https://raw.githubusercontent.com/CLEF-TAR/tar/master/2019-TAR/Task2"
CLEF_API = "https://api.github.com/repos/CLEF-TAR/tar/contents/2019-TAR/Task2"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PARTS = {"Training": "train", "Testing": "test"}
SEED = 20261118
# The latest version of each review published before 1 July 2019: CLEF assembled the 2019 topics in the first half of
# 2019 and used versions published as late as March 2019 (protocol amendment of 18 November 2026)
DATE_LIMIT_BEFORE = (2019, 7)
MIN_RESOLVED = 2
BENCHMARK_VERSION = "citelume-benchmark/clef2019-intervention/1.0"

_client = httpx.Client(timeout=60.0, headers={"User-Agent": "CiteLume-eval/1.0"}, follow_redirects=True)
_last_ncbi = 0.0


def _get_cached(url: str, name: str) -> str:
    path = CACHE / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    r = _client.get(url)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text, encoding="utf-8")
    return r.text


def _ncbi(endpoint: str, params: Dict[str, str], cache_name: str) -> dict:
    """NCBI E-utilities, at most 3 requests a second (no key), cached."""
    global _last_ncbi
    path = CACHE / "pubmed" / cache_name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    wait = _last_ncbi + 0.4 - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        r = _client.post(f"{EUTILS}/{endpoint}", data={**params, "retmode": "json"})
        _last_ncbi = time.monotonic()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    r.raise_for_status()
    return {}


# ---------------------------------------------------------------------------------------------------------------
# CLEF topics and relevance judgements
# ---------------------------------------------------------------------------------------------------------------
def clef_topics() -> List[Tuple[str, str]]:
    """(review ID, CLEF part) for every intervention topic."""
    out: Dict[str, str] = {}
    for part, short in PARTS.items():
        listing = json.loads(_get_cached(f"{CLEF_API}/{part}/Intervention/topics", f"clef/{short}_topics.json"))
        for x in listing:
            # A few reviews are listed in both parts; each counts once (its training-part topic file is used)
            if re.fullmatch(r"CD\d{6}", x["name"]) and x["name"] not in out:
                out[x["name"]] = short
    return sorted(out.items())


def topic_title(review_id: str, part: str) -> str:
    folder = [k for k, v in PARTS.items() if v == part][0]
    text = _get_cached(f"{CLEF_RAW}/{folder}/Intervention/topics/{review_id}", f"clef/topics/{review_id}")
    m = re.search(r"^Title:\s*(.+?)\s*$", text, flags=re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def content_qrels() -> Dict[str, List[str]]:
    """Included studies (content level, relevance 1) per review, in file order."""
    included: Dict[str, List[str]] = {}
    for part, short in PARTS.items():
        listing = json.loads(_get_cached(f"{CLEF_API}/{part}/Intervention/qrels", f"clef/{short}_qrels.json"))
        for f in listing:
            if "content" not in f["name"]:
                continue
            text = _get_cached(f["download_url"], f"clef/qrels/{f['name']}")
            for line in text.splitlines():
                cols = line.split()
                if len(cols) >= 4 and cols[3] == "1":
                    included.setdefault(cols[0], [])
                    if cols[2] not in included[cols[0]]:
                        included[cols[0]].append(cols[2])
    return included


# ---------------------------------------------------------------------------------------------------------------
# PubMed: review versions and included studies
# ---------------------------------------------------------------------------------------------------------------
def _year(pubdate: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", pubdate or "")
    return int(m.group(0)) if m else None


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                        "dec"], start=1)}


def _year_month(pubdate: str) -> Optional[Tuple[int, int]]:
    year = _year(pubdate)
    if not year:
        return None
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", (pubdate or "").lower())
    return year, (_MONTHS[m.group(1)] if m else 12)  # a year alone counts as its last month (conservative)


def _doi(summary: dict) -> str:
    for a in summary.get("articleids") or []:
        if a.get("idtype") == "doi" and a.get("value"):
            return a["value"].strip().lower()
    m = re.search(r"10\.\S+", summary.get("elocationid") or "")
    return m.group(0).strip().lower() if m else ""


def summaries(pmids: List[str], tag: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i + 200]
        data = _ncbi("esummary.fcgi", {"db": "pubmed", "id": ",".join(chunk)}, f"esummary_{tag}_{i}.json")
        res = data.get("result") or {}
        for u in res.get("uids") or []:
            if "error" not in res[u]:
                out[u] = res[u]
    return out


def review_version(review_id: str) -> Optional[dict]:
    """The latest version of the review in PubMed published before DATE_LIMIT_BEFORE."""
    found = _ncbi("esearch.fcgi", {"db": "pubmed", "term": review_id, "retmax": "50"}, f"esearch_{review_id}.json")
    ids = (found.get("esearchresult") or {}).get("idlist") or []
    versions = []
    for pmid, s in summaries(ids, f"review_{review_id}").items():
        doi = _doi(s)
        if review_id.lower() not in doi or "cochrane" not in (s.get("source") or "").lower():
            continue
        ym = _year_month(s.get("pubdate") or s.get("epubdate") or "")
        if ym and ym < DATE_LIMIT_BEFORE:
            versions.append({"pmid": pmid, "doi": doi, "year": ym[0], "month": ym[1],
                             "pubdate": s.get("pubdate", ""), "title": s.get("title", "")})
    versions.sort(key=lambda v: (v["year"], v["month"], v["doi"]))
    return versions[-1] if versions else None


def review_record(review_id: str, part: str, included: List[str]) -> Tuple[Optional[dict], str]:
    """The benchmark line for one review, or (None, reason) when the protocol excludes it."""
    title = topic_title(review_id, part)
    version = review_version(review_id)
    if not version:
        return None, "no version of the review published before July 2019 was found in PubMed"
    found = summaries(included, f"included_{review_id}")
    studies = []
    for pmid in included:
        s = found.get(pmid)
        if s:
            studies.append({"pmid": pmid, "doi": _doi(s), "title": (s.get("title") or "").strip(),
                            "year": _year(s.get("pubdate") or "")})
    if len(studies) < MIN_RESOLVED:
        return None, f"only {len(studies)} of {len(included)} included studies resolved to a PubMed record"
    return {
        "review_id": review_id,
        "clef_part": part,
        "title": title,
        "question": title,
        "review_version": version,
        "year_to": version["year"],
        "included_listed": len(included),
        "included_resolved": len(studies),
        "included": studies,
    }, ""


def draw_split(review_ids: List[str]) -> Dict[str, List[str]]:
    ordered = sorted(review_ids)
    random.Random(SEED).shuffle(ordered)
    n_dev = len(ordered) // 3
    return {"development": sorted(ordered[:n_dev]), "test": sorted(ordered[n_dev:])}


def main() -> int:
    topics = clef_topics()
    included = content_qrels()
    print(f"{len(topics)} intervention topics; judgements for {len(included)}")
    records, excluded = [], []
    for review_id, part in topics:
        rec, reason = review_record(review_id, part, included.get(review_id, []))
        if rec:
            records.append(rec)
            print(f"  {review_id} ({part}) {rec['year_to']}: {rec['included_resolved']} of {rec['included_listed']} "
                  f"included studies resolved")
        else:
            excluded.append((review_id, reason))
            print(f"  {review_id}: EXCLUDED - {reason}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "clef2019_intervention.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    split = draw_split([r["review_id"] for r in records])
    (OUT / "split.json").write_text(json.dumps({
        "benchmark": BENCHMARK_VERSION, "built": date.today().isoformat(), "seed": SEED,
        "rule": "eligible review IDs sorted, shuffled with random.Random(seed), first third (rounded down) = development",
        "development": split["development"], "test": split["test"]}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Reviews excluded from the benchmark", "",
             "Rules: eval/PROTOCOL.md, section 3. Decided before any run.", ""]
    lines += [f"- {rid}: {why}" for rid, why in excluded] or ["None."]
    (OUT / "EXCLUDED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(records)} reviews ({len(split['development'])} development, {len(split['test'])} test); "
          f"{len(excluded)} excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
