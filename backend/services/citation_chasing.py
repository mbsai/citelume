"""
Citation chasing (PRISMA 2020: "identification of studies via other methods", citation searching).

From the studies selected by the database searches (the seeds), fetch what each cites (backward) and what cites it
(forward) from Semantic Scholar, in batches. A paper linked to at least MIN_LINKS seeds that is not already in the
candidate pool is added as a record "identified by other methods". Added records then go through the same
eligibility rules, scoring and (in systematic-style reviews) screening as searched records; nothing is included
just because it was chased.

The reference data fetched for the seeds is also kept on their records, so the literature map can draw citation
links between included studies. Two batch requests per 20 seeds and one per 500 candidates keep the run within the
Semantic Scholar allowance; keyed requests are paced like every other Semantic Scholar call.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import httpx

from models import CitationChasingReport, PaperRecord
from services.pubmed_client import fetch_pubmed_details
from services.source_budget import pace_semantic_scholar
from services.source_health import report_source_failure, retrying_transport

CHASING_VERSION = "citelume-chasing/1.0"
SOURCE_LABEL = "Citation searching"
MIN_LINKS = 2
MAX_SEEDS = 60
MAX_ADDED = 100
SEEDS_PER_REQUEST = 20
DETAILS_PER_REQUEST = 500
# Links kept per seed for the map (a highly cited trial can have thousands of citing papers)
MAP_LINKS_PER_SEED = 200
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
LINK_FIELDS = "paperId,externalIds,references.paperId,citations.paperId"
DETAIL_FIELDS = ("paperId,title,abstract,year,venue,authors,externalIds,citationCount,publicationTypes,url,"
                 "openAccessPdf")


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def seed_identifier(p: PaperRecord) -> Optional[str]:
    """How Semantic Scholar's batch endpoint can find a record: DOI, PubMed ID, arXiv ID or its own ID."""
    if p.doi:
        return f"DOI:{p.doi.strip()}"
    if p.pmid:
        return f"PMID:{p.pmid.strip()}"
    if p.arxiv_id:
        return f"ARXIV:{re.sub(r'v[0-9]+$', '', p.arxiv_id.strip())}"
    if p.s2_id and re.fullmatch(r"[0-9a-f]{40}", p.s2_id.strip()):
        return p.s2_id.strip()
    return None


def pool_keys(records: Iterable[PaperRecord]) -> Set[str]:
    keys: Set[str] = set()
    for p in records:
        if p.doi:
            keys.add(f"doi:{p.doi.strip().lower()}")
        if p.pmid:
            keys.add(f"pmid:{p.pmid.strip()}")
        if p.arxiv_id:
            keys.add(f"arxiv:{re.sub(r'v[0-9]+$', '', p.arxiv_id.strip().lower())}")
        if p.s2_id:
            keys.add(f"s2:{p.s2_id.strip().lower()}")
        t = _norm_title(p.title)
        if len(t) >= 20:
            keys.add(f"title:{t}")
    return keys


def _item_keys(item: dict) -> Set[str]:
    ext = item.get("externalIds") or {}
    keys = {f"s2:{str(item.get('paperId') or '').lower()}"}
    if ext.get("DOI"):
        keys.add(f"doi:{str(ext['DOI']).strip().lower()}")
    if ext.get("PubMed"):
        keys.add(f"pmid:{str(ext['PubMed']).strip()}")
    if ext.get("ArXiv"):
        keys.add(f"arxiv:{str(ext['ArXiv']).strip().lower()}")
    t = _norm_title(item.get("title") or "")
    if len(t) >= 20:
        keys.add(f"title:{t}")
    return keys


async def _post_batch(ids: Sequence[str], fields: str, api_key: str) -> List[Optional[dict]]:
    """One Semantic Scholar batch request. Returns one entry per id (None when the paper is not found)."""
    headers = {"User-Agent": "CiteLume/1.0"}
    if api_key and api_key.strip():
        headers["x-api-key"] = api_key.strip()
    await pace_semantic_scholar("x-api-key" in headers)
    async with httpx.AsyncClient(transport=retrying_transport(), timeout=60.0, headers=headers) as client:
        resp = await client.post(BATCH_URL, params={"fields": fields}, json={"ids": list(ids)})
        resp.raise_for_status()
        data = resp.json()
    return data if isinstance(data, list) else []


def choose_candidates(links_by_seed: Dict[str, Tuple[List[str], List[str]]], pool_s2_ids: Set[str],
                      min_links: int = MIN_LINKS, limit: int = MAX_ADDED * 2) -> List[Tuple[str, int]]:
    """Papers linked to at least `min_links` seeds (citing or cited by them) and not already in the pool, most
    linked first (ties broken by ID, so the choice is repeatable). Returns (Semantic Scholar ID, links)."""
    counts: Counter = Counter()
    for refs, cites in links_by_seed.values():
        for pid in set(refs) | set(cites):
            counts[pid] += 1
    ranked = sorted(((pid, n) for pid, n in counts.items() if n >= min_links and pid.lower() not in pool_s2_ids),
                    key=lambda x: (-x[1], x[0]))
    return ranked[:limit]


def to_record(item: dict, links: int) -> PaperRecord:
    ext = item.get("externalIds") or {}
    pid = item["paperId"]
    oa = item.get("openAccessPdf") or {}
    return PaperRecord(
        id=f"s2:{pid}", s2_id=pid, title=(item.get("title") or "").strip(),
        authors=[a.get("name", "") for a in (item.get("authors") or []) if a.get("name")][:12],
        year=item.get("year"), venue=item.get("venue") or "", source=SOURCE_LABEL, all_sources=[SOURCE_LABEL],
        doi=(str(ext.get("DOI") or "").strip().lower() or None), pmid=(str(ext.get("PubMed") or "").strip() or None),
        arxiv_id=(str(ext.get("ArXiv") or "").strip() or None), abstract=(item.get("abstract") or "").strip(),
        citation_count=int(item.get("citationCount") or 0), citation_source="Semantic Scholar",
        publication_types=[str(t) for t in (item.get("publicationTypes") or []) if t],
        url=item.get("url") or f"https://www.semanticscholar.org/paper/{pid}", pdf_url=oa.get("url"),
        retrieved_by=[f"{SOURCE_LABEL} (linked to {links} selected studies)"],
    )


@dataclass
class ChaseResult:
    added: List[PaperRecord] = field(default_factory=list)
    # Per seed record id: its Semantic Scholar ID and reference / citing tokens for the map
    seed_links: Dict[str, Dict[str, object]] = field(default_factory=dict)
    report: CitationChasingReport = field(default_factory=CitationChasingReport)


async def complete_from_pubmed(records: List[PaperRecord], api_key: str = "") -> int:
    """Semantic Scholar withholds many abstracts. Chased records with a PubMed ID get the abstract and publication
    types PubMed has, so they meet the eligibility rules on the same footing as searched records. Returns how many
    abstracts were filled."""
    want = [p for p in records if p.pmid and (len(p.abstract or "") < 50 or not p.publication_types)]
    if not want:
        return 0
    details = await fetch_pubmed_details([p.pmid for p in want], api_key)
    filled = 0
    for p in want:
        d = details.get(p.pmid.strip())
        if not d:
            continue
        if len(p.abstract or "") < 50 and d["abstract"]:
            p.abstract = d["abstract"]
            filled += 1
        if d["publication_types"]:
            p.publication_types = list(d["publication_types"])
    return filled


async def chase_citations(seeds: List[PaperRecord], pool: List[PaperRecord], api_key: str = "",
                          min_links: int = MIN_LINKS, max_added: int = MAX_ADDED,
                          pubmed_api_key: str = "") -> ChaseResult:
    """Backward and forward citation chasing from `seeds` (the selected studies). Never raises: a failure is reported
    in the result's note and as a source warning, and the review goes on without chased records."""
    report = CitationChasingReport(ran=True, version=CHASING_VERSION, min_links=min_links, max_added=max_added)
    result = ChaseResult(report=report)
    usable = [(p, seed_identifier(p)) for p in seeds[:MAX_SEEDS]]
    usable = [(p, sid) for p, sid in usable if sid]
    report.seeds = len(seeds[:MAX_SEEDS])
    report.seeds_without_identifier = report.seeds - len(usable)
    links_by_seed: Dict[str, Tuple[List[str], List[str]]] = {}
    try:
        for i in range(0, len(usable), SEEDS_PER_REQUEST):
            chunk = usable[i:i + SEEDS_PER_REQUEST]
            data = await _post_batch([sid for _, sid in chunk], LINK_FIELDS, api_key)
            report.requests += 1
            for (p, _), item in zip(chunk, data):
                if not item or not item.get("paperId"):
                    continue
                refs = [r["paperId"] for r in (item.get("references") or []) if r and r.get("paperId")]
                cites = [c["paperId"] for c in (item.get("citations") or []) if c and c.get("paperId")]
                links_by_seed[p.id] = (refs, cites)
                report.seeds_found += 1
                report.seeds_with_references += bool(refs)
                report.seeds_with_citations += bool(cites)
                result.seed_links[p.id] = {
                    "s2_id": item["paperId"],
                    "references": [f"s2:{r}" for r in refs[:MAP_LINKS_PER_SEED]],
                    "cited_by": [f"s2:{c}" for c in cites[:MAP_LINKS_PER_SEED]],
                }
        in_pool = pool_keys(pool)
        pool_s2 = {k[3:] for k in in_pool if k.startswith("s2:")} | {
            str(v["s2_id"]).lower() for v in result.seed_links.values()}
        ranked = choose_candidates(links_by_seed, pool_s2, min_links, limit=max_added * 2)
        report.linked_candidates = len(ranked)
        links_of = dict(ranked)
        details: List[dict] = []
        for i in range(0, len(ranked), DETAILS_PER_REQUEST):
            ids = [pid for pid, _ in ranked[i:i + DETAILS_PER_REQUEST]]
            details += [d for d in await _post_batch(ids, DETAIL_FIELDS, api_key) if d and d.get("paperId")]
            report.requests += 1
        seen: Set[str] = set()
        for item in details:
            keys = _item_keys(item)
            if not (item.get("title") or "").strip():
                continue
            if keys & in_pool or keys & seen:
                report.already_in_pool += 1
                continue
            seen |= keys
            result.added.append(to_record(item, links_of.get(item["paperId"], min_links)))
            if len(result.added) >= max_added:
                break
        report.added = len(result.added)
    except Exception as exc:  # the searches' records still stand
        report_source_failure("Semantic Scholar (citation searching)", exc)
        report.note = f"Citation searching stopped early: {type(exc).__name__}. Records found so far are kept."
    try:
        report.abstracts_from_pubmed = await complete_from_pubmed(result.added, pubmed_api_key)
    except Exception as exc:
        report_source_failure("PubMed (abstracts of chased records)", exc)
    report.added_without_abstract = sum(1 for p in result.added if len(p.abstract or "") < 50)
    return result


def attach_reference_data(raw: List[PaperRecord], seeds: List[PaperRecord], seed_links: Dict[str, Dict[str, object]]) -> int:
    """Copies the fetched Semantic Scholar ID and reference / citing lists onto every raw record of each seed (matched
    by DOI, PubMed ID, arXiv ID or title), where the record has none, so they survive the merge of duplicates.
    Returns the number of raw records updated."""
    by_key: Dict[str, Dict[str, object]] = {}
    for s in seeds:
        info = seed_links.get(s.id)
        if info:
            for k in pool_keys([s]):
                by_key.setdefault(k, info)
    updated = 0
    for r in raw:
        info = next((by_key[k] for k in sorted(pool_keys([r])) if k in by_key), None)
        if not info:
            continue
        changed = False
        if not r.s2_id:
            r.s2_id = str(info["s2_id"])
            changed = True
        if not r.references and info["references"]:
            r.references = list(info["references"])  # type: ignore[arg-type]
            changed = True
        if not r.cited_by and info["cited_by"]:
            r.cited_by = list(info["cited_by"])  # type: ignore[arg-type]
            changed = True
        updated += changed
    return updated


def count_included(included: List[PaperRecord]) -> int:
    return sum(1 for p in included if SOURCE_LABEL in (p.all_sources or []) or p.source == SOURCE_LABEL)
