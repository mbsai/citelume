"""
Search preview for a review protocol: the strings each database will be searched with, and how many records each
database holds for the core search, before anything is retrieved. The counts come from each database's own total
(PubMed esearch count, Semantic Scholar total, Europe PMC hit count for bioRxiv/medRxiv, arXiv total results), with
the same query cleaning and year limits the real searches use.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Dict, Optional, Tuple

import httpx

from models import ApiKeysConfig, ManualSearch, ReviewProtocol, SearchCount, SearchPreview
from services.pubmed_client import EUTILS_BASE, _clean_search_query, _pubmed_slot
from services.query_planning import results_per_search
from services.importers import manual_search_strings
from services.review_protocol import plan_from_protocol
from services.s2_client import S2_SEARCH_URL, _prepare_s2_phrase
from services.source_budget import pace_semantic_scholar
from services.source_health import retrying_transport

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARXIV_API = "https://export.arxiv.org/api/query"
COUNT_TIMEOUT_S = 10.0
COUNT_DEADLINE_S = 15.0   # the longest a preview waits for any one database
CACHE_SECONDS = 600
_CACHE: Dict[Tuple, Tuple[float, SearchCount]] = {}


async def _cached(key: Tuple, fetch) -> SearchCount:
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1].model_copy()
    result = await fetch()
    if result.count is not None:
        if len(_CACHE) > 500:
            _CACHE.clear()
        _CACHE[key] = (time.time(), result)
    return result


def _unavailable(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return "Count not available right now: the database is limiting requests (the search itself falls back or retries)"
    if isinstance(exc, httpx.TimeoutException):
        return "Count not available: the database did not reply in time"
    return "Count not available"


def _years(y1: Optional[int], y2: Optional[int]) -> str:
    return f"{y1 or ''}-{y2 or ''}" if (y1 or y2) else ""


async def count_pubmed(query: str, api_key: str, y1: Optional[int], y2: Optional[int]) -> SearchCount:
    term = _clean_search_query(query, max_terms=5)
    if y1 or y2:
        term = f"({term}) AND ({y1 or 1900}:{y2 or 2030}[dp])"

    async def fetch() -> SearchCount:
        params = {"db": "pubmed", "term": term, "retmax": "0", "retmode": "json"}
        if api_key.strip():
            params["api_key"] = api_key.strip()
        try:
            async with _pubmed_slot(bool(api_key.strip())), \
                    httpx.AsyncClient(transport=retrying_transport(max_retries=1), timeout=COUNT_TIMEOUT_S) as client:
                r = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
                r.raise_for_status()
                return SearchCount(source="PubMed", query=term, count=int(r.json()["esearchresult"]["count"]))
        except Exception as exc:
            return SearchCount(source="PubMed", query=term, note=_unavailable(exc))
    return await _cached(("pubmed", term), fetch)


async def count_semantic_scholar(query: str, api_key: str, y1: Optional[int], y2: Optional[int]) -> SearchCount:
    phrase = _prepare_s2_phrase(query)

    async def fetch() -> SearchCount:
        headers = {"User-Agent": "CiteLume/1.0"}
        if api_key.strip():
            headers["x-api-key"] = api_key.strip()
        params = {"query": phrase, "limit": "1", "fields": "title"}
        if _years(y1, y2):
            params["year"] = _years(y1, y2)
        try:
            async with httpx.AsyncClient(transport=retrying_transport(max_retries=1), timeout=COUNT_TIMEOUT_S,
                                         headers=headers) as client:
                await pace_semantic_scholar("x-api-key" in headers)
                r = await client.get(S2_SEARCH_URL, params=params)
                r.raise_for_status()
                return SearchCount(source="Semantic Scholar", query=phrase, count=int(r.json().get("total", 0)))
        except Exception as exc:
            return SearchCount(source="Semantic Scholar", query=phrase,
                               note=_unavailable(exc))
    return await _cached(("s2", phrase, _years(y1, y2)), fetch)


async def count_preprints(phrase: str, y1: Optional[int], y2: Optional[int]) -> SearchCount:
    q = f"({phrase}) AND (SRC:PPR)" + (f" AND (PUB_YEAR:[{y1 or 2013} TO {y2 or 2030}])" if (y1 or y2) else "")

    async def fetch() -> SearchCount:
        try:
            async with httpx.AsyncClient(transport=retrying_transport(max_retries=1), timeout=COUNT_TIMEOUT_S) as client:
                r = await client.get(EPMC_SEARCH_URL, params={"query": q, "pageSize": "1", "format": "json"})
                r.raise_for_status()
                return SearchCount(source="bioRxiv / medRxiv (via Europe PMC)", query=q,
                                   count=int(r.json().get("hitCount", 0)))
        except Exception as exc:
            return SearchCount(source="bioRxiv / medRxiv (via Europe PMC)", query=q,
                               note=_unavailable(exc))
    return await _cached(("epmc", q), fetch)


async def count_arxiv(query: str) -> SearchCount:
    async def fetch() -> SearchCount:
        try:
            async with httpx.AsyncClient(transport=retrying_transport(max_retries=1), timeout=COUNT_TIMEOUT_S) as client:
                r = await client.get(ARXIV_API, params={"search_query": query, "max_results": "0"})
                r.raise_for_status()
                m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", r.text)
                return SearchCount(source="arXiv", query=query, count=int(m.group(1)) if m else None,
                                   note="" if m else "Count unavailable")
        except Exception as exc:
            return SearchCount(source="arXiv", query=query, note=_unavailable(exc))
    return await _cached(("arxiv", query), fetch)


async def preview_searches(protocol: ReviewProtocol, mode: str, config: ApiKeysConfig) -> SearchPreview:
    plan = plan_from_protocol(protocol, mode)
    facets = plan.search_facets
    preview = SearchPreview(
        facets=facets,
        concepts_used=[c.name + (f" ({', '.join(c.synonyms)})" if c.synonyms else "")
                       for c in plan.question_understanding.concepts.search_concepts],
        populations_and_places=[c.name for c in plan.question_understanding.concepts.required_constraints],
    )
    u = plan.question_understanding
    preview.manual_searches = [ManualSearch(**m) for m in manual_search_strings(
        [(c.name, list(c.synonyms)) for c in u.concepts.search_concepts],
        [list(c.spellings) for c in u.concepts.required_constraints],
        protocol.eligibility.year_from, protocol.eligibility.year_to)]
    from services import demo_mode
    preview.imports_allowed = not demo_mode.demo_enabled()
    if not facets:
        return preview
    core = facets[0].queries
    y1, y2 = protocol.eligibility.year_from, protocol.eligibility.year_to
    tasks = []
    if config.enable_pubmed and core.pubmed:
        tasks.append(count_pubmed(core.pubmed, config.pubmed_api_key, y1, y2))
    if config.enable_s2 and core.semantic_scholar:
        tasks.append(count_semantic_scholar(core.semantic_scholar, config.s2_api_key, y1, y2))
    if config.enable_biorxiv and core.biorxiv_medrxiv:
        tasks.append(count_preprints(core.biorxiv_medrxiv[0], y1, y2))
    if config.enable_arxiv and core.arxiv:
        tasks.append(count_arxiv(core.arxiv))
    async def within_limit(task, source: str) -> SearchCount:
        # A preview must answer quickly: a database that is slow or limiting requests is reported as such
        try:
            return await asyncio.wait_for(task, timeout=COUNT_DEADLINE_S)
        except asyncio.TimeoutError:
            return SearchCount(source=source, note="Count not available: the database did not reply in time")

    sources = ["PubMed", "Semantic Scholar", "bioRxiv / medRxiv (via Europe PMC)", "arXiv"]
    names = [n for n, on in zip(sources, [config.enable_pubmed and core.pubmed, config.enable_s2 and core.semantic_scholar,
                                         config.enable_biorxiv and core.biorxiv_medrxiv, config.enable_arxiv and core.arxiv]) if on]
    counts = list(await asyncio.gather(*[within_limit(t, n) for t, n in zip(tasks, names)]))
    per_db = results_per_search(mode, "core", protocol.review_type == "systematic_prisma")
    for c in counts:
        if c.count is not None and not c.note:
            c.note = (f"CiteLume retrieves the first {per_db} of these for the core search"
                      if c.count > per_db else "All of these are retrieved by the core search")
    preview.core_counts = counts
    return preview
