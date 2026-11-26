"""
OpenAlex as a full source for the core and reviews searches (selection protocol 2.7). Before 2.7 it was only a
fallback when Semantic Scholar failed; it covers journals outside MEDLINE (environmental science, agriculture,
engineering) that PubMed does not index.

The search is a title-and-abstract Boolean search compiled by rules from the concepts (query_planning), not
OpenAlex's relevance search, so it is the same kind of search as the PubMed one. Calls count against the demo's
OpenAlex budget (services/source_budget.py); with no key OpenAlex's small shared budget applies and a refusal is
reported, not fatal.
"""
from __future__ import annotations

from typing import List, Optional

import httpx

from models import PaperRecord
from services.source_budget import openalex_call_allowed
from services.source_health import openalex_api_key, report_source_failure

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SELECT = ("id,doi,ids,title,publication_year,cited_by_count,abstract_inverted_index,authorships,primary_location,"
          "open_access,type,referenced_works")


def _abstract(inverted: Optional[dict]) -> str:
    if not isinstance(inverted, dict):
        return ""
    pos = sorted((p, w) for w, ps in inverted.items() if isinstance(ps, list) for p in ps)
    return " ".join(w for _, w in pos)


def record(item: dict) -> PaperRecord:
    oa_id = (item.get("id") or "").split("/")[-1]
    doi = (item.get("doi") or "").replace("https://doi.org/", "").strip().lower() or None
    pmid_url = (item.get("ids") or {}).get("pmid") or ""
    source = ((item.get("primary_location") or {}).get("source") or {})
    kind = item.get("type")
    return PaperRecord(
        id=f"oa:{oa_id}",
        title=(item.get("title") or "Untitled").strip(),
        authors=[(a.get("author") or {}).get("display_name") for a in item.get("authorships") or []
                 if (a.get("author") or {}).get("display_name")][:12],
        year=item.get("publication_year"),
        venue=source.get("display_name") or "OpenAlex",
        source="OpenAlex",
        all_sources=["OpenAlex"],
        doi=doi,
        pmid=pmid_url.split("/")[-1] or None,
        abstract=_abstract(item.get("abstract_inverted_index")),
        citation_count=int(item.get("cited_by_count") or 0),
        citation_source="OpenAlex",
        publication_types=[kind] if kind in ("review", "editorial", "letter", "erratum", "retraction") else [],
        url=f"https://doi.org/{doi}" if doi else (item.get("id") or ""),
        pdf_url=(item.get("open_access") or {}).get("oa_url"),
        references=[f"oa:{r.split('/')[-1]}" for r in (item.get("referenced_works") or [])[:30]],
    )


async def search_openalex(
    expression: str,
    max_results: int = 25,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    reviews_only: bool = False,
) -> List[PaperRecord]:
    if not expression.strip():
        return []
    key = openalex_api_key()
    blocked = openalex_call_allowed(key) if key else None
    if blocked:
        report_source_failure("OpenAlex", RuntimeError(f"OpenAlex skipped: {blocked}"))
        return []
    filters = [f"title_and_abstract.search:{expression}"]
    if year_from or year_to:
        filters.append(f"publication_year:{year_from or 1800}-{year_to or 2100}")
    if reviews_only:
        filters.append("type:review")
    params = {"filter": ",".join(filters), "per-page": str(min(max_results, 200)), "select": SELECT}
    if key:
        params["api_key"] = key
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(OPENALEX_WORKS_URL, params=params)
            resp.raise_for_status()
            return [record(i) for i in resp.json().get("results") or []]
    except Exception as exc:
        print(f"[OpenAlex Connector Warning] {exc}")
        report_source_failure("OpenAlex", exc)
        return []
