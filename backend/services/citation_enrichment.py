"""
Real citation counts for candidate studies before they are ranked.

PubMed's own records carry no citation count, so PMIDs are looked up in NIH iCite (free, no key, up to 200 per
request). Records from Semantic Scholar, OpenAlex and Europe PMC already carry counts from those services.
Records left without a count (mostly preprints) are scored at the neutral middle value by the selection protocol.
"""
import asyncio
from typing import Dict, List

import httpx

from models import PaperRecord
from services.source_health import retrying_transport

ICITE_URL = "https://icite.od.nih.gov/api/pubs"
ICITE_BATCH = 200
_CACHE: Dict[str, int] = {}


async def _icite_batch(client: httpx.AsyncClient, pmids: List[str]) -> Dict[str, int]:
    try:
        resp = await client.get(ICITE_URL, params={"pmids": ",".join(pmids), "fl": "pmid,citation_count"})
        resp.raise_for_status()
        return {str(d.get("pmid")): int(d.get("citation_count") or 0) for d in (resp.json().get("data") or [])}
    except Exception as exc:
        print(f"[iCite Warning] {str(exc)[:160]}")
        return {}


async def enrich_citation_counts(papers: List[PaperRecord]) -> int:
    """Fill citation counts from iCite for records with a PMID and no sourced count. Returns how many were filled."""
    need = list(dict.fromkeys(p.pmid for p in papers if p.pmid and not p.citation_source and p.pmid not in _CACHE))
    if need:
        async with httpx.AsyncClient(transport=retrying_transport(), timeout=12.0,
                                     headers={"User-Agent": "CiteLume/1.0"}) as client:
            batches = [need[i:i + ICITE_BATCH] for i in range(0, len(need), ICITE_BATCH)]
            for found in await asyncio.gather(*[_icite_batch(client, b) for b in batches]):
                _CACHE.update(found)
        if len(_CACHE) > 20000:
            for k in list(_CACHE)[:10000]:
                _CACHE.pop(k, None)
    filled = 0
    for p in papers:
        if p.pmid and not p.citation_source and p.pmid in _CACHE:
            p.citation_count = _CACHE[p.pmid]
            p.citation_source = "iCite"
            filled += 1
    return filled
