"""
Europe PMC as a mirror for PubMed: the same MEDLINE/PubMed records (SRC:MED) with abstracts, PMIDs, DOIs and
open-access flags, used when NCBI's E-utilities are failing. No API key is required.

PubMed query syntax from the planner is translated to Europe PMC's field syntax:
  "term"[Title/Abstract] -> (TITLE:"term" OR ABSTRACT:"term")
  "Review"[Publication Type] -> PUB_TYPE:"review"
  "Heading"[MeSH Terms] -> MESH:"Heading"
  (2010:2020[dp]) -> PUB_YEAR:[2010 TO 2020]
"""
import re
from typing import List, Optional

import httpx

from models import PaperRecord
from services.source_health import retrying_transport

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

_TIAB = re.compile(r'("[^"]+"|[\w\-*]+)\[(?:Title/Abstract|tiab)\]', re.I)
_TITLE = re.compile(r'("[^"]+"|[\w\-*]+)\[(?:Title|ti)\]', re.I)
# The tagged term is either quoted or a single word; never a run of preceding query text
_PUBTYPE = re.compile(r'(?:"([^"]+)"|([\w\-]+))\[(?:Publication Type|pt)\]', re.I)
_MESH = re.compile(r'(?:"([^"]+)"|([\w\-]+))\[(?:MeSH Terms|MeSH|mh)\]', re.I)
_DATE = re.compile(r"(\d{4}):(\d{4})\[dp\]", re.I)


def translate_pubmed_query(query: str) -> str:
    """PubMed E-utilities query -> Europe PMC query restricted to PubMed records."""
    q = _TIAB.sub(lambda m: f"(TITLE:{m.group(1)} OR ABSTRACT:{m.group(1)})", query)
    q = _TITLE.sub(lambda m: f"TITLE:{m.group(1)}", q)
    q = _PUBTYPE.sub(lambda m: f'PUB_TYPE:"{(m.group(1) or m.group(2)).strip().lower()}"', q)
    q = _MESH.sub(lambda m: f'MESH:"{(m.group(1) or m.group(2)).strip()}"', q)
    q = _DATE.sub(lambda m: f"PUB_YEAR:[{m.group(1)} TO {m.group(2)}]", q)
    q = re.sub(r"\[[A-Za-z /]+\]", "", q)  # any remaining PubMed field tag (not the PUB_YEAR range above)
    return f"({q.strip()}) AND SRC:MED"


def _record(item: dict) -> Optional[PaperRecord]:
    pmid = item.get("pmid") or item.get("id")
    title = re.sub(r"<[^>]+>", "", item.get("title") or "").strip().rstrip(".")
    if not pmid or not title:
        return None
    authors = [a.strip() for a in (item.get("authorString") or "").rstrip(".").split(",") if a.strip()]
    journal = ((item.get("journalInfo") or {}).get("journal") or {}).get("title") or item.get("journalTitle") or ""
    year = item.get("pubYear")
    return PaperRecord(
        id=f"pmid:{pmid}",
        title=title,
        authors=authors[:12],
        year=int(year) if str(year or "").isdigit() else None,
        venue=journal,
        source="PubMed",
        all_sources=["PubMed (via Europe PMC)"],
        doi=(item.get("doi") or "").lower() or None,
        pmid=str(pmid),
        abstract=re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", item.get("abstractText") or "")).strip(),
        citation_count=int(item.get("citedByCount") or 0),
        citation_source="Europe PMC",
        publication_types=[str(t) for t in ((item.get("pubTypeList") or {}).get("pubType") or []) if t],
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


async def search_europe_pmc_medline(pubmed_query: str, max_results: int = 8) -> List[PaperRecord]:
    """PubMed records via Europe PMC for a PubMed-syntax query; raises on failure so callers can report it."""
    params = {
        "query": translate_pubmed_query(pubmed_query),
        "format": "json",
        "resultType": "core",
        "pageSize": str(max_results),
        "sort": "RELEVANCE",
    }
    async with httpx.AsyncClient(transport=retrying_transport(), timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(EPMC_SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    if "resultList" not in data:
        raise RuntimeError(f"Europe PMC error ({str(data.get('errMsg') or data)[:80]})")
    records = [_record(item) for item in data["resultList"].get("result") or []]
    return [r for r in records if r is not None]
