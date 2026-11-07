"""
Queries that are a paper identifier rather than a topic: a DOI, a PubMed ID or link, or an arXiv ID or link.
They are resolved to the paper's title, which then drives a search for related literature.
"""
import re
from typing import Optional, Tuple

import httpx

from services.source_health import retrying_transport

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_PMID_URL = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/)(\d{5,9})", re.I)
_PMID_BARE = re.compile(r"^\s*(?:PMID[:\s]*)?(\d{6,9})\s*$", re.I)
_ARXIV = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def identifier_in_query(query: str) -> Optional[Tuple[str, str]]:
    """(kind, id) when the query is essentially just a paper identifier or link, else None."""
    q = (query or "").strip()
    if not q or len(q.split()) > 3:
        return None
    if m := _PMID_URL.search(q) or _PMID_BARE.match(q):
        return "pmid", m.group(1)
    if m := _DOI.search(q):
        return "doi", m.group(1).rstrip(".,;)/")
    if (m := _ARXIV.search(q)) and ("arxiv" in q.lower() or re.fullmatch(r"\s*\d{4}\.\d{4,5}(v\d+)?\s*", q)):
        return "arxiv", m.group(1)
    return None


async def resolve_identifier_title(kind: str, ident: str) -> Optional[str]:
    """Title of the paper behind an identifier, or None if it cannot be found."""
    try:
        async with httpx.AsyncClient(transport=retrying_transport(), timeout=15.0, follow_redirects=True,
                                     headers={"User-Agent": "CiteLume/1.0"}) as client:
            if kind == "doi":
                r = await client.get(f"https://api.crossref.org/works/{ident}")
                if r.status_code == 200:
                    titles = (r.json().get("message") or {}).get("title") or []
                    return titles[0].strip() if titles else None
            elif kind == "pmid":
                r = await client.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                                     params={"query": f"EXT_ID:{ident} AND SRC:MED", "format": "json"})
                if r.status_code == 200:
                    res = (r.json().get("resultList") or {}).get("result") or []
                    if res:
                        return re.sub(r"<[^>]+>", "", res[0].get("title") or "").strip().rstrip(".") or None
                r = await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                                     params={"db": "pubmed", "id": ident, "retmode": "json"})
                if r.status_code == 200:
                    doc = (r.json().get("result") or {}).get(ident) or {}
                    return (doc.get("title") or "").strip().rstrip(".") or None
            elif kind == "arxiv":
                r = await client.get("https://export.arxiv.org/api/query", params={"id_list": ident})
                if r.status_code == 200:
                    m = re.search(r"<entry>.*?<title>(.*?)</title>", r.text, re.S)
                    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    except Exception:
        return None
    return None
