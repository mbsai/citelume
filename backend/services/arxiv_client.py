import re
import xml.etree.ElementTree as ET
from typing import List, Optional
import httpx
from models import PaperRecord
from services.query_utils import extract_core_scientific_terms
from services.source_health import report_source_failure, retrying_transport

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _build_arxiv_query(query: str, max_terms: int = 4) -> str:
    if any(prefix in query for prefix in ("ti:", "abs:", "all:", " AND ", " OR ")):
        return query.strip()
    tokens = extract_core_scientific_terms(query, max_terms=max_terms)
    if not tokens:
        return "all:machine AND all:learning"
    return " AND ".join(f"all:{t}" for t in tokens)


async def search_arxiv(
    query: str,
    max_results: int = 8,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    fallback_query: str = "",
) -> List[PaperRecord]:
    search_q = _build_arxiv_query(query, max_terms=4)
    params = {
        "search_query": search_q,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    papers: List[PaperRecord] = []
    async with httpx.AsyncClient(transport=retrying_transport(), timeout=18.0, follow_redirects=True) as client:
        try:
            resp = await client.get(ARXIV_API_URL, params=params)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ATOM_NS)
            if not entries:
                fb_tokens = extract_core_scientific_terms(fallback_query or query, max_terms=2)
                broader_q = " AND ".join(f"all:{t}" for t in fb_tokens) if fb_tokens else "all:protein"
                if broader_q != search_q:
                    params["search_query"] = broader_q
                    resp2 = await client.get(ARXIV_API_URL, params=params)
                    if resp2.status_code == 200:
                        root = ET.fromstring(resp2.text)
                        entries = root.findall("atom:entry", ATOM_NS)

            for entry in entries:
                id_url = (entry.findtext("atom:id", "", ATOM_NS) or "").strip()
                if not id_url:
                    continue
                # Extract clean arXiv ID without version suffix
                raw_id = id_url.split("/abs/")[-1]
                arxiv_id = re.sub(r"v\d+$", "", raw_id)

                title = re.sub(r"\s+", " ", (entry.findtext("atom:title", "", ATOM_NS) or "").strip())
                summary = re.sub(r"\s+", " ", (entry.findtext("atom:summary", "", ATOM_NS) or "").strip())
                published = (entry.findtext("atom:published", "", ATOM_NS) or "").strip()

                year = None
                if len(published) >= 4 and published[:4].isdigit():
                    year = int(published[:4])

                if year_from and year and year < year_from:
                    continue
                if year_to and year and year > year_to:
                    continue

                authors = []
                for auth in entry.findall("atom:author", ATOM_NS):
                    name = (auth.findtext("atom:name", "", ATOM_NS) or "").strip()
                    if name:
                        authors.append(name)

                doi_el = entry.find("arxiv:doi", ATOM_NS)
                doi = doi_el.text.strip().lower() if doi_el is not None and doi_el.text else None

                primary_cat = entry.find("arxiv:primary_category", ATOM_NS)
                cat_term = primary_cat.attrib.get("term", "arXiv") if primary_cat is not None else "arXiv"

                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                papers.append(
                    PaperRecord(
                        id=f"arxiv:{arxiv_id}",
                        title=title,
                        authors=authors[:12],
                        year=year,
                        venue=f"arXiv ({cat_term})",
                        source="arXiv",
                        all_sources=["arXiv"],
                        doi=doi,
                        arxiv_id=arxiv_id,
                        abstract=summary,
                        citation_count=0,
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        pdf_url=pdf_url,
                    )
                )
        except Exception as exc:
            print(f"[arXiv Connector Warning] {exc}")
            report_source_failure("arXiv", exc)

    return papers
