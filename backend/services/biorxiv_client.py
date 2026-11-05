import re
from typing import List, Optional, Union
import httpx
from models import PaperRecord
from services.query_utils import build_concise_search_phrase
from services.source_health import report_source_failure, retrying_transport

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BIORXIV_API_BASE = "https://api.biorxiv.org/details/biorxiv"


async def search_biorxiv(
    query: Union[str, List[str]],
    max_results: int = 8,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    fallback_query: str = "",
) -> List[PaperRecord]:
    """
    Searches bioRxiv and medRxiv preprints via EuropePMC's preprint index (SRC:PPR)
    using a flat list of short, dense multi-word search phrases executed sequentially.
    """
    if isinstance(query, list):
        phrase_list = [str(q).strip() for q in query if str(q).strip()]
    else:
        phrase_list = [build_concise_search_phrase(query, max_terms=4)]

    if not phrase_list:
        phrase_list = [build_concise_search_phrase(fallback_query or "antimicrobial peptide", max_terms=3)]

    papers: List[PaperRecord] = []
    seen_ids = set()
    per_phrase_limit = max(3, max_results // max(len(phrase_list), 1))

    async with httpx.AsyncClient(transport=retrying_transport(), timeout=18.0, follow_redirects=True) as client:
        try:
            all_raw_items = []
            for phrase in phrase_list[:3]:
                epmc_query = f"({phrase}) AND (SRC:PPR)"
                if year_from or year_to:
                    y1 = year_from or 2013
                    y2 = year_to or 2030
                    epmc_query += f" AND (PUB_YEAR:[{y1} TO {y2}])"
                params = {
                    "query": epmc_query,
                    "resultType": "core",
                    "pageSize": str(per_phrase_limit),
                    "format": "json",
                }
                resp = await client.get(EPMC_SEARCH_URL, params=params)
                if resp.status_code == 200:
                    batch = resp.json().get("resultList", {}).get("result", [])
                    all_raw_items.extend(batch)

            if not all_raw_items:
                broader = build_concise_search_phrase(fallback_query or phrase_list[0], max_terms=2)
                if broader:
                    resp2 = await client.get(
                        EPMC_SEARCH_URL,
                        params={
                            "query": f"({broader}) AND (SRC:PPR)",
                            "resultType": "core",
                            "pageSize": str(max_results),
                            "format": "json",
                        },
                    )
                    if resp2.status_code == 200:
                        all_raw_items = resp2.json().get("resultList", {}).get("result", [])

            results = []
            for item in all_raw_items:
                iid = item.get("doi") or item.get("id")
                if iid and iid not in seen_ids:
                    seen_ids.add(iid)
                    results.append(item)
                if len(results) >= max_results:
                    break

            for item in results:
                doi = (item.get("doi") or "").strip().lower()
                title = (item.get("title") or "Untitled Preprint").strip()
                title = re.sub(r"<[^>]+>", "", title)

                abstract = (item.get("abstractText") or "").strip()
                abstract = re.sub(r"<[^>]+>", "", abstract)

                pub_year_str = str(item.get("pubYear") or "")
                year = int(pub_year_str) if pub_year_str.isdigit() else None

                authors = []
                author_list = item.get("authorList", {}).get("author", [])
                for a in author_list:
                    full_name = a.get("fullName") or f"{a.get('firstName', '')} {a.get('lastName', '')}".strip()
                    if full_name:
                        authors.append(full_name)

                book_or_report = item.get("bookOrReportDetails", {}) or {}
                publisher = book_or_report.get("publisher") or "bioRxiv"
                if "medrxiv" in publisher.lower():
                    venue = "medRxiv"
                else:
                    venue = "bioRxiv"

                cited_by_count = int(item.get("citedByCount") or 0)
                url = f"https://doi.org/{doi}" if doi else f"https://www.biorxiv.org/search/{cleaned}"
                pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf" if doi and "10.1101" in doi else None

                paper_id = f"doi:{doi}" if doi else f"biorxiv:{item.get('id', len(papers))}"
                papers.append(
                    PaperRecord(
                        id=paper_id,
                        title=title,
                        authors=authors[:12],
                        year=year,
                        venue=venue,
                        source="bioRxiv",
                        all_sources=["bioRxiv"],
                        doi=doi or None,
                        abstract=abstract,
                        citation_count=cited_by_count,
                        url=url,
                        pdf_url=pdf_url,
                    )
                )
        except Exception as exc:
            print(f"[bioRxiv Connector Warning] {exc}")
            report_source_failure("bioRxiv / medRxiv", exc)

    return papers
