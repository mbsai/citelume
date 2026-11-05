import re
from typing import List, Optional
import httpx
from models import PaperRecord
from services.query_utils import build_concise_search_phrase
from services.source_budget import openalex_call_allowed, pace_semantic_scholar
from services.source_health import openalex_api_key, report_source_failure, retrying_transport

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

S2_FIELDS = (
    "paperId,externalIds,url,title,abstract,venue,year,citationCount,"
    "influentialCitationCount,openAccessPdf,authors,references.paperId,"
    "references.externalIds,citations.paperId,citations.externalIds"
)


def _reconstruct_openalex_abstract(inverted_index: Optional[dict]) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def _prepare_s2_phrase(query: str) -> str:
    # Remove any Boolean operators as required by Semantic Scholar rule
    no_bool = re.sub(r"\b(AND|OR|NOT)\b", " ", query or "")
    words = no_bool.split()
    if 1 <= len(words) <= 8:
        return " ".join(words)
    return build_concise_search_phrase(no_bool, max_terms=6)


async def _fallback_openalex_search(
    query: str,
    max_results: int = 8,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[PaperRecord]:
    concise_q = _prepare_s2_phrase(query)
    params = {
        "search": concise_q,
        "per-page": str(max_results),
    }
    if openalex_api_key():
        params["api_key"] = openalex_api_key()
        blocked = openalex_call_allowed(openalex_api_key())
        if blocked:
            report_source_failure("Semantic Scholar / OpenAlex", RuntimeError(f"OpenAlex skipped: {blocked}"))
            return []
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if filters:
        params["filter"] = ",".join(filters)

    papers: List[PaperRecord] = []
    async with httpx.AsyncClient(transport=retrying_transport(), timeout=18.0, follow_redirects=True) as client:
        try:
            resp = await client.get(OPENALEX_WORKS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("results", []):
                oa_id = (item.get("id") or "").split("/")[-1]
                title = (item.get("title") or "Untitled Paper").strip()
                raw_doi = (item.get("doi") or "").replace("https://doi.org/", "").strip().lower()
                doi = raw_doi or None
                ids_dict = item.get("ids") or {}
                pmid_url = ids_dict.get("pmid") or ""
                pmid = pmid_url.split("/")[-1] if pmid_url else None

                year = item.get("publication_year")
                cite_count = int(item.get("cited_by_count") or 0)
                abstract = _reconstruct_openalex_abstract(item.get("abstract_inverted_index"))

                authors = []
                for auth in item.get("authorships", []):
                    name = (auth.get("author") or {}).get("display_name")
                    if name:
                        authors.append(name)

                primary_loc = item.get("primary_location") or {}
                source_obj = primary_loc.get("source") or {}
                venue = source_obj.get("display_name") or "Semantic Scholar / OpenAlex"
                pdf_url = (item.get("open_access") or {}).get("oa_url")
                url = f"https://doi.org/{doi}" if doi else (item.get("id") or "")

                refs = []
                for r_url in (item.get("referenced_works") or [])[:30]:
                    refs.append(f"oa:{r_url.split('/')[-1]}")

                papers.append(
                    PaperRecord(
                        id=f"s2:{oa_id}",
                        s2_id=oa_id,
                        title=title,
                        authors=authors[:12],
                        year=year,
                        venue=venue,
                        source="Semantic Scholar",
                        all_sources=["Semantic Scholar"],
                        doi=doi,
                        pmid=pmid,
                        abstract=abstract,
                        citation_count=cite_count,
                        url=url,
                        pdf_url=pdf_url,
                        references=refs,
                    )
                )
        except Exception as exc:
            print(f"[Semantic Scholar / OpenAlex Fallback Warning] {exc}")
            report_source_failure("Semantic Scholar / OpenAlex", exc)
    return papers


async def search_semantic_scholar(
    query: str,
    max_results: int = 8,
    api_key: str = "",
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[PaperRecord]:
    headers = {"User-Agent": "CiteLume/1.0"}
    if api_key and api_key.strip():
        headers["x-api-key"] = api_key.strip()

    concise_q = _prepare_s2_phrase(query)
    params = {
        "query": concise_q,
        "limit": str(max_results),
        "fields": S2_FIELDS,
    }
    if year_from or year_to:
        y1 = str(year_from) if year_from else ""
        y2 = str(year_to) if year_to else ""
        params["year"] = f"{y1}-{y2}"

    papers: List[PaperRecord] = []
    async with httpx.AsyncClient(transport=retrying_transport(), timeout=18.0, headers=headers, follow_redirects=True) as client:
        try:
            await pace_semantic_scholar("x-api-key" in headers)
            resp = await client.get(S2_SEARCH_URL, params=params)
            if resp.status_code in (429, 403):
                return await _fallback_openalex_search(query, max_results, year_from, year_to)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                s2_id = item.get("paperId") or ""
                if not s2_id:
                    continue
                ext_ids = item.get("externalIds") or {}
                doi = (ext_ids.get("DOI") or "").strip().lower() or None
                arxiv_id = (ext_ids.get("ArXiv") or "").strip() or None
                pmid = str(ext_ids.get("PubMed") or "").strip() or None

                authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
                oa_pdf = (item.get("openAccessPdf") or {}).get("url")
                url = item.get("url") or (f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{s2_id}")

                refs = []
                for ref in (item.get("references") or [])[:35]:
                    r_ext = ref.get("externalIds") or {}
                    if r_ext.get("DOI"):
                        refs.append(f"doi:{r_ext['DOI'].strip().lower()}")
                    elif r_ext.get("ArXiv"):
                        refs.append(f"arxiv:{r_ext['ArXiv'].strip()}")
                    elif ref.get("paperId"):
                        refs.append(f"s2:{ref['paperId']}")

                cited_by = []
                for cit in (item.get("citations") or [])[:35]:
                    c_ext = cit.get("externalIds") or {}
                    if c_ext.get("DOI"):
                        cited_by.append(f"doi:{c_ext['DOI'].strip().lower()}")
                    elif c_ext.get("ArXiv"):
                        cited_by.append(f"arxiv:{c_ext['ArXiv'].strip()}")
                    elif cit.get("paperId"):
                        cited_by.append(f"s2:{cit['paperId']}")

                papers.append(
                    PaperRecord(
                        id=f"s2:{s2_id}",
                        s2_id=s2_id,
                        title=(item.get("title") or "Untitled").strip(),
                        authors=authors[:12],
                        year=item.get("year"),
                        venue=item.get("venue") or "Semantic Scholar",
                        source="Semantic Scholar",
                        all_sources=["Semantic Scholar"],
                        doi=doi,
                        arxiv_id=arxiv_id,
                        pmid=pmid,
                        abstract=(item.get("abstract") or "").strip(),
                        citation_count=int(item.get("citationCount") or 0),
                        influential_citation_count=int(item.get("influentialCitationCount") or 0),
                        url=url,
                        pdf_url=oa_pdf,
                        references=refs,
                        cited_by=cited_by,
                    )
                )
        except Exception as exc:
            print(f"[Semantic Scholar Warning, using fallback] {exc}")
            return await _fallback_openalex_search(query, max_results, year_from, year_to)

    return papers


async def expand_paper_neighbors(
    title: str,
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    pmid: Optional[str] = None,
    s2_id: Optional[str] = None,
    api_key: str = "",
) -> List[PaperRecord]:
    """
    Fetches citing and referenced papers for a node when the user clicks 'Expand Node'
    in the interactive literature graph.
    """
    identifier = None
    if doi:
        identifier = f"DOI:{doi}"
    elif arxiv_id:
        identifier = f"ARXIV:{arxiv_id}"
    elif pmid:
        identifier = f"PMID:{pmid}"
    elif s2_id and len(s2_id) == 40:
        identifier = s2_id

    headers = {"User-Agent": "CiteLume/1.0"}
    if api_key and api_key.strip():
        headers["x-api-key"] = api_key.strip()

    if identifier:
        fields = (
            "references.paperId,references.title,references.year,references.venue,"
            "references.authors,references.citationCount,references.abstract,references.externalIds,references.url,"
            "citations.paperId,citations.title,citations.year,citations.venue,"
            "citations.authors,citations.citationCount,citations.abstract,citations.externalIds,citations.url"
        )
        async with httpx.AsyncClient(transport=retrying_transport(), timeout=15.0, headers=headers, follow_redirects=True) as client:
            try:
                await pace_semantic_scholar("x-api-key" in headers)
                resp = await client.get(f"{S2_PAPER_URL}/{identifier}", params={"fields": fields})
                if resp.status_code == 200:
                    data = resp.json()
                    neighbors: List[PaperRecord] = []
                    combined = (data.get("citations") or [])[:6] + (data.get("references") or [])[:6]
                    for item in combined:
                        pid = item.get("paperId")
                        if not pid or not item.get("title"):
                            continue
                        ext = item.get("externalIds") or {}
                        ndoi = (ext.get("DOI") or "").strip().lower() or None
                        narxiv = (ext.get("ArXiv") or "").strip() or None
                        npmid = str(ext.get("PubMed") or "").strip() or None
                        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
                        neighbors.append(
                            PaperRecord(
                                id=f"s2:{pid}",
                                s2_id=pid,
                                title=item.get("title", "").strip(),
                                authors=authors[:8],
                                year=item.get("year"),
                                venue=item.get("venue") or "Semantic Scholar",
                                source="Semantic Scholar",
                                all_sources=["Semantic Scholar"],
                                doi=ndoi,
                                arxiv_id=narxiv,
                                pmid=npmid,
                                abstract=(item.get("abstract") or "").strip(),
                                citation_count=int(item.get("citationCount") or 0),
                                url=item.get("url") or f"https://www.semanticscholar.org/paper/{pid}",
                            )
                        )
                    if neighbors:
                        return neighbors
            except Exception as exc:
                print(f"[Expand Node S2 Warning] {exc}")

    # Fallback: search by core title keywords via OpenAlex/S2
    return await _fallback_openalex_search(title, max_results=6)
