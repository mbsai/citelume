import re
import xml.etree.ElementTree as ET
from typing import List, Optional
import httpx
from models import PaperRecord
from services.query_utils import build_concise_search_phrase
from services.europepmc_client import search_europe_pmc_medline
from services.source_health import report_source_failure, retrying_transport

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _clean_search_query(query: str, max_terms: int = 5) -> str:
    # If already compiled with PubMed Boolean / field tags, preserve it
    if any(tag in query for tag in ("[Title/Abstract]", "[MeSH", "[Publication Type]", " AND ", " OR ")):
        return query.strip()
    return build_concise_search_phrase(query, max_terms=max_terms)


async def search_pubmed(
    query: str,
    max_results: int = 8,
    api_key: str = "",
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    fallback_query: str = "",
) -> List[PaperRecord]:
    cleaned_q = _clean_search_query(query, max_terms=5)
    if year_from or year_to:
        y_start = year_from or 1900
        y_end = year_to or 2030
        cleaned_q = f"({cleaned_q}) AND ({y_start}:{y_end}[dp])"

    params = {
        "db": "pubmed",
        "term": cleaned_q,
        "retmax": str(max_results),
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()

    papers: List[PaperRecord] = []
    async with httpx.AsyncClient(transport=retrying_transport(), timeout=18.0, follow_redirects=True) as client:
        try:
            resp = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            backend_error = data.get("esearchresult", {}).get("ERROR")
            if backend_error and not id_list:
                # NCBI reports backend outages inside an HTTP 200 body; treat as a failure, not as "no results"
                raise RuntimeError(f"search backend error ({str(backend_error)[:80]})")
            if not id_list:
                # Retry with broader 3-term core query if compiled Boolean query was too restrictive
                broader_q = build_concise_search_phrase(fallback_query or query, max_terms=3)
                if broader_q and broader_q != cleaned_q:
                    params["term"] = broader_q
                    resp2 = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
                    if resp2.status_code == 200:
                        id_list = resp2.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }
            if api_key and api_key.strip():
                fetch_params["api_key"] = api_key.strip()

            fetch_resp = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=fetch_params)
            fetch_resp.raise_for_status()
            root = ET.fromstring(fetch_resp.text)

            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//MedlineCitation/PMID")
                pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
                if not pmid:
                    continue

                title_el = article.find(".//Article/ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled PubMed Article"

                abstract_parts = []
                for abs_el in article.findall(".//Article/Abstract/AbstractText"):
                    label = abs_el.attrib.get("Label", "")
                    txt = "".join(abs_el.itertext()).strip()
                    if txt:
                        abstract_parts.append(f"{label}: {txt}" if label else txt)
                abstract = " ".join(abstract_parts)

                authors = []
                for auth in article.findall(".//Article/AuthorList/Author"):
                    last = auth.findtext("LastName") or ""
                    fore = auth.findtext("ForeName") or auth.findtext("Initials") or ""
                    full = f"{fore} {last}".strip()
                    if full:
                        authors.append(full)

                year = None
                for y_path in [
                    ".//Article/Journal/JournalIssue/PubDate/Year",
                    ".//Article/ArticleDate/Year",
                    ".//MedlineCitation/DateCompleted/Year",
                ]:
                    y_txt = article.findtext(y_path)
                    if y_txt and y_txt.isdigit():
                        year = int(y_txt)
                        break
                if year is None:
                    medline_date = article.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
                    m = re.search(r"(19|20)\d{2}", medline_date)
                    if m:
                        year = int(m.group(0))

                venue = (
                    article.findtext(".//Article/Journal/Title")
                    or article.findtext(".//MedlineCitation/MedlineJournalInfo/MedlineTA")
                    or "PubMed"
                )

                doi = None
                for id_el in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                    if id_el.attrib.get("IdType") == "doi" and id_el.text:
                        doi = id_el.text.strip().lower()
                        break

                refs = []
                for ref_el in article.findall(".//PubmedData/ReferenceList/Reference/ArticleIdList/ArticleId"):
                    if ref_el.attrib.get("IdType") == "pmid" and ref_el.text:
                        refs.append(f"pmid:{ref_el.text.strip()}")
                    elif ref_el.attrib.get("IdType") == "doi" and ref_el.text:
                        refs.append(f"doi:{ref_el.text.strip().lower()}")

                papers.append(
                    PaperRecord(
                        id=f"pmid:{pmid}",
                        title=title,
                        authors=authors[:12],
                        year=year,
                        venue=venue,
                        source="PubMed",
                        all_sources=["PubMed"],
                        doi=doi,
                        pmid=pmid,
                        abstract=abstract,
                        citation_count=max(len(refs) // 2, 1),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        references=refs[:30],
                    )
                )
        except Exception as exc:
            print(f"[PubMed Connector Warning] {exc}")
            # NCBI is down or erroring: fetch the same PubMed records through the Europe PMC mirror
            try:
                papers = await search_europe_pmc_medline(cleaned_q, max_results)
                report_source_failure("PubMed", exc, fallback="the Europe PMC mirror of PubMed")
            except Exception as mirror_exc:
                print(f"[PubMed Mirror Warning] Europe PMC fallback failed: {mirror_exc}")
                report_source_failure("PubMed", exc)

    return papers
