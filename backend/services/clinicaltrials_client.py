"""
ClinicalTrials.gov (API v2, public, no key) as a source for the core search (selection protocol 2.7).

A trial registry is searched for systematic reviews of interventions (Cochrane Handbook ch. 4; PRISMA-S item 3).
For each registered trial the search returns:
  - its publications that the registry links as results (reference type RESULT), fetched from PubMed as
    ordinary records, so they merge with the same paper found by the PubMed search;
  - the registry record itself when the trial posted results on ClinicalTrials.gov but links no publication
    (an unpublished trial with results), with its design, phase and enrolment in the summary.
Registrations without results or a linked publication are not studies that can be read, and are only counted.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import httpx

from models import PaperRecord
from services.pubmed_client import search_pubmed
from services.source_health import report_source_failure

CT_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
CT_FIELDS = ",".join([
    "protocolSection.identificationModule", "protocolSection.descriptionModule.briefSummary",
    "protocolSection.designModule", "protocolSection.statusModule", "protocolSection.referencesModule",
    "protocolSection.conditionsModule.conditions", "hasResults",
])
# RESULT: the trial's own results publication. DERIVED (PubMed found the NCT number in a paper) also covers
# reviews and unrelated papers that merely cite the trial, so it is not used.
LINKED_TYPES = {"RESULT"}


def _year(date_struct: Optional[dict]) -> Optional[int]:
    m = re.match(r"(\d{4})", (date_struct or {}).get("date", "") or "")
    return int(m.group(1)) if m else None


def _design_text(design: dict) -> str:
    info = design.get("designInfo") or {}
    parts = []
    if design.get("studyType"):
        parts.append(design["studyType"].lower().replace("_", " ") + " study")
    if info.get("allocation") and info["allocation"] != "NA":
        parts.append(info["allocation"].lower().replace("_", " ").replace("randomized", "randomised"))
    if info.get("interventionModel"):
        parts.append(info["interventionModel"].lower().replace("_", " ") + " assignment")
    if info.get("observationalModel"):
        parts.append(info["observationalModel"].lower().replace("_", " "))
    phases = [p.replace("PHASE", "phase ").replace("NA", "").strip() for p in design.get("phases") or []]
    if any(phases):
        parts.append("/".join(p for p in phases if p))
    enrol = (design.get("enrollmentInfo") or {}).get("count")
    if enrol:
        parts.append(f"{enrol} participants enrolled")
    return "; ".join(parts)


def parse_studies(studies: List[dict]) -> Tuple[List[str], List[PaperRecord], int]:
    """(linked PubMed IDs, registry records with posted results, registrations with neither)."""
    pmids: List[str] = []
    records: List[PaperRecord] = []
    neither = 0
    for s in studies:
        ps = s.get("protocolSection") or {}
        ident = ps.get("identificationModule") or {}
        nct = ident.get("nctId") or ""
        refs = (ps.get("referencesModule") or {}).get("references") or []
        linked = [str(r["pmid"]) for r in refs if r.get("pmid") and r.get("type") in LINKED_TYPES]
        if linked:
            pmids.extend(p for p in linked if p not in pmids)
            continue
        if not s.get("hasResults"):
            neither += 1
            continue
        status = ps.get("statusModule") or {}
        design = ps.get("designModule") or {}
        summary = ((ps.get("descriptionModule") or {}).get("briefSummary") or "").strip()
        design_txt = _design_text(design)
        records.append(PaperRecord(
            id=f"nct:{nct}",
            title=(ident.get("officialTitle") or ident.get("briefTitle") or nct).strip(),
            year=_year(status.get("resultsFirstPostDateStruct")) or _year(status.get("completionDateStruct")),
            venue="ClinicalTrials.gov (registered trial with posted results)",
            source="ClinicalTrials.gov",
            all_sources=["ClinicalTrials.gov"],
            abstract=(f"{summary} Registered trial {nct}" + (f" ({design_txt})" if design_txt else "")
                      + "; results posted on ClinicalTrials.gov, no linked publication.").strip(),
            url=f"https://clinicaltrials.gov/study/{nct}",
        ))
    return pmids, records, neither


async def search_clinicaltrials(
    expression: str,
    max_results: int = 50,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    pubmed_api_key: str = "",
) -> List[PaperRecord]:
    if not expression.strip():
        return []
    params = {"query.term": expression, "pageSize": str(min(max_results, 100)), "fields": CT_FIELDS,
              "countTotal": "true"}
    if year_from or year_to:
        params["filter.advanced"] = (f"AREA[StudyFirstPostDate]RANGE[{f'{year_from}-01-01' if year_from else 'MIN'},"
                                     f"{f'{year_to}-12-31' if year_to else 'MAX'}]")
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "CiteLume/1.0"}) as client:
            resp = await client.get(CT_STUDIES_URL, params=params)
            resp.raise_for_status()
            studies = resp.json().get("studies") or []
    except Exception as exc:
        print(f"[ClinicalTrials.gov Connector Warning] {exc}")
        report_source_failure("ClinicalTrials.gov", exc)
        return []
    pmids, registry, _ = parse_studies(studies)
    linked: List[PaperRecord] = []
    if pmids:
        query = " OR ".join(f"{p}[uid]" for p in pmids[:100])
        if " OR " not in query:
            query = f"{query} OR {query}"  # keeps the PubMed client from rewriting a single-ID query
        linked = await search_pubmed(query, len(pmids[:100]), pubmed_api_key, year_from, year_to, broaden=False)
    for p in linked:
        p.source = "ClinicalTrials.gov"
        p.all_sources = ["ClinicalTrials.gov"]
        p.venue = p.venue or "Linked from ClinicalTrials.gov"
    return linked + registry
