"""
The corpus layer: turning the raw records returned by the searches into one record per study.

Merging works on copies, so the records handed in are never changed and the same input always gives the same
output. Selection (services/paper_selection.py) then runs on the merged records.
"""
from __future__ import annotations

import re
from typing import Dict, List

from models import PaperRecord


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _is_preprint_record(p: PaperRecord) -> bool:
    v_low = (p.venue or "").lower()
    s_low = (p.source or "").lower()
    if s_low in ("arxiv", "biorxiv") or "biorxiv" in v_low or "medrxiv" in v_low or "arxiv" in v_low:
        return True
    if p.doi and p.doi.startswith("10.1101/"):
        return True
    return False


def merge_duplicates(raw_papers: List[PaperRecord]) -> List[PaperRecord]:
    """
    One record per study, in first-seen order. Records are matched on DOI, arXiv ID, PubMed ID, Semantic Scholar
    ID, then normalised title, which also links a preprint to its published version. The merged record keeps the
    longest abstract, the highest sourced citation count, the journal DOI over a preprint DOI, and every source,
    search and publication type that returned it. Input records are not modified.
    """
    merged_by_key: Dict[str, PaperRecord] = {}
    doi_to_key: Dict[str, str] = {}
    arxiv_to_key: Dict[str, str] = {}
    pmid_to_key: Dict[str, str] = {}
    s2_to_key: Dict[str, str] = {}
    title_to_key: Dict[str, str] = {}

    for original in raw_papers:
        norm_title = _normalize_title(original.title)
        if not norm_title:
            continue
        p = original.model_copy(deep=True)

        clean_doi = p.doi.strip().lower() if p.doi else None
        clean_arxiv = re.sub(r"v\d+$", "", p.arxiv_id.strip()) if p.arxiv_id else None
        clean_pmid = p.pmid.strip() if p.pmid else None
        clean_s2 = p.s2_id.strip() if p.s2_id else None

        existing_key = None
        if clean_doi and clean_doi in doi_to_key:
            existing_key = doi_to_key[clean_doi]
        elif clean_arxiv and clean_arxiv in arxiv_to_key:
            existing_key = arxiv_to_key[clean_arxiv]
        elif clean_pmid and clean_pmid in pmid_to_key:
            existing_key = pmid_to_key[clean_pmid]
        elif clean_s2 and clean_s2 in s2_to_key:
            existing_key = s2_to_key[clean_s2]
        elif norm_title in title_to_key:
            existing_key = title_to_key[norm_title]

        is_prep = _is_preprint_record(p)

        if existing_key and existing_key in merged_by_key:
            curr = merged_by_key[existing_key]
            curr_was_prep = _is_preprint_record(curr)
            if p.source and p.source not in curr.all_sources:
                curr.all_sources.append(p.source)
            if not curr.doi and clean_doi:
                curr.doi = clean_doi
            elif clean_doi and not clean_doi.startswith("10.1101/") and (curr.doi or "").startswith("10.1101/"):
                # Prefer peer-reviewed journal DOI over bioRxiv preprint DOI
                curr.doi = clean_doi
                curr.venue = p.venue or curr.venue
            if not curr.arxiv_id and clean_arxiv:
                curr.arxiv_id = clean_arxiv
            if not curr.pmid and clean_pmid:
                curr.pmid = clean_pmid
            if not curr.s2_id and clean_s2:
                curr.s2_id = clean_s2
            if len(p.abstract or "") > len(curr.abstract or ""):
                curr.abstract = p.abstract
            if p.citation_count > curr.citation_count:
                curr.citation_count = p.citation_count
                curr.citation_source = p.citation_source or curr.citation_source
            curr.retrieved_by = list(dict.fromkeys(curr.retrieved_by + p.retrieved_by))
            curr.publication_types = list(dict.fromkeys(curr.publication_types + p.publication_types))
            if not curr.pdf_url and p.pdf_url:
                curr.pdf_url = p.pdf_url
            curr.references = list(dict.fromkeys(curr.references + p.references))
            curr.cited_by = list(dict.fromkeys(curr.cited_by + p.cited_by))

            # Track preprint + published linkage for the same underlying study
            if (curr_was_prep and not is_prep) or (not curr_was_prep and is_prep):
                curr.version_status = "preprint_with_published_link"
        else:
            key = p.id or f"study:{len(merged_by_key) + 1}"
            p.canonical_study_id = f"study:{norm_title[:32]}"
            p.version_status = "preprint" if is_prep else "published"
            if clean_doi:
                p.doi = clean_doi
            if clean_arxiv:
                p.arxiv_id = clean_arxiv
            if not p.all_sources and p.source:
                p.all_sources = [p.source]
            merged_by_key[key] = p
            title_to_key[norm_title] = key
            if clean_doi:
                doi_to_key[clean_doi] = key
            if clean_arxiv:
                arxiv_to_key[clean_arxiv] = key
            if clean_pmid:
                pmid_to_key[clean_pmid] = key
            if clean_s2:
                s2_to_key[clean_s2] = key

    return list(merged_by_key.values())
