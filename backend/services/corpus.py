"""
The corpus layer: turning the raw records returned by the searches into one record per study.

Merging works on copies, so the records handed in are never changed and the same input always gives the same
output. Selection (services/paper_selection.py) then runs on the merged records.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from models import Corpus, CorpusCandidate, PaperRecord, ReviewProtocol, SelectionReport, UniversalQuestionUnderstanding


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
            # A user's PDF supplies full text, but its guessed abstract never replaces a database abstract
            from_pdf = p.source == "User PDF"
            if len(p.abstract or "") > len(curr.abstract or "") and not (from_pdf and (curr.abstract or "").strip()):
                curr.abstract = p.abstract
            if p.full_text_sections and not curr.full_text_sections:
                curr.full_text_source, curr.full_text_sections = p.full_text_source, dict(p.full_text_sections)
            if p.citation_count > curr.citation_count:
                curr.citation_count = p.citation_count
                curr.citation_source = p.citation_source or curr.citation_source
            curr.retrieved_by = list(dict.fromkeys(curr.retrieved_by + p.retrieved_by))
            curr.publication_types = list(dict.fromkeys(curr.publication_types + p.publication_types))
            if not curr.pdf_url and p.pdf_url:
                curr.pdf_url = p.pdf_url
            curr.references = list(dict.fromkeys(curr.references + p.references))
            curr.cited_by = list(dict.fromkeys(curr.cited_by + p.cited_by))
            curr.mesh_concepts = list(dict.fromkeys(curr.mesh_concepts + p.mesh_concepts))

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


# ---------------------------------------------------------------------------------------------------------------
# The stored corpus: selection with everything needed to audit and replay it
# ---------------------------------------------------------------------------------------------------------------
# A systematic-style review keeps every word-eligible record so the model can screen them before the limit applies
PRISMA_CANDIDATE_POOL = 400


def corpus_hash(corpus: "Corpus") -> str:
    """Identifies a selection: the protocol version, the inputs, the records it ran on and the studies it chose.
    Two corpora with the same hash made the same selection from the same records."""
    payload = {
        "protocol_version": corpus.protocol_version,
        "query": corpus.query,
        "ranking_query": corpus.ranking_query,
        "method": corpus.selection_method,
        "study_limit": corpus.study_limit,
        "pool_limit": corpus.pool_limit,
        "reference_year": corpus.reference_year,
        "records": [r.id for r in corpus.records],
        "protocol": corpus.protocol.model_dump(exclude={"suggestions", "notes"}) if corpus.protocol else None,
        "included": list(corpus.included_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def select_for_corpus(
    raw_papers: List[PaperRecord],
    query: str,
    ranking_query: str,
    study_limit: int,
    understanding: Optional["UniversalQuestionUnderstanding"],
    method: str = "narrative",
    stats: Optional[Dict[str, int]] = None,
    reference_year: Optional[int] = None,
    protocol: Optional["ReviewProtocol"] = None,
) -> Tuple[List[PaperRecord], "SelectionReport", "Corpus"]:
    """
    Merges duplicates and runs selection, keeping what a stored corpus needs. Returns the chosen records (for a
    systematic-style review: the eligible candidates, to be screened before the limit applies), the selection
    report, and a draft corpus to complete with `finish_corpus` once the included studies are final.
    """
    from services.paper_selection import PROTOCOL_VERSION, SelectionDetails, select_studies

    unique = merge_duplicates(raw_papers)
    snapshot = [p.model_copy(deep=True) for p in unique]
    if stats is not None:
        stats["unique_records"] = len(unique)
    pool_limit = PRISMA_CANDIDATE_POOL if method in ("systematic_prisma", "scoping") else study_limit
    details = SelectionDetails()
    chosen, report = ([], None)
    if unique:
        chosen, report = select_studies(unique, ranking_query, pool_limit, understanding, stats,
                                        records_found=len(raw_papers), method=method, details=details,
                                        reference_year=reference_year,
                                        eligibility=protocol.eligibility if protocol else None)
    now = datetime.now(timezone.utc)
    corpus = Corpus(
        corpus_id=uuid.uuid4().hex,
        created_at=now.isoformat(timespec="seconds"),
        search_date=now.strftime("%Y-%m-%d"),
        protocol_version=PROTOCOL_VERSION,
        query=query,
        ranking_query=ranking_query,
        selection_method=method,
        study_limit=study_limit,
        pool_limit=pool_limit,
        reference_year=details.reference_year or now.year,
        records_found=len(raw_papers),
        records=snapshot,
        protocol=protocol.model_copy(deep=True) if protocol else None,
        candidates=[
            CorpusCandidate(record_id=p.id, rank=p.selection_rank or i, relevance=p.relevance or "",
                            study_type=p.study_type or "", scores=dict(p.selection_breakdown or {}))
            for i, p in enumerate(details.candidates, start=1)
        ],
        excluded=dict(details.excluded),
    )
    return chosen, report, corpus


def finish_corpus(corpus: "Corpus", included: List[PaperRecord], report: Optional["SelectionReport"]) -> "Corpus":
    """Records the final included studies (after any screening) and the report, and stamps the content hash."""
    corpus.included_ids = [p.id for p in included]
    corpus.selection_report = report.model_copy(deep=True) if report is not None else None
    if report is not None:
        screened_out = {d.record_id for d in report.screening_decisions if not d.include}
        for rid in screened_out:
            corpus.excluded.setdefault(rid, "Not about the question on title and abstract screening (judged by the model)")
    corpus.content_hash = corpus_hash(corpus)
    return corpus


def replay_selection(corpus: "Corpus") -> List[str]:
    """
    Re-runs selection from the stored records and inputs and returns the included record IDs, in order. No model
    and no network are used: a systematic-style review's screening is re-applied from its stored decisions.
    """
    return [p.id for p in replay_included(corpus)[0]]


def replay_included(corpus: "Corpus") -> Tuple[List[PaperRecord], Optional["SelectionReport"]]:
    """The included studies as selection produces them from the stored records (numbered, scored and typed), and
    the replayed selection report."""
    from services.ai_screening import apply_screening_decisions
    from services.paper_selection import select_studies

    understanding = None
    if corpus.plan is not None:
        understanding = corpus.plan.selection_understanding or corpus.plan.question_understanding
    records = [p.model_copy(deep=True) for p in corpus.records]
    if not records:
        return [], None
    chosen, report = select_studies(records, corpus.ranking_query, corpus.pool_limit, understanding,
                                    records_found=corpus.records_found, method=corpus.selection_method,
                                    reference_year=corpus.reference_year or None,
                                    eligibility=corpus.protocol.eligibility if corpus.protocol else None)
    if corpus.selection_method in ("systematic_prisma", "scoping") and corpus.selection_report is not None:
        chosen = apply_screening_decisions(chosen, report, corpus.selection_report.screening_decisions,
                                           corpus.study_limit, corpus.selection_report.screening_model)
    chosen = apply_forced_includes(chosen, corpus, report)
    attach_user_full_texts(chosen, corpus)
    return chosen, report


def derive_corpus(corpus: "Corpus", study_limit: int, overrides=None) -> "Corpus":
    """
    A corpus with the same records, protocol and decisions but a different study limit. Selection is re-run on the
    stored records (no search, no model); a systematic-style or scoping review re-applies its stored screening.
    """
    derived = corpus.model_copy(deep=True)
    now = datetime.now(timezone.utc)
    derived.corpus_id = uuid.uuid4().hex
    derived.created_at = now.isoformat(timespec="seconds")
    derived.root_id = corpus.root_id or corpus.corpus_id
    derived.study_limit = study_limit
    if derived.selection_method == "narrative":
        derived.pool_limit = study_limit
    if derived.protocol is not None:
        derived.protocol.study_limit = study_limit
        derived.protocol.notes.append(
            f"After the search, the user chose to analyse {study_limit} studies (the search had included "
            f"{len(corpus.included_ids)}).")
    derived.analysis, derived.analysis_model, derived.analysis_runs = None, "", 0
    derived.analysed_at = ""
    if overrides and derived.selection_report is not None:
        # The user's decisions after reviewing the model's screening (each with a reason)
        by_id = {o.record_id: o for o in overrides}
        for d in derived.selection_report.screening_decisions:
            if d.record_id in by_id:
                d.user_override, d.override_reason = by_id[d.record_id].include, by_id[d.record_id].reason.strip()
        if derived.protocol is not None:
            derived.protocol.notes.append(
                f"After the search, the user reviewed the model's screening and changed {len(overrides)} decision(s).")
    papers, report = replay_included(derived)
    if report is not None and derived.selection_report is not None:
        # Keep what the first selection recorded that a replay does not recompute
        report.screening_decisions = list(derived.selection_report.screening_decisions)
        report.screening_model = derived.selection_report.screening_model
        report.screening_criteria = list(derived.selection_report.screening_criteria)
        report.screening_prompt_version = derived.selection_report.screening_prompt_version
        report.screening_date = derived.selection_report.screening_date
    return finish_corpus(derived, papers, report)


def apply_forced_includes(chosen: List[PaperRecord], corpus: "Corpus",
                          report: Optional["SelectionReport"] = None) -> List[PaperRecord]:
    """Adds the studies the user chose to include despite the criteria (each with its recorded reason), after the
    selected ones. Applied the same way when a corpus is built and when it is replayed."""
    from services.paper_selection import classify_study_type

    have = {p.id for p in chosen}
    by_id = {r.id: r for r in corpus.records}
    for rid, reason in corpus.forced_includes.items():
        if rid in have or rid not in by_id:
            continue
        rec = by_id[rid].model_copy(deep=True)
        rec.study_type = rec.study_type or classify_study_type(rec)
        rec.selection_reason = f"Included by the user despite the criteria: {reason}"
        rec.relevance = rec.relevance if rec.relevance and rec.relevance != "off_topic" else "partly"
        rec.citation_index = len(chosen) + 1
        chosen.append(rec)
        if report is not None:
            report.included = len(chosen)
            report.notes.append(f"Included by the user despite the eligibility criteria: \"{rec.title}\" ({reason}).")
    return chosen


def attach_user_full_texts(papers: List[PaperRecord], corpus: "Corpus") -> None:
    """Puts the full text of the user's PDFs back on their studies (full text is not part of a stored record)."""
    for p in papers:
        ft = corpus.user_full_texts.get(p.id)
        if ft and not p.full_text_sections:
            p.full_text_source, p.full_text_sections = ft.get("source", ""), dict(ft.get("sections", {}))
