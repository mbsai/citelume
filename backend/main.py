import time
import re
import asyncio
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import (
    SearchRequest,
    ExpandNodeRequest,
    ExportRequest,
    ResearchResponse,
    PaperRecord,
    ProviderModelsRequest,
    TestLlmRequest,
    TestSourceKeyRequest,
    SearchFacet,
    FacetApiQueries,
    ApiKeysConfig,
    Corpus,
    CorpusSummary,
    ReselectRequest,
    ScreeningOverridesRequest,
    RiskOfBiasEditRequest,
    StudyRiskOfBias,
    ImportRequest,
    ImportResult,
    ImportSummary,
    PdfUploadRequest,
    PdfUploadResult,
    UserPdfSummary,
    ProtocolPreviewRequest,
    ProtocolProposeRequest,
    ReviewProtocol,
    SearchPreview,
    ModuleRequest,
    QueryDecompositionPlan,
    SelectionReport,
)
from services.pubmed_client import search_pubmed
from services.arxiv_client import search_arxiv
from services.biorxiv_client import search_biorxiv
from services.s2_client import search_semantic_scholar, expand_paper_neighbors
from services.hf_scraper import search_huggingface_hub
from services.llm_service import (
    fetch_available_hf_llm_models,
    fetch_provider_models,
    test_llm_connection,
    test_source_api_key,
    resolve_model_capabilities,
    begin_llm_request_health,
    llm_request_stats,
    classify_model_scale,
)
from frameworks.reporting import build_review_protocol_report
from services.prisma_report import compose_prisma_report
from services.ai_screening import screen_and_limit, screening_criteria
from services.risk_of_bias_assessment import apply_edit as apply_rob_edit, assess_risk_of_bias
from services.paper_selection import label_subquestions
from services import corpus_store
from services.corpus import (
    apply_forced_includes,
    attach_user_full_texts,
    derive_corpus,
    finish_corpus,
    replay_included,
    select_for_corpus,
)
from services import importers
from services.importers import match_pdf, parse_export, parse_pdf
from services.review_protocol import default_protocol, plan_from_protocol, propose_protocol
from services.search_counts import preview_searches
from services.study_limits import ceiling, estimate_minutes, resolve_study_limit, selection_method
from services import demo_mode
from services.citation_enrichment import enrich_citation_counts
from services.query_planning import results_per_search, searches_arxiv
from services.source_budget import begin_search_budget
from services.fulltext import attach_open_access_full_texts
from services.source_health import begin_source_health, report_source_failure, set_openalex_api_key, source_failure_warnings
from services.query_utils import generate_multi_api_decomposition_plan, understand_research_question
from services.identifier_query import identifier_in_query, resolve_identifier_title
from services.graph_builder import deduplicate_and_rank_papers, build_literature_graph
from services.synthesis_engine import (
    generate_bibliography,
    generate_synthesis_and_extractions,
    build_knowledge_gap_matrix,
    build_shared_evidence_base,
    begin_synthesis_scope,
)
from services.claim_synthesis import get_last_synthesis_package
from services.evidence_table_engine import (
    build_structured_evidence_matrix,
    export_evidence_matrix_to_csv,
    export_evidence_matrix_to_json,
)
from services.network_engine import (
    export_network_to_graphml,
    export_network_nodes_csv,
    export_network_edges_csv,
    export_network_json,
)
from services.verification_gates import run_verification_and_quality_gates
from services.export_service import (
    generate_markdown_report,
    generate_txt_report,
)
from services.pdf_report import generate_pdf_report

app = FastAPI(
    title="CiteLume API",
    description="Universal, Domain-Adaptive, Evidence-Grounded Literature Research Engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "CiteLume",
        "version": "2.0.0",
        "features": [
            "universal_question_understanding_engine",
            "composable_domain_and_methodological_profiles",
            "universal_search_planner_F1_to_F6",
            "validated_5_api_query_compilers",
            "multi_round_adaptive_retrieval_and_selection",
            "shared_scientific_evidence_base",
            "deterministic_verification_quality_gates",
            "txt_md_pdf_exporter",
        ],
    }


@app.get("/api/hf-models")
async def get_hf_models(hf_api_key: str = "", search: str = "") -> Dict[str, Any]:
    models = await fetch_available_hf_llm_models(hf_api_key=hf_api_key, search_filter=search)
    return {"models": models, "count": len(models)}


def _demo_guard_url(provider: str, base_url: str) -> None:
    # On the public demo the server must never be pointed at its own network (local model servers etc.)
    if demo_mode.demo_enabled() and (provider == "ollama" or (base_url and demo_mode._is_private_url(base_url))):
        raise HTTPException(400, "Local models (Ollama or a local server address) cannot be used on the public demo.")


@app.get("/api/demo-status")
async def get_demo_status(request: Request) -> Dict[str, Any]:
    if not demo_mode.demo_enabled():
        return {"demo": False}
    ip = demo_mode.client_ip(request.headers.get("x-forwarded-for"), request.client.host if request.client else None)
    return demo_mode.status_for(ip)


@app.post("/api/provider-models")
async def get_provider_models(req: ProviderModelsRequest) -> Dict[str, Any]:
    _demo_guard_url(req.llm_provider, req.llm_base_url)
    if demo_mode.demo_enabled() and not (req.llm_api_key or "").strip():
        models = [{"id": m, "name": m, "is_free": True} for m in demo_mode.allowed_models()]
        return {
            "detected_provider": "openrouter",
            "detected_base_url": "",
            "source_label": "the free demo (NVIDIA free models only)",
            "live_fetched": True,
            "count": len(models),
            "free_count": len(models),
            "models": models,
            "warning": "",
        }
    return await fetch_provider_models(
        provider=req.llm_provider,
        api_key=req.llm_api_key,
        base_url=req.llm_base_url,
        hf_api_key=req.hf_api_key,
    )


@app.get("/api/model-scale")
async def get_model_scale(provider: str = "", model: str = "") -> Dict[str, Any]:
    tier, reason = classify_model_scale(provider, model)
    return {"tier": tier, "reason": reason}


@app.post("/api/test-llm")
async def test_llm_endpoint(req: TestLlmRequest) -> Dict[str, Any]:
    _demo_guard_url(req.llm_provider, req.llm_base_url)
    if demo_mode.demo_enabled() and not (req.llm_api_key or "").strip():
        return {
            "success": demo_mode.server_key_configured(),
            "latency_ms": 0,
            "message": ("No key needed: the free demo runs on NVIDIA's free models via OpenRouter."
                        if demo_mode.server_key_configured()
                        else "The demo's shared key is not configured; enter your own LLM API key."),
        }
    return await test_llm_connection(req)


@app.post("/api/test-source")
async def test_source_endpoint(req: TestSourceKeyRequest) -> Dict[str, Any]:
    return await test_source_api_key(source=req.source, api_key=req.api_key)


async def _harvest_facet_queries(
    facet: SearchFacet,
    max_per_source: int,
    req: SearchRequest,
    fallback_entities: str,
    domain: str = "",
) -> Tuple[List[PaperRecord], Dict[str, int]]:
    cfg = req.config
    tasks = []
    task_names = []
    q_map = facet.queries
    # Only the core search may loosen itself when it finds nothing; an aspect search that finds nothing stays empty
    broaden = facet.kind == "core"
    use_arxiv = cfg.enable_arxiv and searches_arxiv(domain, facet.kind)

    if cfg.enable_pubmed:
        tasks.append(
            search_pubmed(
                q_map.pubmed or fallback_entities,
                max_per_source,
                cfg.pubmed_api_key,
                req.year_from,
                req.year_to,
                fallback_query=fallback_entities,
                broaden=broaden,
            )
        )
        task_names.append("PubMed")
    if use_arxiv:
        tasks.append(
            search_arxiv(
                q_map.arxiv or fallback_entities,
                max_per_source,
                req.year_from,
                req.year_to,
                fallback_query=fallback_entities,
                broaden=broaden,
            )
        )
        task_names.append("arXiv")
    if cfg.enable_biorxiv:
        tasks.append(
            search_biorxiv(
                q_map.biorxiv_medrxiv or fallback_entities,
                max_per_source,
                req.year_from,
                req.year_to,
                fallback_query=fallback_entities,
                broaden=broaden,
            )
        )
        task_names.append("bioRxiv")
    if cfg.enable_s2:
        tasks.append(
            search_semantic_scholar(
                q_map.semantic_scholar or fallback_entities,
                max_per_source,
                cfg.s2_api_key,
                req.year_from,
                req.year_to,
            )
        )
        task_names.append("Semantic Scholar")

    async def within_budget(name: str, coro):
        # One slow database must not hold up the others: past the limit its results for this search are skipped
        try:
            return await asyncio.wait_for(coro, timeout=SOURCE_SEARCH_BUDGET_S)
        except asyncio.TimeoutError:
            report_source_failure(name, RuntimeError(f"no reply within {SOURCE_SEARCH_BUDGET_S:.0f}s for one search"))
            return []

    results = await asyncio.gather(*[within_budget(n, t) for n, t in zip(task_names, tasks)], return_exceptions=True)
    papers: List[PaperRecord] = []
    counts: Dict[str, int] = {name: 0 for name in task_names}

    for name, res in zip(task_names, results):
        if isinstance(res, list):
            for p in res:
                p.sub_query = facet.facet_name
                p.retrieved_by = [facet.facet_name]
            papers.extend(res)
            counts[name] = len(res)
    return papers, counts


# Longest any one database may take for one search before it is skipped for that search
SOURCE_SEARCH_BUDGET_S = 20.0

# Pipeline stages reported to the frontend progress indicator, in execution order
SEARCH_STAGES: List[str] = [
    "Understanding the research question and planning source-specific searches...",
    "Harvesting papers and preprints from the enabled academic sources...",
    "Selecting papers; retrieving open-access full text and scanning Hugging Face Hub...",
    "Extracting study data and formulating evidence-backed claims (LLM)...",
    "Writing the cited synthesis and analysing knowledge gaps in parallel (LLM)...",
    "Building the literature network map and running quality checks...",
]

# progress_id -> {"stage": 1-based index, "total": N, "label": str, "detail": str}
_SEARCH_PROGRESS: Dict[str, Dict[str, Any]] = {}


_STAGE_CLOCK: Dict[int, float] = {}


def _report_stage(req: SearchRequest, stage: int, detail: str = "") -> None:
    # Stage timings go to the server log (no query text), so slow runs can be diagnosed
    now = time.monotonic()
    started = _STAGE_CLOCK.setdefault(id(req), now)
    print(f"[search-stage] {req.mode} stage {stage}/{len(SEARCH_STAGES)} at +{now - started:.0f}s {detail}".rstrip(), flush=True)
    if stage >= len(SEARCH_STAGES) or len(_STAGE_CLOCK) > 200:
        _STAGE_CLOCK.pop(id(req), None)
    if not req.progress_id:
        return
    _SEARCH_PROGRESS[req.progress_id] = {
        "stage": stage,
        "total": len(SEARCH_STAGES),
        "label": SEARCH_STAGES[stage - 1],
        "detail": detail,
    }


@app.get("/api/search-progress/{progress_id}")
async def get_search_progress(progress_id: str) -> Dict[str, Any]:
    return _SEARCH_PROGRESS.get(progress_id) or {"stage": 0, "total": len(SEARCH_STAGES), "label": "", "detail": ""}


async def _prepare_query(req: SearchRequest) -> Tuple[SearchRequest, str]:
    """
    Make the query searchable before any work (or demo allowance) is spent: a bare DOI / PubMed / arXiv identifier
    is replaced by its paper's title, and a query with no research topic is refused with advice.
    Returns the request to run and a note for the results ("" if nothing changed).
    """
    q = re.sub(r"\s+", " ", req.query or "").strip()
    if len(q) > 4000:
        q = q[:4000]
    note = ""
    ident = identifier_in_query(q)
    if ident:
        title = await resolve_identifier_title(*ident)
        if not title:
            raise HTTPException(400, f"Could not look up the paper for {ident[0].upper()} {ident[1]}. "
                                     "Please type the research topic or the paper's title instead.")
        note = f"Query interpreted: searched for literature related to the paper \"{title}\" ({ident[0].upper()} {ident[1]})."
        q = title
    concepts = understand_research_question(q).concepts
    if not re.search(r"[^\W\d_]{2,}", q) or not (concepts.concept_phrases or concepts.required_constraints):
        raise HTTPException(400, "Please describe a research topic to search, for example \"statins after myocardial "
                                 "infarction\" or \"How does sleep affect memory in older adults?\". General words "
                                 "alone (such as \"state of the art\" or \"what is it about\") cannot be searched.")
    return (req.model_copy(update={"query": q}) if q != req.query else req), note


def _with_note(result: ResearchResponse, note: str) -> ResearchResponse:
    if note:
        result.warnings.insert(0, note)
    return result


@app.post("/api/search", response_model=ResearchResponse)
async def execute_research_search(req: SearchRequest, request: Request) -> ResearchResponse:
    """The whole pipeline in one call (kept for the page): build the corpus, then analyse it."""
    req, note = await _prepare_query(req)
    try:
        result = await _with_demo_limits(
            request, req.mode, req.config, lambda cfg: _run_research_search(req.model_copy(update={"config": cfg})))
        return _with_note(result, note)
    finally:
        if req.progress_id:
            _SEARCH_PROGRESS.pop(req.progress_id, None)


async def _with_demo_limits(request: Request, mode: str, config: ApiKeysConfig, work, charge: bool = True):
    """Runs `work(config)`. On the public demo: shared NVIDIA free models, a per-address daily allowance (charged
    when `charge`; given back if the run fails) and a limit on concurrent runs. Elsewhere it simply runs."""
    if not demo_mode.demo_enabled():
        return await work(config)
    ip = demo_mode.client_ip(request.headers.get("x-forwarded-for"), request.client.host if request.client else None)
    try:
        cfg, uses_server_key = demo_mode.apply_demo_config(config)
        bucket = demo_mode.bucket_for(mode, uses_server_key)
        if charge:
            demo_mode.take_quota(ip, bucket)
    except demo_mode.DemoLimitError as exc:
        raise HTTPException(exc.status, str(exc))
    slots = demo_mode.search_slots()
    if slots.locked():
        if charge:
            demo_mode.USAGE.give_back(ip, bucket)
        raise HTTPException(503, "The demo is busy with other searches. Please try again in a minute.")
    completed = False
    try:
        async with slots:
            result = await work(cfg)
            completed = True
            return result
    finally:
        if charge and not completed:
            demo_mode.USAGE.give_back(ip, bucket)  # a crashed run does not use up the visitor's allowance


# ---------------------------------------------------------------------------------------------------------------
# The corpus and its modules
#
#   POST /api/corpus                    plan, search and select; store the corpus (charged like a search on the demo)
#   GET  /api/corpus/{id}               the stored corpus: records, inputs, decisions (for audit and replay)
#   POST /api/corpus/{id}/analyse       extraction, synthesis, evidence table, gaps and map (run once, then stored)
#   POST /api/corpus/{id}/synthesis | evidence-table | gaps     that module's part of the analysis
#   POST /api/corpus/{id}/map           the map; built from the included studies alone if no analysis exists yet
#
# The evidence table and the gaps still depend on the per-study extraction and claims that the synthesis run
# produces, so the three share one stored analysis. Separating them is later work (docs/PUBLICATION_PLAN.md).
# ---------------------------------------------------------------------------------------------------------------
MODULE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "synthesis": ("executive_answer", "sections", "consensus", "bibliography", "review_protocol", "quality_report",
                  "warnings"),
    "evidence-table": ("papers", "evidence_matrix", "synthesis_evidence"),
    "gaps": ("gap_matrix",),
    "map": ("graph",),
}
# On the demo with the shared key, each corpus may be analysed once; a re-run needs a visitor's own key
DEMO_ANALYSES_PER_CORPUS = 1


def _corpus_or_404(corpus_id: str) -> Corpus:
    corpus = corpus_store.load(corpus_id)
    if corpus is None:
        raise HTTPException(404, "No stored corpus with this id. Corpora expire; run the search again.")
    return corpus


def _summary(run: "CorpusRun") -> CorpusSummary:
    c = run.corpus
    return CorpusSummary(
        corpus_id=c.corpus_id, corpus_hash=c.content_hash, created_at=c.created_at,
        protocol_version=c.protocol_version, query=c.query, mode=c.mode, review_framework=c.review_framework,
        records_found=c.records_found, unique_records=len(c.records), candidates=len(c.candidates),
        source_stats={k: v for k, v in c.source_stats.items() if v}, included=run.papers,
        selection_report=run.selection_report, warnings=list(c.search_warnings), analysed=c.analysis is not None,
        selection_method=c.selection_method, study_limit=c.study_limit,
        eligible=run.selection_report.eligible if run.selection_report else len(run.papers),
        max_study_limit=ceiling(c.mode, c.review_framework), estimated_minutes=estimate_minutes(len(run.papers)),
        imports=c.imports, user_pdfs=c.user_pdfs,
    )


@app.post("/api/protocol/propose", response_model=ReviewProtocol)
async def propose_review_protocol(req: ProtocolProposeRequest) -> ReviewProtocol:
    """The default review protocol for a question, for the user to confirm or edit before searching. With
    `suggest`, the configured model's extra synonyms and aspects come back as suggestions (never applied)."""
    prepared, _note = await _prepare_query(SearchRequest(query=req.query, mode=req.mode))
    cfg = req.config
    if demo_mode.demo_enabled() and req.suggest:
        cfg, _ = demo_mode.apply_demo_config(cfg)
    if req.suggest:
        begin_llm_request_health()
    return await propose_protocol(prepared.query, mode=req.mode, review_framework=req.review_framework,
                                  purpose=req.purpose, domain_override=req.domain, suggest=req.suggest, config=cfg)


@app.post("/api/protocol/preview", response_model=SearchPreview)
async def preview_review_protocol(req: ProtocolPreviewRequest) -> SearchPreview:
    """The search strings a protocol will run on every database, and how many records each database holds for
    the core search, before anything is retrieved."""
    if not any(c.name.strip() and c.role != "comparison" for c in req.protocol.concepts):
        raise HTTPException(400, "The protocol needs at least one concept to search for.")
    cfg = demo_mode.fill_source_keys(req.config) if demo_mode.demo_enabled() else req.config
    begin_source_health()
    begin_search_budget()
    set_openalex_api_key(cfg.openalex_api_key)
    return await preview_searches(req.protocol, req.mode, cfg)


USER_PDF_SOURCE = "Full texts supplied by the user"


def _local_only() -> None:
    if demo_mode.demo_enabled():
        raise HTTPException(403, "Importing records and PDFs is available only when CiteLume runs on your own "
                                 "computer: exports and full texts are licensed to you and are not uploaded to "
                                 "the public demo.")


def _decode(content_base64: str) -> bytes:
    try:
        return base64.b64decode(content_base64, validate=False)
    except (ValueError, binascii.Error):
        raise HTTPException(400, "The file could not be read.")


@app.post("/api/imports", response_model=ImportResult)
async def import_records(req: ImportRequest) -> ImportResult:
    """Parses a database export (RIS, BibTeX, CSV, Web of Science or PubMed text) and holds its records until a
    corpus is built with its import_id."""
    _local_only()
    try:
        records, fmt, source = parse_export(_decode(req.content_base64), req.filename, req.source)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    key = importers.hold("export", {"records": records, "format": fmt, "source": source, "filename": req.filename,
                                    "search_string": req.search_string.strip()})
    return ImportResult(import_id=key, filename=req.filename, source=source, format=fmt, records=len(records),
                        with_abstract=sum(1 for r in records if len(r.abstract) >= 20),
                        sample_titles=[r.title for r in records[:3]])


@app.post("/api/imports/pdf", response_model=PdfUploadResult)
async def import_pdf(req: PdfUploadRequest) -> PdfUploadResult:
    """Reads a PDF's text layer and holds it until a corpus is built with its pdf_id."""
    _local_only()
    try:
        record = parse_pdf(_decode(req.content_base64), req.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    key = importers.hold("pdf", {"record": record, "filename": req.filename})
    return PdfUploadResult(pdf_id=key, filename=req.filename, title=record.title, doi=record.doi or "",
                           year=record.year, text_characters=sum(len(v) for v in record.full_text_sections.values()),
                           sections=sorted(record.full_text_sections))


def _user_material(req: SearchRequest, raw: List[PaperRecord], source_stats: Dict[str, int]):
    """Adds the user's held exports and PDFs to the raw records. Returns the import summaries, the PDF records,
    their summaries and the PDFs to include regardless of the criteria."""
    if not (req.import_ids or req.user_pdfs):
        return [], [], [], {}
    _local_only()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    imports: List[ImportSummary] = []
    for key in req.import_ids:
        batch = importers.held(key, "export")
        if batch is None:
            raise HTTPException(400, "An imported file has expired; upload it again.")
        recs = [r.model_copy(deep=True) for r in batch["records"]]
        raw.extend(recs)
        label = f"{batch['source']} (imported by the user)"
        source_stats[label] = source_stats.get(label, 0) + len(recs)
        imports.append(ImportSummary(source=batch["source"], format=batch["format"], filename=batch["filename"],
                                     records=len(recs), search_string=batch["search_string"], imported_at=now))
    pdf_records, summaries, forced = [], [], {}
    for choice in req.user_pdfs:
        held_pdf = importers.held(choice.pdf_id, "pdf")
        if held_pdf is None:
            raise HTTPException(400, "An uploaded PDF has expired; upload it again.")
        rec = held_pdf["record"].model_copy(deep=True)
        match = match_pdf(rec, raw, " ".join(rec.full_text_sections.values())[:3000] + " " + rec.title)
        if match is not None:
            # Same DOI and title as the searched record, so the two are merged and the full text joins it
            rec.doi, rec.title = match.doi or rec.doi, match.title
        else:
            source_stats[USER_PDF_SOURCE] = source_stats.get(USER_PDF_SOURCE, 0) + 1
        raw.append(rec)
        pdf_records.append(rec)
        summaries.append(UserPdfSummary(filename=held_pdf["filename"], title=rec.title,
                                        matched_record_id=match.id if match is not None else "",
                                        force_include=choice.force_include, reason=choice.reason.strip()))
        if choice.force_include:
            if not choice.reason.strip():
                raise HTTPException(400, f"Give a reason for including \"{held_pdf['filename']}\" regardless of the criteria.")
            forced[len(pdf_records) - 1] = choice.reason.strip()
    return imports, pdf_records, summaries, forced


def _record_user_pdfs(corpus: Corpus, pdf_records: List[PaperRecord], summaries: List[UserPdfSummary],
                      forced: Dict[int, str]) -> None:
    """Finds which merged study each PDF became, keeps its full text with the corpus, and records forced
    inclusions by study id."""
    for i, rec in enumerate(pdf_records):
        merged = match_pdf(rec, corpus.records) or next((r for r in corpus.records if r.id == rec.id), None)
        if merged is None:
            continue
        summaries[i].record_id = merged.id
        corpus.user_full_texts[merged.id] = {"source": rec.full_text_source, "sections": dict(rec.full_text_sections)}
        if i in forced:
            corpus.forced_includes[merged.id] = forced[i]
    corpus.user_pdfs = summaries


@app.post("/api/corpus", response_model=CorpusSummary)
async def create_corpus(req: SearchRequest, request: Request) -> CorpusSummary:
    req, note = await _prepare_query(req)
    try:
        run = await _with_demo_limits(
            request, req.mode, req.config, lambda cfg: _build_corpus(req.model_copy(update={"config": cfg})))
    finally:
        if req.progress_id:
            _SEARCH_PROGRESS.pop(req.progress_id, None)
    summary = _summary(run)
    if note:
        summary.warnings.insert(0, note)
    return summary


@app.post("/api/corpus/{corpus_id}/reselect", response_model=CorpusSummary)
async def reselect_corpus(corpus_id: str, body: ReselectRequest) -> CorpusSummary:
    """A new corpus from the same records with a different number of studies: no new search and no model (a
    systematic-style or scoping review re-applies its stored screening decisions). The original is unchanged."""
    corpus = _corpus_or_404(corpus_id)
    top = ceiling(corpus.mode, corpus.review_framework)
    if corpus.selection_method != "narrative":
        # Only studies already screened can be chosen without the model, so the limit can only go down
        top = min(top, corpus.study_limit)
    if not (5 <= body.study_limit <= top):
        raise HTTPException(400, f"Choose between 5 and {top} studies.")
    derived = derive_corpus(corpus, body.study_limit)
    corpus_store.save(derived)
    return _summary(_run_from_corpus(derived, ApiKeysConfig()))


@app.post("/api/corpus/{corpus_id}/screening-overrides", response_model=CorpusSummary)
async def override_screening(corpus_id: str, body: ScreeningOverridesRequest) -> CorpusSummary:
    """The user's decisions after reviewing the model's screening, each with a reason. Returns a new corpus from
    the same records (no search, no model) with the decisions applied; the original is unchanged."""
    corpus = _corpus_or_404(corpus_id)
    decided = {d.record_id for d in (corpus.selection_report.screening_decisions if corpus.selection_report else [])}
    for o in body.overrides:
        if o.record_id not in decided:
            raise HTTPException(400, "Only records the model screened can be overridden.")
        if not o.reason.strip():
            raise HTTPException(400, "Give a reason for each change to the model's screening.")
    derived = derive_corpus(corpus, corpus.study_limit, overrides=body.overrides)
    corpus_store.save(derived)
    return _summary(_run_from_corpus(derived, ApiKeysConfig()))


@app.post("/api/corpus/{corpus_id}/risk-of-bias", response_model=StudyRiskOfBias)
async def edit_risk_of_bias(corpus_id: str, body: RiskOfBiasEditRequest) -> StudyRiskOfBias:
    """The reviewer's change to one risk-of-bias judgement, with a reason; recorded in the stored analysis."""
    corpus = _corpus_or_404(corpus_id)
    if corpus.analysis is None:
        raise HTTPException(400, "This corpus has not been analysed yet.")
    if not body.reason.strip():
        raise HTTPException(400, "Give a reason for the change.")
    paper = next((p for p in corpus.analysis.papers if p.id == body.record_id), None)
    rob = paper.elicit.risk_of_bias if paper else None
    if rob is None or rob.overall == "Not assessed":
        raise HTTPException(400, "This study has no risk-of-bias judgement to change.")
    try:
        apply_rob_edit(rob, body.domain, body.judgement, body.reason)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    corpus_store.save(corpus)
    return rob


@app.get("/api/corpus/{corpus_id}", response_model=Corpus)
async def get_corpus(corpus_id: str) -> Corpus:
    return _corpus_or_404(corpus_id)


def _run_from_corpus(corpus: Corpus, cfg: ApiKeysConfig, progress_id: Optional[str] = None) -> "CorpusRun":
    """Rebuilds the included studies from the stored records by replaying selection (no model, no network) and
    refuses if the replay does not reproduce the stored selection."""
    _begin_request_scope(cfg)
    papers, replayed_report = replay_included(corpus)
    if [p.id for p in papers] != list(corpus.included_ids):
        raise HTTPException(409, (
            f"This corpus was selected with {corpus.protocol_version}; replaying it here gives a different set of "
            "studies, so it cannot be analysed reproducibly. Run the search again."))
    plan = corpus.plan.model_copy(deep=True) if corpus.plan else QueryDecompositionPlan()
    req = SearchRequest(query=corpus.query, mode=corpus.mode, review_framework=corpus.review_framework,
                        organization=corpus.organization, year_from=corpus.year_from, year_to=corpus.year_to,
                        config=cfg, progress_id=progress_id)
    return CorpusRun(
        req=req, plan=plan, corpus=corpus, papers=papers,
        selection_report=corpus.selection_report.model_copy(deep=True) if corpus.selection_report else replayed_report,
        source_stats=dict(corpus.source_stats), dedup_stats=dict(corpus.selection_stats),
        sub_queries=[f"[{f.facet_name}] {f.queries.semantic_scholar}" for f in plan.search_facets],
        fallback_entities=" ".join(plan.extracted_technical_entities[:3]) or corpus.query,
    )


async def _stored_analysis(corpus_id: str, mreq: ModuleRequest, request: Request) -> ResearchResponse:
    """The corpus's stored analysis, running it first if there is none (or if a re-run is asked for)."""
    corpus = _corpus_or_404(corpus_id)
    if corpus.analysis is not None and not mreq.rerun:
        return corpus.analysis
    # Allowances are counted on the original corpus, so deriving corpora with other limits does not add analyses
    counter = (corpus_store.load(corpus.root_id) if corpus.root_id else None) or corpus
    if demo_mode.demo_enabled():
        _, uses_server_key = demo_mode.apply_demo_config(mreq.config)
        if uses_server_key and counter.analysis_runs >= DEMO_ANALYSES_PER_CORPUS:
            raise HTTPException(429, "This search has already been analysed with the demo's shared model. "
                                     "Add your own API key to analyse it again with another model.")
    # The first analysis of a search is covered by the allowance charged when its corpus was built
    charge = counter.analysis_runs > 0
    counter.analysis_runs += 1
    if counter is not corpus:
        corpus_store.save(counter)
    try:
        return await _with_demo_limits(
            request, corpus.mode, mreq.config,
            lambda cfg: _analyse(_run_from_corpus(corpus, cfg, mreq.progress_id)), charge=charge)
    except Exception:
        counter.analysis_runs -= 1
        raise
    finally:
        if mreq.progress_id:
            _SEARCH_PROGRESS.pop(mreq.progress_id, None)


@app.post("/api/corpus/{corpus_id}/analyse", response_model=ResearchResponse)
async def analyse_corpus(corpus_id: str, mreq: ModuleRequest, request: Request) -> ResearchResponse:
    return await _stored_analysis(corpus_id, mreq, request)


def _module_payload(corpus: Corpus, result: ResearchResponse, module: str) -> Dict[str, Any]:
    payload = result.model_dump(include=set(MODULE_FIELDS[module]))
    payload.update(corpus_id=corpus.corpus_id, corpus_hash=corpus.content_hash, module=module,
                   analysis_model=corpus.analysis_model, analysed_at=corpus.analysed_at)
    return payload


@app.post("/api/corpus/{corpus_id}/synthesis")
async def corpus_synthesis(corpus_id: str, mreq: ModuleRequest, request: Request) -> Dict[str, Any]:
    result = await _stored_analysis(corpus_id, mreq, request)
    return _module_payload(_corpus_or_404(corpus_id), result, "synthesis")


@app.post("/api/corpus/{corpus_id}/evidence-table")
async def corpus_evidence_table(corpus_id: str, mreq: ModuleRequest, request: Request) -> Dict[str, Any]:
    result = await _stored_analysis(corpus_id, mreq, request)
    return _module_payload(_corpus_or_404(corpus_id), result, "evidence-table")


@app.post("/api/corpus/{corpus_id}/gaps")
async def corpus_gaps(corpus_id: str, mreq: ModuleRequest, request: Request) -> Dict[str, Any]:
    result = await _stored_analysis(corpus_id, mreq, request)
    return _module_payload(_corpus_or_404(corpus_id), result, "gaps")


@app.post("/api/corpus/{corpus_id}/map")
async def corpus_map(corpus_id: str, mreq: ModuleRequest) -> Dict[str, Any]:
    """The analysed map when an analysis exists; otherwise the map of the included studies alone (citation and
    wording links), which needs no model."""
    corpus = _corpus_or_404(corpus_id)
    if corpus.analysis is not None and not mreq.rerun:
        return _module_payload(corpus, corpus.analysis, "map")
    run = _run_from_corpus(corpus, mreq.config)
    graph = build_literature_graph(papers=run.papers, hf_resources=[], query=corpus.query, mode=corpus.mode,
                                   decomposition_plan=run.plan)
    return {"graph": graph.model_dump(), "corpus_id": corpus.corpus_id, "corpus_hash": corpus.content_hash,
            "module": "map", "analysis_model": "", "analysed_at": ""}


@dataclass
class CorpusRun:
    """A built corpus and what the analysis stage needs from the search that built it."""
    req: SearchRequest
    plan: QueryDecompositionPlan
    corpus: Corpus
    papers: List[PaperRecord]                 # the included studies, numbered
    selection_report: Optional[SelectionReport]
    source_stats: Dict[str, int]
    dedup_stats: Dict[str, int]
    sub_queries: List[str]
    fallback_entities: str


def _begin_request_scope(cfg: ApiKeysConfig) -> None:
    # Isolate per-search LLM health (circuit breaker), claim/audit state and source budgets from concurrent searches
    begin_llm_request_health()
    begin_synthesis_scope()
    begin_source_health()
    begin_search_budget()
    set_openalex_api_key(cfg.openalex_api_key)


async def _run_research_search(req: SearchRequest) -> ResearchResponse:
    """The whole search: build and store the corpus, then analyse its included studies."""
    return await _analyse(await _build_corpus(req))


async def _build_corpus(req: SearchRequest) -> CorpusRun:
    """Stages 1-3: plan the searches, run them, merge duplicates, select (and in PRISMA mode screen) the
    studies, and store the corpus. No study is read or summarised here."""
    cfg = req.config
    _begin_request_scope(cfg)
    # Stage A, B, C: Universal Question Understanding -> Research Strategy -> 5-API Compilation
    _report_stage(req, 1)
    if req.protocol is not None:
        # The user's review protocol drives the searches and selection; no model plans anything
        protocol = req.protocol.model_copy(deep=True)
        req = req.model_copy(update={
            "review_framework": protocol.review_type if req.mode == "deep_research" else "narrative",
            "organization": protocol.organization,
            "year_from": protocol.eligibility.year_from, "year_to": protocol.eligibility.year_to,
        })
        plan = plan_from_protocol(protocol, req.mode, domain_override=req.domain)
    else:
        plan = await generate_multi_api_decomposition_plan(req.query, req.mode, cfg, domain_override=req.domain)
        # Every corpus records a protocol: here the one the tool followed on its own (selection reads the
        # rule-based reading of the question; a model, if configured, may have widened the searches)
        protocol = default_protocol(plan, req.query, review_type=req.review_framework, organization=req.organization,
                                    year_from=req.year_from, year_to=req.year_to)
    fallback_entities_str = " ".join(plan.extracted_technical_entities[:3]) or req.query

    # How many studies, and how they are chosen, follow the review type and the protocol (services/study_limits.py)
    method = selection_method(req.mode, req.review_framework)
    max_total_papers = resolve_study_limit(req.mode, req.review_framework, protocol.study_limit)
    # Selection reads the rule-based parse of the question, so it does not depend on the model that planned the searches
    selection_understanding = plan.selection_understanding or plan.question_understanding
    per_q_limit = results_per_search(req.mode, "core")
    domain = plan.question_understanding.primary_domain if plan.question_understanding else ""

    sub_queries = [
        f"[{f.facet_name}] {f.queries.semantic_scholar}"
        for f in plan.search_facets
    ]

    # ROUND 1: the core search, the reviews search and one search per aspect of the question, on every source
    _report_stage(req, 2, f"{len(plan.search_facets)} searches (core, reviews and {max(0, len(plan.search_facets) - 2)} aspects)")
    wave_tasks = [
        _harvest_facet_queries(facet, results_per_search(req.mode, facet.kind, method != "narrative"),
                               req, fallback_entities_str, domain)
        for facet in plan.search_facets
    ]
    wave_results = await asyncio.gather(*wave_tasks)

    all_raw_papers: List[PaperRecord] = []
    source_stats: Dict[str, int] = {"PubMed": 0, "arXiv": 0, "bioRxiv": 0, "Semantic Scholar": 0, "Hugging Face": 0}

    for papers_batch, counts_batch in wave_results:
        all_raw_papers.extend(papers_batch)
        for k, v in counts_batch.items():
            source_stats[k] = source_stats.get(k, 0) + v

    # ROUND 2: Adaptive Expansion (triggered when Round 1 yields sparse candidates or in Deep Research mode)
    if len(all_raw_papers) < 6 and fallback_entities_str:
        broad_facet = SearchFacet(
            facet_code="F1_adaptive_broadening",
            facet_name="Adaptive Recall Expansion",
            facet_intent="Broaden concept conjunction when initial strict queries return sparse records.",
            expansion_reason="Round 1 sparse candidate recovery",
            queries=FacetApiQueries(
                pubmed=fallback_entities_str,
                semantic_scholar=fallback_entities_str,
                arxiv=" AND ".join(f"all:{t}" for t in plan.extracted_technical_entities[:2]) or "all:science",
                biorxiv_medrxiv=[fallback_entities_str],
                hugging_face=fallback_entities_str,
            ),
        )
        broad_facet.kind = "core"
        extra_papers, extra_counts = await _harvest_facet_queries(broad_facet, per_q_limit, req, fallback_entities_str, domain)
        all_raw_papers.extend(extra_papers)
        for k, v in extra_counts.items():
            source_stats[k] = source_stats.get(k, 0) + v
        if plan.search_plan:
            plan.search_plan.adaptive_expansion_triggered = True
            plan.search_plan.expansion_notes.append("Triggered Round 2 adaptive recall broadening due to sparse Round 1 results.")

    # Real citation counts (NIH iCite for PubMed records) before anything is ranked by them
    await enrich_citation_counts(all_raw_papers)

    # F6 Citation & Literature-Network Expansion in Deep Research mode
    if req.mode == "deep_research" and all_raw_papers:
        prelim = deduplicate_and_rank_papers(
            all_raw_papers,
            req.query,
            max_total=15,
            understanding=selection_understanding,
        )
        seminal = sorted(prelim, key=lambda x: x.citation_count, reverse=True)[:2]
        snowball_tasks = [
            expand_paper_neighbors(
                title=sp.title,
                doi=sp.doi,
                arxiv_id=sp.arxiv_id,
                pmid=sp.pmid,
                s2_id=sp.s2_id,
                api_key=cfg.s2_api_key,
            )
            for sp in seminal
        ]
        snowball_res = await asyncio.gather(*snowball_tasks, return_exceptions=True)
        for sb in snowball_res:
            if isinstance(sb, list):
                for nb in sb:
                    nb.retrieved_by = ["Citation links of highly cited studies"]
                all_raw_papers.extend(sb)
                source_stats["Semantic Scholar"] = source_stats.get("Semantic Scholar", 0) + len(sb)

    # What the user brought (local installs only): exported records and PDFs join the pool before duplicates are
    # merged, so they pass the same eligibility and scoring as searched records
    imports, pdf_records, pdf_summaries, forced_keys = _user_material(req, all_raw_papers, source_stats)

    # ROUND 3: Verification, Deduplication & Multi-Criteria Study Selection
    _report_stage(req, 3, f"{len(all_raw_papers)} candidate records harvested")
    # Scored against the question plus the rule-based parse's key terms (not the model's plan)
    ranking_query = f"{req.query} {' '.join(plan.selection_entities or plan.extracted_technical_entities)}"
    dedup_stats: Dict[str, int] = {}
    await enrich_citation_counts(all_raw_papers)  # records added by citation links
    # Systematic-style and scoping reviews keep every word-eligible record, so the model can screen them before
    # the limit applies
    ranked_papers, selection_report, corpus = select_for_corpus(
        all_raw_papers,
        query=req.query,
        ranking_query=ranking_query,
        study_limit=max_total_papers,
        understanding=selection_understanding,
        method=method,
        stats=dedup_stats,
        protocol=protocol,
    )
    if method != "narrative" and selection_report and ranked_papers:
        _report_stage(req, 3, f"Screening {len(ranked_papers)} eligible records on title and abstract")
        ranked_papers = await screen_and_limit(ranked_papers, selection_report, req.query, cfg, max_total_papers,
                                               criteria=screening_criteria(protocol, method))
    # The user's material: which merged study each PDF became, its full text, and studies included by the user
    corpus.imports = imports
    _record_user_pdfs(corpus, pdf_records, pdf_summaries, forced_keys)
    ranked_papers = apply_forced_includes(ranked_papers, corpus, selection_report)
    attach_user_full_texts(ranked_papers, corpus)
    # The corpus: the records, the inputs and the decisions, so the selection can be audited and replayed
    corpus.mode, corpus.review_framework, corpus.organization = req.mode, req.review_framework, req.organization
    corpus.year_from, corpus.year_to = req.year_from, req.year_to
    corpus.plan = plan.model_copy(deep=True)
    corpus.source_stats = dict(source_stats)
    corpus.selection_stats = dict(dedup_stats)
    corpus.search_warnings = source_failure_warnings()
    finish_corpus(corpus, ranked_papers, selection_report)
    corpus_store.save(corpus)
    return CorpusRun(req=req, plan=plan, corpus=corpus, papers=ranked_papers, selection_report=selection_report,
                     source_stats=source_stats, dedup_stats=dedup_stats, sub_queries=sub_queries,
                     fallback_entities=fallback_entities_str)


async def _analyse(run: CorpusRun) -> ResearchResponse:
    """Stages 3-6 on a corpus's included studies: full text, extraction, synthesis, evidence table, gaps and map.
    The result is stored with the corpus, so the per-module endpoints can return their part of it."""
    req, plan, corpus = run.req, run.plan, run.corpus
    cfg = req.config
    ranked_papers, selection_report = run.papers, run.selection_report
    source_stats, dedup_stats = dict(run.source_stats), run.dedup_stats
    sub_queries, fallback_entities_str = run.sub_queries, run.fallback_entities

    # Which sub-questions of the search plan each included study bears on (labelling only, after selection)
    label_subquestions(ranked_papers, plan.question_understanding)

    # Open-access full text for the included studies, fetched while Hugging Face Hub is scanned below
    fulltext_task = (
        asyncio.create_task(attach_open_access_full_texts(ranked_papers)) if cfg.enable_fulltext and ranked_papers
        else None
    )

    # Scrape Hugging Face Hub using decomposed HF facet queries + arXiv IDs
    arxiv_ids = [p.arxiv_id for p in ranked_papers if p.arxiv_id]
    hf_facet_queries = [f.queries.hugging_face for f in plan.search_facets if f.queries.hugging_face]
    hf_resources = []
    if cfg.enable_hf:
        hf_resources = await search_huggingface_hub(
            query=fallback_entities_str,
            arxiv_ids=arxiv_ids,
            hf_api_key=cfg.hf_api_key,
            max_models=10 if req.mode == "deep_research" else 8,
            max_datasets=6 if req.mode == "deep_research" else 4,
            hf_facet_queries=hf_facet_queries,
        )
        source_stats["Hugging Face"] = len(hf_resources)

    if fulltext_task is not None:
        full_text_count = await fulltext_task
        _report_stage(req, 3, f"Open-access full text found for {full_text_count} of {len(ranked_papers)} studies")

    # Build Literature Graph (assigns thematic cluster_id, cluster_label, and typed edges)
    graph = build_literature_graph(ranked_papers, hf_resources)

    # Build deterministic Bibliography [1]..[N]
    bibliography = generate_bibliography(ranked_papers)

    # Generate Claim-Level Synthesis, Evidence Agreement Breakdown & Domain-Adaptive Extractions
    _report_stage(req, 4, f"{len(ranked_papers)} papers selected")
    downstream: Dict[str, Any] = {}

    async def _build_evidence_and_gaps(claim_evidence_pkg) -> None:
        # Runs concurrently with the claim (D-G) and narrative (J) LLM calls; needs only Task C results and claims
        _report_stage(req, 5)
        downstream["evidence_matrix"] = build_structured_evidence_matrix(
            query=req.query,
            papers=ranked_papers,
            decomposition_plan=plan,
            synthesis_package=claim_evidence_pkg,
        )
        # Gaps are judged from studies that address the question (relevance is set by Task C before this runs)
        on_topic = [p for p in ranked_papers if p.relevance != "off_topic"]
        downstream["gap_matrix"] = await build_knowledge_gap_matrix(
            query=req.query,
            papers=on_topic if len(on_topic) >= 3 else ranked_papers,
            hf_resources=hf_resources,
            config=cfg,
            decomposition_plan=plan,
            synthesis_package=claim_evidence_pkg,
            evidence_matrix=downstream["evidence_matrix"],
            all_candidate_pool=corpus.records,
        )

    # Systematic-style review: risk of bias from the full text, alongside the synthesis
    rob_task = (asyncio.create_task(assess_risk_of_bias(ranked_papers, cfg))
                if req.mode == "deep_research" and req.review_framework == "systematic_prisma" and ranked_papers else None)

    exec_answer, sections, consensus, warnings = await generate_synthesis_and_extractions(
        query=req.query,
        mode=req.mode,
        papers=ranked_papers,
        hf_resources=hf_resources,
        bibliography=bibliography,
        config=cfg,
        decomposition_plan=plan,
        on_claim_evidence_ready=_build_evidence_and_gaps,
        review_framework=req.review_framework,
        organization=req.organization,
        previous_reviews=selection_report.excluded_reviews if selection_report else (),
    )

    if rob_task is not None:
        judgements = await rob_task
        for p in ranked_papers:
            if p.id in judgements:
                p.elicit.risk_of_bias = judgements[p.id]

    # Sync updated stances onto graph nodes
    stance_by_id = {p.id: p.stance for p in ranked_papers}
    for node in graph.nodes:
        if node.id in stance_by_id:
            node.stance = stance_by_id[node.id]

    synthesis_pkg = get_last_synthesis_package()
    if "gap_matrix" not in downstream:
        # Synthesis returned early (e.g. no eligible papers) without starting the concurrent hand-off
        _report_stage(req, 5)
        await _build_evidence_and_gaps(synthesis_pkg)
    evidence_matrix = downstream["evidence_matrix"]
    gap_matrix = downstream["gap_matrix"]

    # Rebuild Section 4 Multilayer Scientific Literature Network Map with full Layer C (Claim, Dataset & Gap) links
    _report_stage(req, 6)
    print(f"[search-llm] {req.mode} {llm_request_stats()}", flush=True)
    graph =build_literature_graph(
        papers=ranked_papers,
        hf_resources=hf_resources,
        query=req.query,
        mode=req.mode,
        decomposition_plan=plan,
        synthesis_package=synthesis_pkg,
        evidence_matrix=evidence_matrix,
        gap_matrix=gap_matrix,
    )

    # Build Phase 7 Shared Scientific Evidence Base
    evidence_base = build_shared_evidence_base(
        papers=ranked_papers,
        gap_matrix=gap_matrix,
        graph=graph,
        decomposition_plan=plan,
    )

    # Run Phase 9 & Section 6 Deterministic Verification & Quality Gates
    quality_report = run_verification_and_quality_gates(
        plan=plan,
        papers=ranked_papers,
        executive_answer=exec_answer,
        sections=sections,
        gap_matrix=gap_matrix,
        graph=graph,
        evidence_base=evidence_base,
        task_audit_trail=evidence_base.task_audit_trail,
        model_capability=resolve_model_capabilities(cfg),
    )

    # Sources that still failed after retries are named in the results instead of silently returning fewer papers
    warnings = list(dict.fromkeys(corpus.search_warnings + source_failure_warnings())) + warnings
    if req.mode == "deep_research":
        tier, reason = classify_model_scale(cfg.llm_provider, cfg.llm_model)
        if tier == "small":
            warnings.insert(0, (
                f"Model guidance: this Deep Research review was written with {cfg.llm_model} ({reason}). Deep Research "
                "works best with a large model (about 70B+ parameters or a frontier model); small models write shorter, "
                "less detailed reviews, and sections they cannot complete fall back to rule-based text."
            ))

    enabled_sources = [
        name for name, on in (("PubMed", cfg.enable_pubmed), ("arXiv", cfg.enable_arxiv),
                              ("bioRxiv", cfg.enable_biorxiv), ("Semantic Scholar", cfg.enable_s2)) if on
    ] + [k for k in source_stats if k.endswith("(imported by the user)") or k == USER_PDF_SOURCE]
    review_protocol = build_review_protocol_report(
        framework_id=req.review_framework,
        organization_id=req.organization,
        primary_domain=(plan.active_profile_summary or {}).get("primary_domain", "biomedical_clinical"),
        plan=plan,
        included_papers=ranked_papers,
        identified_by_source=source_stats,
        identified_total=corpus.records_found,
        unique_records=dedup_stats.get("unique_records", len(ranked_papers)),
        enabled_sources=enabled_sources,
        year_from=req.year_from,
        year_to=req.year_to,
        constraint_excluded=dedup_stats.get("constraint_excluded", 0),
        selection_report=selection_report,
    )
    for imp in corpus.imports:
        if imp.search_string:
            review_protocol.search_strategies[f"{imp.source} (run by the user, {imp.imported_at[:10]})"] = [imp.search_string]
    if req.mode == "deep_research" and req.review_framework == "systematic_prisma" and selection_report and ranked_papers:
        # PRISMA 2020 reporting order: structured abstract, computed Methods and descriptive Results around the
        # model-written Introduction, syntheses and Discussion
        exec_answer, sections = compose_prisma_report(
            query=req.query, executive_answer=exec_answer, sections=sections, papers=ranked_papers,
            selection_report=selection_report, protocol=review_protocol, model=cfg.llm_model,
            organization_id=req.organization,
        )

    result = ResearchResponse(
        query=req.query,
        mode=req.mode,
        sub_queries=sub_queries,
        decomposition_plan=plan,
        evidence_base=evidence_base,
        quality_report=quality_report,
        synthesis_evidence=synthesis_pkg,
        evidence_matrix=evidence_matrix,
        source_stats=source_stats,
        executive_answer=exec_answer,
        sections=sections,
        consensus=consensus,
        papers=ranked_papers,
        bibliography=bibliography,
        hf_resources=hf_resources,
        graph=graph,
        gap_matrix=gap_matrix,
        warnings=warnings,
        review_protocol=review_protocol,
        selection_report=selection_report,
        corpus_id=corpus.corpus_id,
        corpus_hash=corpus.content_hash,
        protocol=corpus.protocol,
    )
    # Kept with the corpus so the module endpoints can return their part without running the models again
    corpus.analysis = result.model_copy(deep=True)
    corpus.analysis_model = f"{cfg.llm_provider}:{cfg.llm_model}" if (cfg.llm_api_key or cfg.llm_provider in ("ollama", "custom")) else "none (rule-based)"
    corpus.analysed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    corpus_store.save(corpus)
    return result



@app.post("/api/expand-node")
async def expand_graph_node(req: ExpandNodeRequest) -> Dict[str, Any]:
    if demo_mode.demo_enabled():
        req = req.model_copy(update={"config": demo_mode.fill_source_keys(req.config)})
    begin_search_budget()
    set_openalex_api_key(req.config.openalex_api_key)
    neighbors = await expand_paper_neighbors(
        title=req.title,
        doi=req.doi,
        arxiv_id=req.arxiv_id,
        pmid=req.pmid,
        s2_id=req.s2_id,
        api_key=req.config.s2_api_key,
    )
    return {
        "parent_id": req.paper_id,
        "neighbors": [p.model_dump() for p in neighbors[:8]],
    }


@app.post("/api/export")
async def export_synthesis_report(req: ExportRequest) -> Response:
    fmt = req.format.lower()
    if fmt == "md":
        md_text = generate_markdown_report(req)
        return Response(
            content=md_text.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="literature_synthesis_report.md"'},
        )
    elif fmt == "txt":
        txt_content = generate_txt_report(req)
        return Response(
            content=txt_content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="literature_synthesis_report.txt"'},
        )
    elif fmt == "pdf":
        pdf_bytes = generate_pdf_report(req)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="literature_synthesis_report.pdf"'},
        )
    elif fmt == "csv":
        matrix = req.evidence_matrix or build_structured_evidence_matrix(
            query=req.query,
            papers=req.papers,
            synthesis_package=req.synthesis_evidence,
        )
        csv_text = export_evidence_matrix_to_csv(matrix)
        return Response(
            content=csv_text.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="structured_evidence_matrix.csv"'},
        )
    elif fmt == "json":
        matrix = req.evidence_matrix or build_structured_evidence_matrix(
            query=req.query,
            papers=req.papers,
            synthesis_package=req.synthesis_evidence,
        )
        json_text = export_evidence_matrix_to_json(matrix)
        return Response(
            content=json_text.encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="structured_evidence_matrix.json"'},
        )
    elif fmt in ("graphml", "graph_nodes_csv", "graph_edges_csv", "graph_json"):
        graph = req.graph or build_literature_graph(
            papers=req.papers,
            hf_resources=req.hf_resources,
            query=req.query,
            mode=req.mode,
            synthesis_package=req.synthesis_evidence,
            evidence_matrix=req.evidence_matrix,
            gap_matrix=req.gap_matrix,
        )
        if fmt == "graphml":
            xml_text = export_network_to_graphml(graph)
            return Response(
                content=xml_text.encode("utf-8"),
                media_type="application/xml; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="literature_network.graphml"'},
            )
        elif fmt == "graph_nodes_csv":
            csv_text = export_network_nodes_csv(graph)
            return Response(
                content=csv_text.encode("utf-8"),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="literature_network_nodes.csv"'},
            )
        elif fmt == "graph_edges_csv":
            csv_text = export_network_edges_csv(graph)
            return Response(
                content=csv_text.encode("utf-8"),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="literature_network_edges.csv"'},
            )
        else:
            json_text = export_network_json(graph)
            return Response(
                content=json_text.encode("utf-8"),
                media_type="application/json; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="literature_network.json"'},
            )
    return Response(content=b"Unsupported format", status_code=400)


# Mount frontend directory for seamless single-port serving as well as Vite dev mode
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if (FRONTEND_DIR / "dist").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR / "dist"), html=True), name="frontend_dist")
elif FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    app.mount("/src", StaticFiles(directory=str(FRONTEND_DIR / "src")), name="frontend_src")
