import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
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
    classify_model_scale,
)
from frameworks.reporting import build_review_protocol_report
from services import demo_mode
from services.source_budget import begin_search_budget
from services.fulltext import attach_open_access_full_texts
from services.source_health import begin_source_health, set_openalex_api_key, source_failure_warnings
from services.query_utils import generate_multi_api_decomposition_plan
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
    generate_pdf_report,
)

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
) -> Tuple[List[PaperRecord], Dict[str, int]]:
    cfg = req.config
    tasks = []
    task_names = []
    q_map = facet.queries

    if cfg.enable_pubmed:
        tasks.append(
            search_pubmed(
                q_map.pubmed or fallback_entities,
                max_per_source,
                cfg.pubmed_api_key,
                req.year_from,
                req.year_to,
                fallback_query=fallback_entities,
            )
        )
        task_names.append("PubMed")
    if cfg.enable_arxiv:
        tasks.append(
            search_arxiv(
                q_map.arxiv or fallback_entities,
                max_per_source,
                req.year_from,
                req.year_to,
                fallback_query=fallback_entities,
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

    results = await asyncio.gather(*tasks, return_exceptions=True)
    papers: List[PaperRecord] = []
    counts: Dict[str, int] = {name: 0 for name in task_names}

    for name, res in zip(task_names, results):
        if isinstance(res, list):
            for p in res:
                p.sub_query = facet.facet_name
            papers.extend(res)
            counts[name] = len(res)
    return papers, counts


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


def _report_stage(req: SearchRequest, stage: int, detail: str = "") -> None:
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


@app.post("/api/search", response_model=ResearchResponse)
async def execute_research_search(req: SearchRequest, request: Request) -> ResearchResponse:
    if not demo_mode.demo_enabled():
        try:
            return await _run_research_search(req)
        finally:
            if req.progress_id:
                _SEARCH_PROGRESS.pop(req.progress_id, None)

    # Public demo: shared NVIDIA free models with per-address daily limits; visitors' own keys are not limited
    ip = demo_mode.client_ip(request.headers.get("x-forwarded-for"), request.client.host if request.client else None)
    try:
        cfg, uses_server_key = demo_mode.apply_demo_config(req.config)
        bucket = demo_mode.bucket_for(req.mode, uses_server_key)
        demo_mode.take_quota(ip, bucket)
    except demo_mode.DemoLimitError as exc:
        raise HTTPException(exc.status, str(exc))
    slots = demo_mode.search_slots()
    if slots.locked():
        demo_mode.USAGE.give_back(ip, bucket)
        raise HTTPException(503, "The demo is busy with other searches. Please try again in a minute.")
    completed = False
    try:
        async with slots:
            run_req = req.model_copy(update={"config": cfg})
            result = await _run_research_search(run_req)
            completed = True
            return result
    finally:
        if not completed:
            demo_mode.USAGE.give_back(ip, bucket)  # a crashed run does not use up the visitor's allowance
        if req.progress_id:
            _SEARCH_PROGRESS.pop(req.progress_id, None)


async def _run_research_search(req: SearchRequest) -> ResearchResponse:
    cfg = req.config
    # Isolate per-search LLM health (circuit breaker) and claim/audit state from concurrent searches
    begin_llm_request_health()
    begin_synthesis_scope()
    begin_source_health()
    begin_search_budget()
    set_openalex_api_key(cfg.openalex_api_key)
    # Stage A, B, C: Universal Question Understanding -> Research Strategy -> 5-API Compilation
    _report_stage(req, 1)
    plan = await generate_multi_api_decomposition_plan(req.query, req.mode, cfg, domain_override=req.domain)
    fallback_entities_str = " ".join(plan.extracted_technical_entities[:3]) or req.query

    if req.mode == "deep_research":
        per_q_limit = max(14, req.max_per_source)
        max_total_papers = 60
    else:
        per_q_limit = max(6, req.max_per_source)
        max_total_papers = 20

    sub_queries = [
        f"[{f.facet_name}] {f.queries.semantic_scholar}"
        for f in plan.search_facets
    ]

    # ROUND 1: Initial Discovery across planned facets
    _report_stage(req, 2, f"{len(plan.search_facets)} search facets")
    wave_tasks = [
        _harvest_facet_queries(facet, per_q_limit, req, fallback_entities_str)
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
        extra_papers, extra_counts = await _harvest_facet_queries(broad_facet, per_q_limit, req, fallback_entities_str)
        all_raw_papers.extend(extra_papers)
        for k, v in extra_counts.items():
            source_stats[k] = source_stats.get(k, 0) + v
        if plan.search_plan:
            plan.search_plan.adaptive_expansion_triggered = True
            plan.search_plan.expansion_notes.append("Triggered Round 2 adaptive recall broadening due to sparse Round 1 results.")

    # F6 Citation & Literature-Network Expansion in Deep Research mode
    if req.mode == "deep_research" and all_raw_papers:
        prelim = deduplicate_and_rank_papers(
            all_raw_papers,
            req.query,
            max_total=15,
            understanding=plan.question_understanding,
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
                all_raw_papers.extend(sb)
                source_stats["Semantic Scholar"] = source_stats.get("Semantic Scholar", 0) + len(sb)

    # ROUND 3: Verification, Deduplication & Multi-Criteria Study Selection
    _report_stage(req, 3, f"{len(all_raw_papers)} candidate records harvested")
    ranking_query = f"{req.query} {' '.join(plan.extracted_technical_entities)}"
    dedup_stats: Dict[str, int] = {}
    ranked_papers = deduplicate_and_rank_papers(
        all_raw_papers,
        ranking_query,
        max_total=max_total_papers,
        understanding=plan.question_understanding,
        stats=dedup_stats,
    )

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
            all_candidate_pool=all_raw_papers,
        )

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
    )

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
    warnings = source_failure_warnings() + warnings
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
    ]
    review_protocol = build_review_protocol_report(
        framework_id=req.review_framework,
        organization_id=req.organization,
        primary_domain=(plan.active_profile_summary or {}).get("primary_domain", "biomedical_clinical"),
        plan=plan,
        included_papers=ranked_papers,
        identified_by_source=source_stats,
        identified_total=len(all_raw_papers),
        unique_records=dedup_stats.get("unique_records", len(ranked_papers)),
        enabled_sources=enabled_sources,
        year_from=req.year_from,
        year_to=req.year_to,
        constraint_excluded=dedup_stats.get("constraint_excluded", 0),
    )

    return ResearchResponse(
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
    )


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
