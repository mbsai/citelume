"""
Evidence gap map: how much research exists for each combination of an intervention (or exposure) and an outcome,
counted by the databases themselves, by study design.

This is the count-based gap view planned as Session 11 (docs/PUBLICATION_PLAN.md, docs/GAPS.md). It follows the
evidence gap map method (Snilstveit et al. 2016; Campbell and 3ie maps): a grid fixed before searching, rows by
columns, each cell filled from a documented search. Unlike the gap view of a review, it does not depend on which
studies were selected: every cell is one count query per database, run with no model, and reported with its exact
query string and the date it ran.

For each cell (row x column, within the population if one is given):
  PubMed       all records, randomised trials, systematic reviews and meta-analyses, observational studies
               (publication types and MeSH), with MeSH headings for each term where PubMed maps one
  OpenAlex     all records, by words in the title and abstract (it has no study-design index; it also covers
               preprints and journals outside MEDLINE)
Europe PMC is not used: its MeSH field gave counts that changed with the order of the terms, and field-scoped
queries took 15 to 25 seconds each or timed out (tested 24 Nov 2026).
Row and column totals (PubMed, all records) show how much exists for each term at all.

Counts are records that match a query, not screened studies: a broad term over-counts and a missing synonym
under-counts. The page says so, and every query can be opened in the database to check it.
"""
from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from models import ApiKeysConfig
from services.mesh_terms import ESEARCH, lookup as mesh_lookup, mesh_alternative

GAP_MAP_VERSION = "citelume-gap-map/1.0"
OPENALEX_WORKS = "https://api.openalex.org/works"
MAX_AXIS = 8          # most rows or columns on a local install
DEMO_MAX_AXIS = 5     # on the public demo (one shared set of database keys)
SPARSE_BELOW = 5      # a cell with fewer records than this in every database is "few records"
TRIALS_FOR_REVIEW = 3  # trials without any systematic review: evidence that has not been synthesised

# (key, label, PubMed filter). The filters use PubMed publication types and MeSH; Europe PMC gets them translated.
DESIGNS: List[Tuple[str, str, str]] = [
    ("all", "All records", ""),
    ("rct", "Randomised trials", '"randomized controlled trial"[Publication Type]'),
    ("sr", "Systematic reviews and meta-analyses",
     '("systematic review"[Publication Type] OR "meta-analysis"[Publication Type])'),
    ("obs", "Observational studies",
     '("observational study"[Publication Type] OR "cohort studies"[MeSH Terms] OR "case-control studies"[MeSH Terms]'
     ' OR "cross-sectional studies"[MeSH Terms])'),
]

FLAG_TEXT = {
    "none": "No records in any database",
    "none_pubmed": "No records in PubMed (OpenAlex not counted)",
    "sparse": f"Fewer than {SPARSE_BELOW} records",
    "no_trials": "No randomised trials",
    "unsynthesised": "Trials but no systematic review",
}
FLAG_ORDER = ["none", "none_pubmed", "sparse", "no_trials", "unsynthesised"]


# ---------------------------------------------------------------------------------------------------------------
# Request and result
# ---------------------------------------------------------------------------------------------------------------

class MapTerm(BaseModel):
    name: str
    synonyms: List[str] = Field(default_factory=list)


class GapMapRequest(BaseModel):
    topic: str = ""
    population: Optional[MapTerm] = None
    rows: List[MapTerm]
    cols: List[MapTerm]
    row_title: str = "Interventions or exposures"
    col_title: str = "Outcomes"
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    use_mesh: bool = True
    config: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    progress_id: str = ""


class GapMapCell(BaseModel):
    row: int
    col: int
    # design key -> database ("pubmed") -> count (None if that query failed)
    counts: Dict[str, Dict[str, Optional[int]]] = Field(default_factory=dict)
    openalex_all: Optional[int] = None
    flags: List[str] = Field(default_factory=list)
    # The query for "all records"; a design's query is "(<this>) AND <design filter>" (see GapMapResult.designs)
    pubmed_query: str = ""
    openalex_query: str = ""


class GapMapGap(BaseModel):
    row: int
    col: int
    flag: str
    text: str


class GapMapResult(BaseModel):
    version: str = GAP_MAP_VERSION
    topic: str = ""
    population: Optional[MapTerm] = None
    rows: List[MapTerm] = Field(default_factory=list)
    cols: List[MapTerm] = Field(default_factory=list)
    row_title: str = ""
    col_title: str = ""
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    designs: List[Dict[str, str]] = Field(default_factory=list)
    cells: List[GapMapCell] = Field(default_factory=list)
    row_totals: List[Optional[int]] = Field(default_factory=list)
    col_totals: List[Optional[int]] = Field(default_factory=list)
    term_queries: Dict[str, str] = Field(default_factory=dict)  # "row:0" / "col:2" / "population" -> PubMed clause
    gaps: List[GapMapGap] = Field(default_factory=list)
    searched_at: str = ""
    queries_run: int = 0
    queries_failed: int = 0
    duration_s: float = 0.0
    notes: List[str] = Field(default_factory=list)  # e.g. a database that could not be counted, and why
    method: List[str] = Field(default_factory=list)


class GapMapSuggestion(BaseModel):
    population: Optional[MapTerm] = None
    rows: List[MapTerm] = Field(default_factory=list)
    cols: List[MapTerm] = Field(default_factory=list)
    source: str = "rules"  # "rules" (from the question's own words) or the model's name
    note: str = ""


# ---------------------------------------------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------------------------------------------

def _clean(term: str) -> str:
    # Quotes, brackets and commas would break the query syntax of one database or another
    return re.sub(r"\s+", " ", re.sub(r'["\[\](),;]', " ", term or "")).strip()


def _words(term: MapTerm) -> List[str]:
    return [w for w in dict.fromkeys(_clean(t) for t in [term.name, *term.synonyms]) if w]


async def pubmed_clause(term: MapTerm, client: Optional[httpx.AsyncClient], api_key: str = "",
                        use_mesh: bool = True) -> str:
    """A term and its synonyms in title or abstract, OR its MeSH heading(s) where PubMed maps one."""
    parts = [f'"{w}"[Title/Abstract]' for w in _words(term)]
    if use_mesh and client is not None:
        try:
            groups = await mesh_lookup(_clean(term.name), client, api_key)
        except Exception:
            groups = []
        if groups:
            parts.append(mesh_alternative(groups))
    return "(" + " OR ".join(dict.fromkeys(parts)) + ")" if parts else ""


def openalex_clause(term: MapTerm) -> str:
    return "(" + " OR ".join(f'"{w}"' for w in _words(term)) + ")"


def _years_pubmed(req: GapMapRequest) -> str:
    if not (req.year_from or req.year_to):
        return ""
    return f"{req.year_from or 1800}:{req.year_to or datetime.now(timezone.utc).year}[dp]"


def combine(*clauses: str) -> str:
    return " AND ".join(c for c in clauses if c)


def with_design(base: str, design_filter: str) -> str:
    return f"({base}) AND {design_filter}" if design_filter else base


# ---------------------------------------------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------------------------------------------

class _Pacer:
    """Keeps requests to one service at most one per `interval` seconds (NCBI allows 3 a second without a key)."""

    def __init__(self, interval: float) -> None:
        self.interval, self._next, self._lock = interval, 0.0, asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


RETRIES = 3


async def count_pubmed(client: httpx.AsyncClient, query: str, api_key: str, pacer: _Pacer) -> Optional[int]:
    """Every attempt, retries included, waits its turn: NCBI refuses a fourth request within a second."""
    params = {"db": "pubmed", "term": query, "retmax": "0", "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    for attempt in range(RETRIES):
        await pacer.wait()
        try:
            resp = await client.get(ESEARCH, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                await asyncio.sleep(1.0 + attempt)
                continue
            resp.raise_for_status()
            return int(resp.json()["esearchresult"]["count"])
        except (httpx.TransportError, ValueError, KeyError):
            await asyncio.sleep(1.0 + attempt)
        except httpx.HTTPStatusError:
            return None
    return None


class _OpenAlexState:
    """Once OpenAlex says the shared keyless budget is spent, the rest of the map does not ask again."""

    def __init__(self) -> None:
        self.out_of_budget = False


async def count_openalex(client: httpx.AsyncClient, expr: str, years: str, api_key: str,
                         slots: asyncio.Semaphore, state: Optional[_OpenAlexState] = None) -> Optional[int]:
    flt = f"title_and_abstract.search:{expr}" + (f",publication_year:{years}" if years else "")
    params = {"filter": flt, "per_page": "1", "select": "id"}
    if api_key:
        params["api_key"] = api_key
    async with slots:
        for attempt in range(RETRIES):
            if state and state.out_of_budget:
                return None
            try:
                resp = await client.get(OPENALEX_WORKS, params=params)
                if resp.status_code == 429:
                    if "budget" in resp.text.lower() and state:
                        state.out_of_budget = True
                        return None
                    await asyncio.sleep(1.0 + attempt)
                    continue
                if resp.status_code >= 500:
                    await asyncio.sleep(1.0 + attempt)
                    continue
                resp.raise_for_status()
                return int(resp.json()["meta"]["count"])
            except (httpx.TransportError, ValueError, KeyError):
                await asyncio.sleep(1.0 + attempt)
            except httpx.HTTPStatusError:
                return None
        return None


def cell_flags(cell: GapMapCell) -> List[str]:
    """Fixed rules, no model. Designs come from PubMed; OpenAlex adds an independent total."""
    def best(design: str) -> Optional[int]:
        vals = [v for v in (cell.counts.get(design) or {}).values() if v is not None]
        return max(vals) if vals else None

    total = best("all")
    every = [v for v in [total, cell.openalex_all] if v is not None]
    if not every:
        return []  # every query failed: say nothing rather than call it a gap
    if max(every) == 0:
        # "No records in any database" only when every database answered
        return ["none"] if cell.openalex_all is not None and total is not None else ["none_pubmed"]
    flags: List[str] = []
    if max(every) < SPARSE_BELOW:
        flags.append("sparse")
    rct, sr = best("rct"), best("sr")
    if rct == 0:
        flags.append("no_trials")
    elif rct is not None and rct >= TRIALS_FOR_REVIEW and sr == 0:
        flags.append("unsynthesised")
    return flags


async def build_gap_map(req: GapMapRequest, max_axis: int = MAX_AXIS,
                        progress: Optional[Callable[[int, int], None]] = None) -> GapMapResult:
    if not req.rows or not req.cols:
        raise ValueError("The map needs at least one row and one column.")
    if len(req.rows) > max_axis or len(req.cols) > max_axis:
        raise ValueError(f"At most {max_axis} rows and {max_axis} columns here.")
    started = time.monotonic()
    pubmed_key = (req.config.pubmed_api_key or "").strip()
    openalex_key = (req.config.openalex_api_key or "").strip()
    pacer = _Pacer(0.12 if pubmed_key else 0.4)
    oa_slots, oa_state = asyncio.Semaphore(4), _OpenAlexState()
    years_pm = _years_pubmed(req)
    years_oa = f"{req.year_from or 1800}-{req.year_to or datetime.now(timezone.utc).year}" \
        if (req.year_from or req.year_to) else ""

    # A plain client: retries are paced here, so they do not arrive in bursts over the databases' rate limits
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        async def clause(term: MapTerm) -> str:
            if req.use_mesh:
                await pacer.wait()
            return await pubmed_clause(term, client if req.use_mesh else None, pubmed_key, req.use_mesh)

        pop_clause = await clause(req.population) if req.population and req.population.name.strip() else ""
        row_clauses = [await clause(t) for t in req.rows]
        col_clauses = [await clause(t) for t in req.cols]
        pop_oa = openalex_clause(req.population) if pop_clause else ""

        cells = [GapMapCell(row=i, col=j) for i in range(len(req.rows)) for j in range(len(req.cols))]
        jobs: List[Awaitable[None]] = []
        total = len(cells) * (len(DESIGNS) + 1) + len(req.rows) + len(req.cols)
        done = 0
        failed = 0

        def tick(value: Optional[int]) -> None:
            nonlocal done, failed
            done += 1
            if value is None:
                failed += 1
            if progress:
                progress(done, total)

        async def run_pubmed(target: Dict[str, Optional[int]], key: str, query: str) -> None:
            target[key] = await count_pubmed(client, query, pubmed_key, pacer)
            tick(target[key])

        async def run_openalex(cell: GapMapCell, expr: str) -> None:
            cell.openalex_all = await count_openalex(client, expr, years_oa, openalex_key, oa_slots, oa_state)
            tick(cell.openalex_all)

        for cell in cells:
            base = combine(row_clauses[cell.row], col_clauses[cell.col], pop_clause, years_pm)
            cell.pubmed_query = base
            cell.openalex_query = combine(openalex_clause(req.rows[cell.row]), openalex_clause(req.cols[cell.col]),
                                          pop_oa)
            for key, _, flt in DESIGNS:
                cell.counts[key] = {"pubmed": None}
                jobs.append(run_pubmed(cell.counts[key], "pubmed", with_design(base, flt)))
            jobs.append(run_openalex(cell, cell.openalex_query))

        row_totals: Dict[str, Optional[int]] = {}
        col_totals: Dict[str, Optional[int]] = {}
        for i, rc in enumerate(row_clauses):
            jobs.append(run_pubmed(row_totals, str(i), combine(rc, pop_clause, years_pm)))
        for j, cc in enumerate(col_clauses):
            jobs.append(run_pubmed(col_totals, str(j), combine(cc, pop_clause, years_pm)))
        await asyncio.gather(*jobs)

    for cell in cells:
        cell.flags = cell_flags(cell)
    notes: List[str] = []
    if oa_state.out_of_budget:
        notes.append("OpenAlex could not be counted: without an OpenAlex API key, requests share a small free daily "
                     "budget, which was spent. Add a free key (openalex.org) in API Keys & Sources Setup." if not
                     openalex_key else "OpenAlex could not be counted: the key's daily budget was spent.")
    pubmed_failed = sum(1 for c in cells for d in c.counts.values() if d.get("pubmed") is None)
    if pubmed_failed:
        notes.append(f"{pubmed_failed} PubMed counts failed after retries (shown as –); a PubMed API key raises the "
                     "rate limit from 3 to 10 requests a second.")
    gaps = sorted(
        (GapMapGap(row=c.row, col=c.col, flag=f, text=FLAG_TEXT[f]) for c in cells for f in c.flags),
        key=lambda g: (FLAG_ORDER.index(g.flag), g.row, g.col))
    terms = {f"row:{i}": q for i, q in enumerate(row_clauses)}
    terms.update({f"col:{j}": q for j, q in enumerate(col_clauses)})
    if pop_clause:
        terms["population"] = pop_clause
    return GapMapResult(
        topic=req.topic, population=req.population if pop_clause else None, rows=req.rows, cols=req.cols,
        row_title=req.row_title, col_title=req.col_title, year_from=req.year_from, year_to=req.year_to,
        designs=[{"key": k, "label": label, "pubmed_filter": f} for k, label, f in DESIGNS],
        cells=cells,
        row_totals=[row_totals.get(str(i)) for i in range(len(req.rows))],
        col_totals=[col_totals.get(str(j)) for j in range(len(req.cols))],
        term_queries=terms, gaps=gaps,
        searched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        queries_run=total, queries_failed=failed, duration_s=round(time.monotonic() - started, 1), notes=notes,
        method=method_notes(),
    )


def method_notes() -> List[str]:
    return [
        "Each cell is a count of records matching a query, run in each database on the date shown; no model is "
        "used and no record is screened. A broad term over-counts and a missing synonym under-counts: open a "
        "cell's query in the database to check what it finds.",
        "PubMed: each term and its synonyms in the title or abstract, or its MeSH heading where PubMed maps one; "
        "study designs by publication type and MeSH. OpenAlex: the words in the title or abstract, all designs "
        "together; it also covers preprints and journals outside MEDLINE.",
        f"Flags, by fixed rules: no records in any database; fewer than {SPARSE_BELOW} records; no randomised "
        f"trials; at least {TRIALS_FOR_REVIEW} trials but no systematic review or meta-analysis (evidence that has "
        "not been synthesised). 'No randomised trials' matters for interventions; for exposures that cannot be "
        "randomised it is expected.",
        "Method: an evidence gap map (Snilstveit et al. 2016, J Clin Epidemiol 79:120-129) filled by database "
        "counts rather than by coded studies, so it shows where research is likely thin, to be confirmed by "
        "screening.",
    ]


# ---------------------------------------------------------------------------------------------------------------
# Suggested axes
# ---------------------------------------------------------------------------------------------------------------

async def suggest_axes_from_rules(topic: str) -> GapMapSuggestion:
    """Rows, columns and population from the question's own words (the guided review's rule-based reading)."""
    from services.review_protocol import propose_protocol

    protocol = await propose_protocol(topic, mode="deep_research")
    rows = [MapTerm(name=c.name, synonyms=c.synonyms[:4]) for c in protocol.concepts
            if c.role in ("intervention", "exposure")]
    cols = [MapTerm(name=c.name, synonyms=c.synonyms[:4]) for c in protocol.concepts if c.role == "outcome"]
    others = [MapTerm(name=c.name, synonyms=c.synonyms[:4]) for c in protocol.concepts
              if c.role in ("topic", "method")]
    if not rows:
        rows, others = others[:1], others[1:]
    if not cols and others:
        cols = others[:1]
    pops = [c for c in protocol.populations_and_places if c.kind == "population"]
    population = MapTerm(name=pops[0].name, synonyms=pops[0].spellings[:6]) if pops else None
    return GapMapSuggestion(population=population, rows=rows, cols=cols, source="rules",
                            note="Read from the question's own words. Add more rows and columns to map the topic.")


class _ModelAxes(BaseModel):
    population: str = ""
    population_synonyms: List[str] = Field(default_factory=list)
    rows: List[MapTerm] = Field(default_factory=list)
    cols: List[MapTerm] = Field(default_factory=list)


async def suggest_axes_with_model(topic: str, config: ApiKeysConfig) -> Optional[GapMapSuggestion]:
    from services.llm_service import execute_validated_llm_task

    system = (
        "You help set up an evidence gap map. Given a research topic, propose the grid's axes: 4 to 6 rows "
        "(interventions, exposures or approaches studied for this topic) and 4 to 6 columns (outcomes that matter "
        "for it), plus the population if the topic names one. Use short standard terms as they appear in titles "
        "and abstracts, with 1 to 3 common synonyms or spellings each (no abbreviations shorter than 3 letters, no "
        "quotes or brackets). Do not repeat the population in rows or columns. Return ONLY JSON: "
        '{"population": "...", "population_synonyms": ["..."], "rows": [{"name": "...", "synonyms": ["..."]}], '
        '"cols": [{"name": "...", "synonyms": ["..."]}]}'
    )

    def check(out: _ModelAxes) -> List[str]:
        errs = []
        if not (2 <= len(out.rows) <= MAX_AXIS) or not (2 <= len(out.cols) <= MAX_AXIS):
            errs.append("Give 4 to 6 rows and 4 to 6 columns.")
        return errs

    out, record = await execute_validated_llm_task(
        task_id="GapMap_axes", task_name="Evidence gap map: suggested axes", system_prompt=system,
        user_prompt=f"Topic: {topic}", response_model=_ModelAxes, config=config, custom_validator=check,
        max_retries=1, temperature=0.2, max_tokens=900, deadline_s=60)
    if out is None or record.degraded_to_deterministic:
        return None
    clean = lambda ts: [MapTerm(name=_clean(t.name), synonyms=[_clean(s) for s in t.synonyms if _clean(s)][:3])
                        for t in ts if _clean(t.name)][:6]
    pop = MapTerm(name=_clean(out.population), synonyms=[_clean(s) for s in out.population_synonyms][:4]) \
        if _clean(out.population) else None
    return GapMapSuggestion(population=pop, rows=clean(out.rows), cols=clean(out.cols),
                            source=config.llm_model or config.llm_provider,
                            note="Suggested by the model: check and edit every term before counting.")
