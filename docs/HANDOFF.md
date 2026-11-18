# CiteLume: handoff for Claude Code

Written 17 Nov 2026 from a planning session in the Claude app. It covers what CiteLume is, what the owner (Bhargava) wants next, and what was proposed but not yet agreed. Read it, then inspect the code before changing anything: the planning session read the README, the docs and the module headers, but did not read function bodies or run the tests.

## The project

- **Local path:** `/Volumes/ExtremeSSD/Antigravity/litmap-studio` (the folder keeps an old name; the product is CiteLume)
- **Remote:** `https://github.com/mbsai/citelume.git`, MIT licence, public since 5 Nov 2026
- **Live demo:** Google Cloud Run, deployed with `deploy/cloudrun.sh`
- **Stack:** Python backend (`backend/main.py`, `backend/services/`, `backend/frameworks/`), Vite frontend (`frontend/src/`)
- **Tests:** `cd backend && python -m unittest discover -s tests`
- **Batch runs (no browser):** `cd backend && python -m services.batch run questions.jsonl --mode standard --out ../runs/x` (keys from `--config keys.json` or `CITELUME_LLM_*` environment variables; see the module docstring; example questions in `docs/batch_questions_example.jsonl`; `runs/` is git-ignored)
- **Replay a corpus:** `cd backend && python -m services.replay path/to/corpus.json.gz`

CiteLume takes a research question, searches PubMed, Europe PMC, arXiv, bioRxiv/medRxiv and Semantic Scholar/OpenAlex, selects 20 studies (Standard) or 60 (Deep Research), and produces four outputs: a cited synthesis, an evidence table, a knowledge-gap matrix and a literature network.

## Two goals

1. **Restructure into modules** (this document's main subject).
2. **Publish in a peer-reviewed journal.** Code changes should keep the publishable claim intact; see "Publication" below.

## Current structure and its problem

One endpoint, `POST /api/search` in `main.py`, runs the whole pipeline. All four outputs are built from the same 20 or 60 selected studies.

Bhargava's view, which the planning session agreed with:

- A knowledge gap inferred from 20 or 60 papers is not reliable. "Nobody has studied X" is a sampling artefact at that size.
- He is unsure what the literature map is for.
- He wants each output to become a separate module.

Coupling to untangle (seen in imports, not yet traced through the code):

- `services/network_engine.py` takes `KnowledgeGapMatrix`, `StructuredEvidenceMatrix` and `SynthesisEvidencePackage` as inputs.
- `services/graph_builder.py` calls `select_studies` from `paper_selection.py` and `build_multilayer_literature_network`.
- Gap logic is split between `gap_derivation.py` (author-stated open questions, design-profile gaps) and `gap_matrix_engine.py` (coverage grid, `execute_targeted_gap_verification`, `run_adversarial_gap_check`).

## Proposed design (not yet confirmed by Bhargava)

Confirm this with him before building. He had not replied to it when the session ended.

**A shared corpus layer with two tiers.**

- *Candidate pool:* every record that passes deduplication and eligibility. Hundreds to thousands of records, metadata only. Persisted with an ID and the selection-protocol version.
- *Included set:* the top 20 or 60 by the existing score, with full text where available.

**Modules as separate endpoints that take a corpus ID.**

| Module | Reads | Notes |
|---|---|---|
| Synthesis | Included set | Unchanged in substance |
| Evidence table | Included set, or a user-supplied list | Unchanged in substance |
| Knowledge gaps | Candidate pool plus database counts | Rebuild; see below |
| Literature map | Candidate pool | Reduce; see below |

**Knowledge gaps, by type.**

- *Author-stated open questions:* valid from any sample but may have been answered since. Make the targeted verification search mandatory and record its date.
- *Coverage gaps:* build the grid (population × exposure or intervention × outcome × design) and run a count query per cell against PubMed, Europe PMC and OpenAlex. Report counts, the query string and the date. No LLM in this step.
- *Contradictions:* compute within one grid cell, fetching that cell's full record set on demand.

**Literature map.** Keep the bibliographic layer and give it three jobs: citation chasing (papers cited by several included studies but absent from the pool), a cluster view of the pool, and a view of which clusters the included set came from. Park the semantic and research-evolution layers unless a use for them appears.

**Suggested order.**

1. Persist the candidate pool; split `/api/search` into corpus creation plus per-module endpoints, keeping the old endpoint working until the frontend moves over.
2. Remove the cross-module inputs to the network and graph code.
3. Rebuild gaps on counts.
4. Reduce the map.

## Invariants

- **Selection stays deterministic.** Steps 2 to 5 of the protocol (merge, eligibility, score, select) use no LLM: the same records give the same result. This is the core publishable claim. Protocol version is `citelume-selection/2.0`; bump it if selection behaviour changes.
- **Generated docs are not edited by hand.** `docs/RANKING.md` is generated from `paper_selection.py` and `query_planning.py`; `docs/FRAMEWORKS.md` comes from `python -m frameworks.export_doc`. Regenerate after changes.
- **The tool must still run without an LLM** (rule-based fallback) and with small local models.
- **Do not push or deploy without asking.** The demo is public and has daily rate limits tied to a shared key.
- The repo is on an external drive, so macOS `._*` sidecar files appear beside real files. Ignore them and keep them out of commits.

## Publication

**Claim to protect:** auditable, reproducible study selection for LLM-assisted literature reviews, independent of the model used. Synthesis quality alone is not a viable claim: OpenScholar (Nature, Feb 2026) is open source and reports expert-level citation accuracy, and GAPMAP covers LLM gap identification.

**Evidence the paper needs.** `backend/tests/test_cross_model_benchmark.py` may be a starting point.

- Overlap of included studies for the same questions across several models, a local model and no model.
- Share of a Cochrane review's included studies that reach the top 20 or 60.
- Claim-level error rate with the verification gates on and off.
- A run on ScholarQABench for comparability.
- Sensitivity of results to the score weights (50 relevance, 20 study type, 15 citations, 10 recency, 5 search agreement).

**Later, possibly a second paper:** the count-based gap module, validated against published evidence gap maps.

**Candidate journals (scope and fees unchecked):** Research Synthesis Methods; Systematic Reviews; JMIR. JOSS requires six months of public development history, so not before May 2027.

**Housekeeping still to do:** `CITATION.cff`, tagged releases, a Zenodo DOI, and an AI-usage disclosure recording what the author designed and verified.

## Open questions for Bhargava

- Does he accept the corpus-layer design, or does he want four fully independent tools?
- Should the gap module be usable without running a synthesis first?
- Keep or drop the semantic and evolution network layers?
- How large may the candidate pool grow on the free demo, given source rate limits?
