# CiteLume: plan to a peer-reviewed publication

Written 17 Nov 2026 after inspecting the code (commit `35da30c`, 172 tests passing) and checking journals, comparable
tools and benchmarks. It follows `docs/HANDOFF.md` and the decision to build the shared corpus layer. Each numbered
session below is sized for one working session and ends with a "done when" check. Start a session by reading this
file and `docs/HANDOFF.md`.

**Progress:** Sessions 1-3, 3b and 3c done 17 Nov 2026, plus study limits by review type (selection protocol 2.3:
quick answer 20; narrative 60 by default; systematic-style and scoping reviews take every eligible study up to a
ceiling of 150 / 200 locally and 60 on the demo; the number can be lowered after the search without searching
again). Session 3d done 17 Nov 2026: Scopus / Web of Science strings, imports (RIS, BibTeX, CSV, Web of Science
and PubMed text) and user PDFs (attach or add; include regardless with a recorded reason). Decision: imports and PDFs
are disabled on the public demo. Session 3e done 17 Nov 2026: the model answers each protocol criterion (yes / no /
unclear); a fixed rule decides (any "no" excludes, the reason named in the order design, population, intervention,
outcome); answers stored with model, prompt version and date; user overrides with reasons. Second-model screening:
decided by Session 6 (if one model keeps fewer than 95% of the benchmark reviews' included studies, add a second
model, keeping a record if either includes it and flagging disagreements; otherwise report single-model sensitivity
and two-model agreement measured offline; check the threshold against the screening-automation literature first). Session 3f done 17 Nov 2026: risk of bias from the full text only (others "Not assessed"), every answer backed
by a quote checked by code, RoB 2 domains computed from signalling answers (frameworks/rob2.py; checked 18 Nov 2026
against every branch of the decision figures in the official RoB 2 cribsheet of 22 Aug 2019, which corrected two
branches: 2.5 NI now gives High, and 4.1 NI no longer lowers the measurement domain; the adhering-to-intervention
variant is not covered), reviewer edits recorded, Unpaywall when
UNPAYWALL_EMAIL is set. First real run: 4 of 18 studies had full text; 34 of 36 trial quotes verified; many
domains High because of missing information, as RoB 2's rules require. Session 4 done 18 Nov 2026: `python -m services.batch run` writes corpus.json.gz, result.json and run.json (status, timings per stage, model, settings without keys, versions, code commit) per question plus summary.csv; resumable; one question at a time with a pause; refuses demo mode and the demo key. First run: 5 standard questions end to end (Nemotron, 113-152 s each, each corpus replays); the model's relevance check dropped 1-5 of 20 selected studies per question, worth measuring in Session 6. Session 4b done 18 Nov 2026: citation chasing (services/citation_chasing.py, on for
Deep Research, a request switch otherwise): Semantic Scholar batch requests fetch the references and citing papers of
up to 60 selected studies; papers linked to at least 2 of them and not in the pool (up to 100) are added as records
identified by other methods, missing abstracts and publication types are filled from PubMed, and selection runs
again over the enlarged pool (stored and replayed); counted separately in the PRISMA flow. The map shows citation
links only by default (word similarity, lineage, Hugging Face and the claim / dataset / gap layer, now in
services/network_evidence_layer.py, behind a "view all" switch); it builds from a stored corpus alone and carries a
view of the whole pool (services/pool_map.py: TF-IDF k-means groups, included studies marked, large groups with none
included flagged, click to list). Real runs (Deep Research, Nemotron): vitamin D and fractures 100 papers added,
19 included, among them the WHI, Chapuy, Trivedi and RECORD trials the searches had missed (before the PubMed
abstract fill, 46 of 100 chased records had no abstract and only 11 were included); exercise and depression 100
added, 3 included. Seeds with reference lists: 31 of 58 and 35 of 56. Both corpora replay. Session 5 done 18 Nov 2026: eval/PROTOCOL.md pre-registered and committed first
(3b00c62); benchmark = CLEF TAR 2019 intervention reviews (MIT licence), built by eval/build_benchmark.py through
PubMed: 33 reviews, 785 included studies all resolved to PubMed IDs (714 with DOIs), 5 reviews excluded for fewer
than 2 included studies; seeded split 11 development / 22 test, committed before any run (f68fa5a). One protocol
amendment before any run (review-version date rule; recorded in the protocol). eval/scoring.py matches included
studies to corpus records and refuses test reviews before Session 6; `python -m eval.scoring questions
development` writes a batch-runner file. Short of the 40-50 target (CLEF 2019 has only 38 intervention topics); decision 18 Nov 2026: keep the 33 reviews,
SYNERGY+ stays an exploratory second set for Session 6 only.
Session 5b done 18 Nov 2026 (eval/results/5b_findings.md): on the 11
development reviews CiteLume selected 8 of 141 included studies (6%); search lost 72%, eligibility 21% (76% of the
studies found), merging and screening 1% each. Search depth explains only 10 of the 102 search losses; the rest is
query sensitivity (question words only, no MeSH or drug-class terms; a slash-parsing defect). So 5c goes first,
reordered: MeSH / pharmacological-action terms, synonyms and the slash fix before deeper paging. Then 5d: MeSH and
synonym matching in eligibility, outcome phrases not required, title-only records assessed on their titles.
Session 5c done 18 Nov 2026 (eval/results/5c_findings.md): MeSH headings / pharmacological actions in
the PubMed queries (services/mesh_terms.py), review-title parsing fixes (dashes, slashes, purpose words, class
labels, bracketed abbreviations), PubMed depth setting (`pubmed_depth`, local and evaluation only) with "retrieved N
of M" logged and reported, reference lists of excluded reviews as a second "other method". Development set:
included studies found 28% -> 36% (MeSH) -> 55% (plus depth 500 and review reference lists); selected 6% -> 17%.
Eligibility now loses 53 of the 78 found. Session 5d built 18 Nov 2026 (selection protocol 2.4; 57b5935,
702073c): eligibility reads PubMed's MeSH indexing of each record (exploded headings, stored on the record as
mesh_concepts); in systematic-style reviews partial matches go to the model's screening (and stay excluded when no
model answers); title-only records kept when the title or indexing matches. Concept position and the embedding
option were not built: the development data show recall, not precision, is the loss. Development re-run
incomplete: OpenRouter's free daily quota ran out after 2 of 11 reviews (screening load rose to 100-300 decisions
per review); early signal CD008018 0 -> 5 of 13 selected. Next: finish the 5d development re-run, then Session 6.
Session 3c notes: the source check matches sentences by shared words (a reading aid, not proof); the guided review is
the page's Deep Research mode (backend mode name unchanged: `deep_research`).
Session 3 note: the evidence table and gaps still share one stored analysis with the synthesis, because both depend
on the extraction and claims the synthesis run produces; the map can be built from a corpus alone. The evidence-table
endpoint does not yet accept a user-supplied list of studies.
Session 3b notes: without a protocol the model may still widen the searches (selection stays rule-based) and the
default protocol is recorded; with a protocol no model plans anything. Preview counts for Semantic Scholar and arXiv
are often unavailable without a key because of rate limits. Selection protocol is now `citelume-selection/2.2`.

Standing rules: do not push or deploy without asking; do not hand-edit `docs/RANKING.md` or `docs/FRAMEWORKS.md`
(regenerate them); the tool must still run with no model and with small local models; keep `._*` files out of commits.

## 1. What the paper claims

**Claim:** protocol-first, auditable and repeatable study selection for LLM-assisted literature reviews. The user's
decisions (review type, question framing, eligibility criteria) are captured as an explicit review protocol before
any search runs. Given a stored corpus (the records found, that protocol and the selection-protocol version), anyone
can re-run selection and get the same included studies, with the reason for every inclusion and exclusion, whatever
model wrote the synthesis.

**Product shape (decided 17 Nov 2026):** two modes. *Quick answer* (today's Standard synthesis) and *Guided review*,
which replaces Deep Research. Accepting every default in the guided review gives what Deep Research gives today.

Why this claim and not synthesis quality:

- OpenScholar (Nature, 4 Feb 2026, doi:10.1038/s41586-025-10072-4) is open source and reports citation accuracy on a
  par with experts on ScholarQABench. CiteLume will not beat it on synthesis quality.
- Repeatability is a documented weakness of current tools. A study in *Research Synthesis Methods* found Elicit's
  results varied with prompt design, user account and algorithm changes. That is the gap CiteLume addresses.
- Cochrane, Campbell, JBI and CEE issued a joint position statement (2025) asking tool developers to follow the RAISE
  recommendations. RAISE 2 covers building and evaluating tools (living document on OSF, osf.io/fwaud; version 3 was
  reported as released 13 Mar 2026). A paper that reports against RAISE 2 speaks the reviewers' language.

What the field says it lacks (checked 17 Nov 2026; published studies and librarian commentary, no forum threads):

- Repeatability and transparency: a May 2026 benchmark (arXiv:2605.10125) found literature review tools had low
  reproducibility, rarely disclosed their sources and were unsuitable for systematic reviews.
- Verification: the same benchmark found highlighted source passages often did not match the generated answer.
- Oversight people actually use: a Sept 2026 study of a system with mandatory human checkpoints (arXiv:2609.22248,
  63 participants) found 35% would trust AI summaries without further checking. Guided steps must therefore ask for
  a real decision, not a click on "Next".
- Assistance, not replacement: a 2025 BMJ Open choice experiment found reviewers want high sensitivity and lower
  workload with modest skill needed.
- Help with framing: studies of graduate students find the hard parts are the question, the scope and choosing
  relevant papers.

## 2. What stands between the code and that claim

Found by reading the code, most important first.

| # | Problem | Why it matters for the paper | Fixed in |
|---|---|---|---|
| 1 | Selection scores studies against concepts that come from the model's search plan when a model is configured | "Independent of the model" is not true today | Session 1 |
| 2 | PRISMA mode adds model screening to selection | Must be reported as a separate, optional, logged step, not part of the deterministic protocol | Session 1 |
| 3 | Protocol version still `citelume-selection/2.0` after selection behaviour changed in `35da30c` | Version must identify behaviour | Session 1 |
| 4 | The candidate pool is discarded; deduplication and selection modify records in place and run twice in Deep Research | Nothing can be re-run or audited after the request ends | Sessions 1-2 |
| 5 | No evaluation exists beyond unit tests | A methods journal needs measured recall, repeatability and error rates | Sessions 5-9 |
| 6 | No way to run questions in bulk without the web page | Experiments need a headless runner | Session 4 |
| 7 | Gap verification never searches (it checks 10 leftover records by word overlap); gaps come from 20 or 60 studies | Indefensible if described as "verified gaps" | Paper 1 labels gaps exploratory; rebuilt in Session 11 |
| 8 | Literature map has no stated purpose | Reviewers will ask what it is for | Wording fixed in Session 1; rebuilt around citation chasing in Session 4b |
| 9 | No release hygiene: no CI, unpinned dependencies, no `CITATION.cff`, no tagged release or DOI | Required or expected for software papers | Session 13 |
| 10 | Matching is by words, English only, mostly abstracts; weights (50/20/15/10/5) are judgement calls | Must be stated as limitations and tested for sensitivity | Sessions 5d, 7, 14 |
| 11 | Searches fetch the first 50-100 records per database; a Cochrane review screens thousands | Low recall is the biggest risk to acceptance | Sessions 5b, 5c, 4b, 3d |
| 12 | PRISMA screening is one yes/no model call per record, not stored | The repeatability claim stops at that step | Session 3e |
| 13 | Risk of bias is rated from abstracts with no supporting quote and never validated | Indefensible as presented | Sessions 3f, 9 |

Assumed answers to the open questions (change these here if you decide otherwise):

- Selection reads the review protocol (Session 3b). Its default comes from the rule-based question parsing; the
  model may only suggest synonyms for the user to accept. Until Session 3b, selection uses the rule-based parse
  directly (Session 1). The protocol used is stored with the corpus.
- The gap module may run without a synthesis for coverage counts.
- Semantic and research-evolution map layers are parked behind a switch, not deleted.
- Demo candidate pools stay at today's size (about 200-400 unique records), stored for 24 hours.
- Undecided: whether the public demo may store the question text with the corpus. Until decided, the demo stores the
  corpus without the question text and experiments run locally.

## 3. Sessions

### Phase A: make selection reproducible (corpus layer)

**Session 1. Guard rails and a clean selection step.**
- Add a fixture test that runs `/api/search` with fake sources and the test responder and records included IDs and
  response shape (the "golden" test).
- Move duplicate merging out of `graph_builder.py` into `services/corpus.py`; make merging and selection work on
  copies; remove the second selection call's side effects in Deep Research.
- Selection reads concepts from the rule-based parse; record which plan was used.
- Name the PRISMA model screening as its own logged step with per-record decisions kept.
- Bump to `citelume-selection/2.1`; regenerate `docs/RANKING.md`.
- Honest wording for gaps and the map, on the page and in exports: rename the gap view to "Open questions and thin
  areas in these studies"; remove "verified" and "corroborated" (no search verifies anything today); label the grid
  as coverage within the included studies, not the field; keep the authors' own stated open questions. Describe the
  map as links among the included studies only.
- Done when: golden test passes before and after; a new test shows identical included IDs with and without a model
  configured for the same records.

**Session 2. Corpus model, store and replay.**
- `select_studies` returns every eligible record with its score breakdown and every excluded record with its reason.
- `Corpus` model: id, creation and search dates, protocol version, question, parsed concepts, plan, search strings,
  per-source counts, candidates, included IDs, selection report, content hash.
- File store (JSON, gzip) with expiry: `backend/.cache/corpora` locally, the mounted bucket on the demo.
- `python -m services.replay <corpus file>` re-runs selection from the stored candidates.
- Done when: replay of a saved corpus reproduces the included IDs and hash exactly; store round-trip test passes.

**Session 3. Stage functions and module endpoints.**
- Split `_run_research_search` into `build_corpus`, extraction, narrative, evidence table, gaps, map; pass results
  explicitly instead of through `get_last_synthesis_package()`.
- Endpoints: `POST /api/corpus`, `GET /api/corpus/{id}`, `POST /api/corpus/{id}/synthesis | evidence-table | gaps | map`.
- `/api/search` calls the same functions and returns the same response plus `corpus_id`.
- Demo: allowance charged at corpus creation; module runs limited per corpus.
- Done when: golden test passes unchanged; each endpoint has a test.

**Session 3b. Review protocol (backend).**
- `ReviewProtocol` model stored in the corpus: review type and why; question framing (population, intervention or
  exposure, comparison, outcome, or the equivalent for non-clinical questions) with synonyms for each part;
  eligibility (study designs, years, population or place, how existing reviews are treated); for each field, whether
  the user set it or the default was kept.
- `propose_protocol(question)` builds the default from the rule-based parse (the model may suggest synonyms, shown as
  suggestions only). `POST /api/protocol/propose` returns it; `POST /api/corpus` accepts a protocol.
- Searches and selection read the protocol instead of the parsed concepts. A quick answer uses the default protocol
  unchanged, so both modes go through one path.
- A search preview: the strings for each database and a result count per database before the full run.
- Done when: the same protocol and records give the same included IDs; editing a synonym or criterion changes the
  search strings and selection as expected; the golden test still passes with the default protocol.

**Session 3c. Guided review (page).**
- Replace the Deep Research option with a guided review of four steps, each needing a decision: (1) purpose, leading
  to a recommended review type with the reason shown; (2) confirm or edit each part of the question and its synonyms;
  (3) eligibility criteria; (4) search preview with counts, then run.
- Each step states what the choice changes downstream. "Accept defaults" is allowed but recorded as such.
- Verification in one click: each claim in the synthesis opens the source sentence it rests on (the tool already
  checks figures against source text).
- The protocol is shown with the results and included in every export.
- Done when: a guided review runs end to end in the browser; the exported report contains the protocol; a default
  run matches today's Deep Research output for the golden question.

**Session 3d. Imports: other databases and the user's own PDFs.**
- Search strings for databases CiteLume cannot query: written by rule from the protocol in Scopus
  (`TITLE-ABS-KEY(...)`) and Web of Science (`TS=(...)`) syntax, shown in the search preview step with a copy button.
- Import of the user's export (RIS, BibTeX, CSV; PubMed NBIB if simple): records join the candidate pool, are merged
  with duplicates, labelled by origin (for example "Scopus (imported by user)") and go through the same eligibility
  and scoring. The corpus stores the string, the date and the record count; the PRISMA flow shows them.
- User PDFs: a PDF either attaches full text to a study already in the pool (matched by DOI or title) or adds a
  study as "identified by other methods". Added studies still pass the eligibility check; an override needs a
  recorded reason. No OCR: PDFs without a text layer are refused with a clear message.
- Licensed content: imports and uploaded PDFs are the user's licensed material. Allowed in local installs; on the
  public demo either disabled or deleted when the run ends (decide before building). The page states that excerpts
  of uploaded full text are sent to the configured model and points to the local-model option.
- Done when: a Scopus RIS export and a Web of Science export import correctly with duplicates merged against PubMed
  records; an uploaded PDF attaches to the right study; replay of the corpus still reproduces the included IDs.

**Session 3e. Screening by criterion, stored and confirmed.** (Replaces the single yes/no model screening in
`services/ai_screening.py`. It cannot be made deterministic, so it is made auditable, replayable and measured.)
- For each record the model answers each protocol criterion separately (population, intervention or exposure,
  outcome, study design) as yes, no or unclear, with a short reason. A fixed rule computes the decision; "unclear"
  counts as include. Every exclusion names the criterion that failed.
- Every answer is stored in the corpus with the model, prompt version and date. Replay uses the stored answers, so a
  review is reproduced exactly even though a fresh run might differ.
- The page lists the model's exclusions for the user to confirm or overturn; overrides are recorded. Optional
  second model: disagreements between the two go to the user.
- Without a model the rule-based eligibility stands, as now.
- Done when: replay reproduces the screened set; each exclusion shows its failed criterion; an override changes the
  included set and appears in the flow diagram.

**Session 3f. Risk of bias that can be checked.**
- Judge only when full text is available (open access or a user PDF); otherwise "Not assessed (abstract only)",
  which replaces most of today's "No information". Add Unpaywall and PubMed Central lookups to raise coverage.
- Each domain judgement needs a supporting quote; code checks the quote exists in the source text (as figures are
  checked now) and the table shows it. A judgement without a verifiable quote becomes "No information".
- For randomised trials, ask the RoB 2 signalling questions and compute each domain by the tool's published
  algorithm, in code. Other designs keep the domain-level judgement.
- Choose the tool from PubMed publication types first, then wording; let the user correct the design.
- Users can edit any judgement; edits are recorded. The page and exports call these "suggested judgements for the
  reviewer to confirm".
- Done when: every shown judgement has a quote found in the source; abstract-only studies are "Not assessed"; a
  trial's domain ratings follow from its signalling answers by the coded algorithm (unit-tested).

**Session 4. Headless runner.**
- `python -m services.batch run questions.jsonl --mode standard --out runs/` saving the corpus, outputs, timings and
  the model used for each question; resumable; respects source rate limits; uses your own keys, never the demo key.
- Done when: 5 questions run end to end from the command line with no browser.

**Session 4b. Literature map: citation chasing and a view of the pool.** (Moved here from Phase C because citation
chasing adds to the candidate pool and so affects the recall measured in Session 6.)
- Remove the gap, evidence and synthesis inputs from `network_engine.py`; citation links only by default. Word
  similarity, research-evolution, claim, gap and Hugging Face nodes go behind a switch, not deleted.
- Job 1, citation chasing: papers cited by, or citing, several included studies but absent from the pool are added
  as "identified by other methods" and pass through eligibility like any other record.
- Job 2, where the included studies sit: cluster the whole candidate pool and mark the clusters the included studies
  came from; flag large clusters with none included.
- Job 3, navigation: clicking a cluster lists its studies.
- Report how many studies had reference data (13 of 20 in one earlier run) and stay within the Semantic Scholar and
  OpenAlex allowances.
- Done when: the map builds from a stored corpus alone; chased records appear in the flow diagram with their own
  count; map tests no longer import gap or synthesis models.

### Phase B: evidence for the paper

**Session 5. Gold standard.**
- Choose the benchmark: CSMeD (325 reviews, medicine and computer science) or CLEF TAR (Cochrane reviews); SYNERGY
  (26 reviews) as a second set. Check licences.
- Script that turns each review into: the question, the search date limit, and the list of included study IDs.
- Pre-register the analysis in a short file (`eval/PROTOCOL.md`): questions, metrics, exclusions. Commit it before
  running anything.
- Split the reviews into a development set (about a third) and a held-out test set, fixed by a seeded shuffle and
  committed. Sessions 5b-5d may look only at the development set; Session 6 reports on the test set. Without this
  split, improving recall on the benchmark and then reporting recall on it would overstate the result.
- Done when: 40-50 reviews are loaded with their included studies resolved to PubMed IDs or DOIs, and the split is
  committed.

**Session 5b. Where studies are lost (development set only).**
- For each development review, count its included studies still present after each stage: found by the searches,
  after duplicate merging, after eligibility, in the top 60, after screening. One table, losses by stage.
- Check how many of the missed studies are indexed in the searched databases at all (the ceiling for any fix).
- Done when: the table exists and names the stage with the largest loss. Sessions 5c and 5d are sized by it: if the
  search stage loses most, do 5c first; if eligibility or ranking does, do 5d first.

**Session 5c. Finding more: deeper and more sensitive searching.**
- Page through all matching records in PubMed and Europe PMC (metadata only, up to a stated cap) instead of the
  first 50-100; show "retrieved N of M" for each search. Local installs and evaluation only; the demo keeps a cap.
- Add MeSH headings to the PubMed strings, and published search filters where the protocol asks for a design (for
  example the Cochrane sensitive filter for randomised trials). Cite each filter.
- Harvest the reference lists of the reviews excluded at eligibility: their cited studies enter the pool as
  "identified from reference lists of reviews", with their own count in the flow diagram.
- Optional source: ClinicalTrials.gov.
- Done when: the loss-by-stage table is re-run on the development set and shows the change at the search stage;
  source allowances are respected; the golden test passes at default depth.

**Session 5d. Matching on meaning, by rule.**
- Match concepts on MeSH headings as well as text where records carry them (for example the heading "Aged" for
  older adults); text matching stays for records without headings.
- Concept position: a concept in the title, objective or conclusion counts as the study's subject; one only in the
  background counts as a mention and no longer satisfies eligibility on its own.
- Optional, behind a switch: a small pinned embedding model run locally replaces the word-overlap part of the
  relevance score (same version, same score); word overlap stays the default and the fallback.
- Bump the selection protocol version; regenerate `docs/RANKING.md`.
- Done when: on the development set, fewer eligible-by-words records are rejected by screening (precision up)
  without losing included studies (recall not down); replay tests pass under the new version.

**Session 6. Recall experiment (E1).**
- For each review question, build a corpus (Standard, Deep Research, PRISMA) and measure: share of the review's
  included studies in the candidate pool, in the top 20 and top 60, and after PRISMA screening; precision of the
  model screening against the review's decisions.
- Run each review twice: with the automatic default protocol, and with a protocol written from the review's own
  stated question and eligibility criteria. The difference measures what the guided protocol adds.
- Run on the held-out test set only. Report the loss-by-stage table for it.
- Screening (Session 3e): sensitivity against the reviews' included studies, which matters more than precision;
  agreement between three fresh runs and between models; share of exclusions per failed criterion.
- Report recall with and without citation chasing (Session 4b), as its own number.
- Use only the open sources for these runs; imports (Session 3d) are described as a feature, not measured.
- Baselines on the same pools: BM25 only, citations only, random.
- Done when: a results table and one figure exist, with confidence intervals across reviews. Report the numbers as
  they come out; low top-60 recall is a finding that motivates the candidate pool, not a reason to tune.

**Session 7. Repeatability and sensitivity (E2, E3).** Runs offline on stored corpora.
- E2: same questions with no model, a small local model, Nemotron and one frontier model, three repeats each. Report
  overlap of included sets (selection, given the same records) and, separately, overlap of the records found.
- E3: vary each weight and cap; report how much the included set changes (set overlap and rank correlation).
- Done when: both tables exist and the selection-overlap figure is reproducible from stored corpora with one command.

**Session 8. Verification gates (E4).**
- Run 30 questions with the gates on and off; sample about 200 claims; export an annotation sheet (claim, cited
  study, source sentence).
- Human step: you, and ideally a second annotator, mark each claim supported or not. Report agreement.
- Done when: claim-level error rates with and without gates, with inter-annotator agreement.

**Session 9. Component checks.**
- Study-type classifier against PubMed publication types (free gold labels) on a few thousand records.
- Duplicate merging: precision on a hand-checked sample of merged pairs.
- Risk of bias (Session 3f): for trials in the benchmark reviews that have published Cochrane judgements, compare
  per domain and report agreement. Find and cite the published agreement between human reviewers to use as the
  yardstick (to be confirmed with a citation; do not state a figure from memory). If agreement is poor, the paper
  describes the judgements as exploratory.
- Optional: ScholarQABench run for comparability, reported without claiming superiority.
- Optional user study of the guided review (needed before the paper can call it easier or more intuitive): 10-15
  participants, novices and experienced reviewers, the same task with and without guidance; measure time, a standard
  usability score, how many defaults were changed, and whether participants opened the source sentences. Check
  whether your institution requires ethics approval first.
- Done when: accuracy tables for both components.

### Phase C: modules (can follow submission of paper 1)

**Session 10.** Moved to Phase A as Session 4b.

**Session 11. Count-based coverage gaps.** Grid of population x exposure or intervention x outcome x design; one
count query per cell on PubMed, Europe PMC and OpenAlex; report counts, query string and date; no model in this step.
Author-stated open questions get a real, dated verification search.

**Session 12. Validate gaps against published evidence gap maps.** This is the basis of a possible second paper.

### Phase D: release and manuscript

**Session 13. Release hygiene.**
- GitHub Actions running the tests; pinned dependencies; `CITATION.cff`; tag `v1.0.0`; Zenodo DOI; archive the
  evaluation corpora and results with their own DOI. The archive must contain no imported Scopus or Web of Science
  records and no uploaded PDFs.
- `docs/AI_USAGE.md`: what was written with AI assistance and what you designed, decided and verified.
- RAISE 2 table: each recommendation, where CiteLume meets it, where it does not.
- Remove internal labels ("Phase 7", "Level A/B/C", "Universal, Multilayer...") from the page and exports.
- Done when: a fresh clone installs, passes tests in CI and reproduces one results table from the archived corpora.

**Session 14. Manuscript.** Structure: background (repeatability problem, RAISE); the protocol (from
`docs/RANKING.md`); implementation; evaluation (E1-E4, components); limitations; availability. Report the search
method to PRISMA-S. Figures: pipeline, PRISMA flow example, recall, repeatability.

**Session 15. Pre-submission review.** Internal peer review of the draft, reference check, journal checklist, cover
letter, preprint.

## 4. Where to submit

| Journal | Fit | Cost (unverified: confirm on the journal's own page) |
|---|---|---|
| *Research Synthesis Methods* (Cambridge) | Best: methods for evidence synthesis, software in scope, published the Elicit repeatability study | All articles open access since 2025; covered by many institutional agreements; waiver available without funding; an older listing gave about EUR 2,970 |
| *Systematic Reviews* (BMC) | Good: methodology and software articles | An aggregator lists about USD 2,790; check fee waivers |
| *JMIR* family | Possible: broader digital-health readership | Not checked |
| JOSS | Not before May 2027 (needs six months of public history); software-only review | Free |

## 5. Things only you can do

- Decide the journal and confirm fees, article type and its AI-use policy. Read RAISE 2 on OSF.
- Consider a co-author who does systematic reviews or evidence-synthesis methods; it would strengthen the paper and
  provides the second annotator for Session 8.
- Provide your own API keys for the evaluation runs (OpenRouter or another provider, Semantic Scholar, OpenAlex).
- Annotate the claim sample (Session 8).
- ORCID, Zenodo-GitHub link, and the decision on storing question text on the demo.

## 6. Order

Sessions 1-3, 3b-3f, 4, 4b, then 5, 5b-5d, 6-9, then 13-15. Sessions 11-12 (count-based gaps and their validation)
can wait until paper 1 is submitted, provided the manuscript describes gaps as exploratory. Rough size: 21 sessions
to submission, plus the time the evaluation runs
take under free-tier rate limits (plan on several days of unattended running for Session 6).

## Sources

- OpenScholar: https://allenai.org/blog/openscholar-nature
- Joint position statement on AI in evidence synthesis: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12594113/
- RAISE guidance (Cochrane event page): https://www.cochrane.org/events/recommendations-and-guidance-responsible-ai-evidence-synthesis
- Elicit accuracy and repeatability study: https://www.cambridge.org/core/product/C97DAEC70C3173A260F0B12E729E7250/core-reader
- AI research tools benchmark (May 2026): https://arxiv.org/abs/2605.10125
- CoSLR checkpoint and trust study (Sept 2026): https://arxiv.org/abs/2609.22248
- AI adoption in systematic literature reviews, choice experiment (BMJ Open 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12530368
- CSMeD: https://proceedings.neurips.cc/paper_files/paper/2023/hash/4962a23916103301b27bde29a27642e8-Abstract.html
- Benchmark overview (SYNERGY, CLEF TAR): https://arxiv.org/html/2505.24757v1
- *Research Synthesis Methods* author instructions: https://www.cambridge.org/core/journals/research-synthesis-methods/information/author-instructions
