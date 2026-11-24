# CiteLume evaluation protocol (pre-registered)

Written and committed on 18 November 2026, before any benchmark review was loaded or run. Later changes are made
only by new commits that say what changed and why; the git history is the record. Analyses not described here are
reported as exploratory.

## 1. Benchmark

**Primary set: CLEF eHealth TAR 2019, Task 2, intervention reviews** (github.com/CLEF-TAR/tar, MIT licence). The 40
Cochrane intervention reviews of the 2019 track (20 training and 20 test topics), each with the PubMed IDs of the
studies the review included after full-text screening ("content-level" relevance judgements).

Why this set: Cochrane intervention reviews ask effect questions, the kind CiteLume's guided protocol is built
for; their included studies are given as PubMed IDs, so a study can be found in a CiteLume corpus without fuzzy
matching; the licence allows reuse. The 2017-2019 diagnostic-accuracy, prognosis and qualitative topics are not
used: CiteLume has no diagnostic-accuracy or prognosis workflow, and including them would measure something else.

**Second set (Session 6, exploratory): SYNERGY+** (github.com/asreview/synergy-dataset, CC0), its medical reviews
only, identified by OpenAlex ID and DOI. Reported separately, never pooled with the primary set.

CSMeD (325 reviews) was considered. It aggregates the CLEF sets and others under mixed licences; using the CLEF
source directly avoids a second layer of processing.

## 2. What is taken from each review

- **Review ID** (Cochrane CD number) and **title**, from the CLEF topic file.
- **Question:** the review title, unchanged. It is what a user would plausibly type, and it is not tuned.
- **Search date limit:** the publication year of the version of the review CLEF used, found from PubMed (the latest
  version published before 1 July 2019). Searches are limited to records up to that year, so studies published
  after the review cannot count against or for CiteLume.
- **Included studies:** the content-level PubMed IDs, each resolved to its DOI, title and year through PubMed.
- **Not stored in this repository:** the Cochrane search strategies and the review abstracts (copyright of the
  publisher). Session 6's "protocol written from the review's own question and eligibility criteria" fetches the
  review's PubMed abstract at run time into a git-ignored cache.

## 3. Exclusions (decided now)

- A review is excluded if fewer than 2 of its included studies resolve to a PubMed record, or if no version of it
  published before July 2019 can be found.
- A review listed in both CLEF parts (training and test) counts once. Each exclusion is listed with its reason in `eval/benchmark/EXCLUDED.md`.
- No review is excluded for any reason that depends on CiteLume's results.
- Reviews with very many included studies (for example 245) stay in. Recall at a fixed study limit is reported per
  review and summarised both over all reviews and over reviews with at most 60 included studies, because a review
  with 245 included studies cannot reach full recall with a limit of 60.

## 4. Development and test split

- The eligible reviews are sorted by review ID, shuffled with Python's `random.Random(20261118)`, and the first
  third (rounded down) form the **development set**; the rest form the **test set**. The split is written to
  `eval/benchmark/split.json` and committed before any run.
- Sessions 5b, 5c and 5d may look only at development reviews. Session 6 reports on test reviews only. Any run on
  a test review before Session 6 is a protocol deviation and is reported as one.

## 5. Measures (Session 6)

Primary measure: **recall of the review's included studies in the candidate pool** (records after duplicate
merging), per review, summarised as the median and the mean with a 95% bootstrap confidence interval across
reviews (10,000 resamples of reviews, seed 20261118).

Secondary measures, same summaries:
- Recall in the top 20 and the top 60 selected studies, and after PRISMA screening.
- Recall with and without citation chasing.
- Screening sensitivity (included studies kept by the model screening, of those reaching screening) and precision.
- Loss by stage: share of included studies lost at search, duplicate merging, eligibility, ranking and screening.

A study counts as found when a corpus record has its PubMed ID or DOI (case-insensitive), or, for records with
neither, the same title after removing case, punctuation and spaces.

Baselines on the same pools: BM25 ranking of the pool by the question, citation count only, and random order
(seed 20261118); recall in the top 20 and top 60.

## 6. Runs

- Modes: Standard (quick answer) and Deep Research with the systematic-style (PRISMA) review type.
- Model: the model recorded in each run's `run.json`; the main runs use one model throughout. Model comparisons
  belong to Session 7.
- Sources: the open sources only (PubMed, Europe PMC, Semantic Scholar / OpenAlex, arXiv, bioRxiv/medRxiv).
  No imports, no user PDFs.
- Every run goes through `python -m services.batch`, which stores the corpus, result, timings, model and code
  commit. Failed runs are re-run once; a review that fails twice is reported as failed, not dropped.
- Results are reported as they come out. Low recall is a finding, not a reason to change the protocol.

## Amendments

- **18 November 2026, before any run:** the date rule was "the latest version published before 1 January 2019".
  Building the benchmark showed that CLEF 2019 used nine reviews whose only PubMed version is dated March 2019, so
  that rule excluded reviews CLEF itself used. The rule is now "before 1 July 2019" (CLEF assembled the 2019
  topics in the first half of 2019). Two reviews listed in both CLEF parts are counted once. No CiteLume run had
  been made; the split was drawn after this change.

- **23 November 2026, after the Session 6 PRISMA test runs: the measured pipeline is frozen.** The recall results
  reported in the paper are those of selection protocol `citelume-selection/2.5` at code commit `47e40ef` (git tag
  `eval-freeze-2.5`): question parsing, search planning and MeSH terms, PubMed depth, citation chasing and review
  reference lists, eligibility, ranking and model screening. Test results were looked at on this date (including
  two reviews that failed at the question parse), so no change to these parts may be made on their account. The
  Standard-mode test runs, made later, use the same frozen code. Later sessions (repeatability, verification gates,
  component checks, release) build on top without changing what was measured; any later change to search or
  selection gets a new protocol version and is reported as post-freeze, separately from the recall results.

- **24 November 2026, post-freeze fix (no effect on the reported results):** the bioRxiv connector
  (`services/biorxiv_client.py`) referred to an undefined name when a preprint had no DOI, so such a result ended
  that search's bioRxiv part with an error. Found in the public demo's logs. It now links the preprint to its Europe
  PMC record. The error does not appear in any evaluation run log (the one bioRxiv warning there, in a development
  run, is a different, empty-message failure), so no measured corpus was affected and the selection protocol stays
  `citelume-selection/2.5`. The new file hash and the old one are recorded in eval/FREEZE.json under `post_freeze`.
