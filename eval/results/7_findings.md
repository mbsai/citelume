# Session 7 findings: repeatability and sensitivity (in progress)

Measured on the stored Session 6 test corpora (frozen pipeline, protocol 2.5). No model calls were needed for the
parts done so far.

## Done

**E2a. Selection given the same records is exactly repeatable.** All 44 stored test corpora (22 systematic-style,
22 Standard) reproduce their included studies, in the same order, with matching content hashes, from their stored
records alone (`services.replay`; no model, no network). The model's screening decisions are stored with the corpus
and replayed, so this covers the systematic-style selection too.

**E3. Sensitivity to weights and caps** (`7_sensitivity_test.md`, figure `7_sensitivity_test.png`; one command:
`python -m eval.sensitivity --standard runs/6_test_standard --prisma runs/6_test_prisma --set test --name
7_sensitivity_test`). Each weight or cap at half and one and a half times its value:
- Score weights (concept match, wording overlap, study type, citation impact, recency) change about one of the 20
  Standard selections (median overlap 0.90-0.95, lowest 0.54-0.82) with rank correlation 0.85-0.95 among the
  studies kept, and change recall by at most 0.3 percentage points.
- Halving the review share caps changes the Standard selection most (median overlap 0.57) without changing recall.
- In the systematic-style runs the records the model would screen are unchanged in most reviews (median overlap
  1.00, lowest 0.85): most pools have fewer than 300 eligible records.
So the reported recall does not hinge on the chosen weights; it is set by search and eligibility.

## To run next (needs model calls)

**E2b. Screening repeatability on the same records** (`eval/rescreen.py`): a fixed sample per review of up to 96
records the stored run screened (every screened included study first, then a seeded random fill), screened again
with the frozen prompt and criteria; agreement, Cohen's kappa and sensitivity against the stored decisions and
between runs.
- Nemotron 3 Ultra, free endpoint, three repeats (about 176 calls each, about 530 in all: one day's free quota).
- One other free model (Gemma 4 31B, `google/gemma-4-31b-it:free`), about 176 calls.
- Optional: the local `qwen3.5:9b` through Ollama (about 4 minutes a review, 80 minutes in all; 2 of 22 done on
  23 Nov and kept, the run resumes), and a paid frontier model (about 3-4 USD; ask first).
- A two-review check on 23 Nov (local model against the stored decisions, 192 records): 77% agreement, kappa 0.52.
  Too small to report; shown only to confirm the tool works.

**E2c. Records found on repeated searches.** Two parts:
- Database variation: build each test review's corpus twice more with no model (rule-based planning; free; about
  3-5 minutes a review), then compare the record sets (Jaccard) and the included studies found.
- Planning variation: plan each question three times with Nemotron (planning calls only) and compare the concepts
  and PubMed strings. Needs a small script (`eval/replan.py`), to write next session.
