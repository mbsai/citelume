# Session 7 findings: repeatability and sensitivity

Measured on the 22 test reviews and their stored Session 6 corpora (frozen pipeline, protocol 2.5), 23-24 Nov 2026.
Model runs used Nemotron 3 Ultra on OpenRouter's free endpoint; no cost.

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

**E2b. Screening on the same records is highly repeatable** (`7_screening_agreement.md`; `python -m eval.rescreen
report --prisma runs/6_test_prisma --set test`). The same sample of 1,638 records (up to 96 per review, every
screened included study first) screened three more times by the same model with the frozen prompt:
- Repeat against repeat: 96-97% agreement, Cohen's kappa 0.91-0.94.
- Repeats against the original Session 6 decisions: 90-91%, kappa 0.80-0.82. The original run screened records in
  batches of 12 in rank order, the repeats in sample order, so some of this difference is the batch a record was
  screened with; the prompt and model were the same.
- Sensitivity on the reviews' included studies: 92% originally, 93% in each repeat (117 included studies screened).
- A second model could not be measured: Google's free Gemma 4 31B, and the other free models tried, were
  rate-limited upstream on both days. The local `qwen3.5:9b` screened 2 of 22 reviews before it was stopped as too
  slow (192 records: 77% agreement with the original decisions, kappa 0.52); too few to report as a result.

**E2c. Records found vary with source availability, not with chance** (`7_search_variation.md`; `python -m
eval.search_variation --runs runs/7_search_r1 runs/7_search_r2 --stored runs/6_test_prisma --set test`). Each test
question was searched twice more with no model (rule-based planning):
- The two identical repeats share a median 63% of their records (lowest 24%); their pool recall is similar (34% and
  37%).
- The difference comes from sources that did not answer in one run: searches that ran past the 20-second budget
  (Semantic Scholar, PubMed, bioRxiv) and citation searching that Semantic Scholar rate-limited (no API key), which
  is all or nothing (for example 0 against 100 records added). Every corpus records these warnings. The repeats
  also ran alongside the screening repeats, which may have added load.
- So records found are repeatable only as far as the sources answer; reproduction runs should use API keys and
  longer time budgets, and the stored corpus, not a re-run, is the record of what was found.

**E2d. Model planning adds about 20 points of pool recall, and names stable concepts** (`7_planning_variation.md`;
`python -m eval.replan report --prisma runs/6_test_prisma --set test`).
- The rule-planned repeats share only a median 10-12% of their records with the model-planned Session 6 run, and
  hold 34-37% of the included studies against 54%: the model's plan finds more of what the reviews included.
- Planning the same question three more times with the model: the concept names match the stored ones (median
  overlap 1.00; identical in every new plan for 14 of 22 questions), but the exact core PubMed search matches in
  only 14 of 66 plans, because the synonyms and aspects the model adds vary.

## What the paper can claim

- Selection is exactly reproducible from a stored corpus (E2a), and its result does not hinge on the chosen weights
  (E3).
- The model's screening is highly consistent on the same records (kappa above 0.9 between repeats).
- What is found depends on the model's search plan (it adds recall; its synonyms vary) and on which sources answer;
  the corpus stores both, so a review can be audited and replayed even when a new search would differ.

## Not done

- A second screening model (free models rate-limited upstream; a paid frontier model was not run).
- Records-found repeats with model planning (each costs a full systematic-style run).
