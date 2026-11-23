# Session 6 findings: recall on the held-out test set (interim: PRISMA runs)

Runs `runs/6_test_prisma`, 23 Nov 2026, code 47e40ef (selection protocol 2.5): 22 test reviews, Deep Research with
the systematic-style (PRISMA) review type, PubMed core search 500 deep, citation chasing and the reference lists of
excluded reviews on, each review's title as the question and its year as the upper date limit. Model: Nemotron 3
Ultra on OpenRouter's free endpoint (`nvidia/nemotron-3-ultra-550b-a55b:free`) for planning, screening and analysis;
all 22 searches were planned by the model. No cost.

Tables: `6_test_prisma.md` (pre-registered measures, per review, baselines; figure `6_test_prisma.png`) and
`6_test_loss_by_stage.md`. The Standard-mode runs are still to be made (the free daily quota ran out at the end of
the PRISMA runs).

## Results (mean of per-review shares, 95% bootstrap CI; 22 reviews)

| Measure | All reviews | Reviews with <= 60 included (19) |
|---|---|---|
| Pool recall (primary) | 54% (41-67%), median 49% | 60% (48-73%) |
| Pool recall, databases only | 50% (36-63%) | 55% (41-70%) |
| Top 20 of the screening order | 15% (9-22%) | 18% (10-25%) |
| Top 60 | 22% (14-31%) | 26% (17-35%) |
| Selected after model screening | 25% (15-35%) | 28% (17-39%) |
| Screening sensitivity (17 reviews with an included study screened) | 87% (72-99%) | 86% (68-99%) |
| Screening precision (20 reviews) | 10% (5-17%) | 10% (5-17%) |

Baselines on the same pools, top 60: BM25 of the question 15% (8-24%), citation count 11% (6-17%), random 3%
(1-7%). Top 20: BM25 7%, citations 1%, random 0%. CiteLume's order recalls more than each baseline at both cut-offs.

Pooled over all 644 included studies (one review has 245): found 32%, eligible 22%, selected 16%. Losses: search
68%, eligibility 10%, screening and the study limit 6%, duplicate merging 0%.

## What the numbers say

- **Search depth is the largest loss on the test set.** 266 of the 435 included studies no search returned are
  matched by CiteLume's own PubMed queries beyond the first 500 records; the reviews with many included studies
  (68, 81 and 245) lose most this way.
- **Two reviews failed at the question parse**, with 0 selected: CD008170 ("First-line drugs inhibiting the renin
  angiotensin system versus other first-line antihypertensive drug classes for hypertension") became the concept
  "first-line drugs inhibiting", and 609 of 640 records were judged off topic; CD012347 also selected none. These
  are reported as results; the parser was not changed after seeing test reviews.
- **The test set is harder than the development set** (development, same pipeline: pool 69%, selected 42%). The
  rules were developed on the development reviews, so some fitting to them is expected; this is why the split was
  made.
- **Screening keeps most included studies it sees** (median 100%, mean 87%) but CD010778 (17%) and CD011380 (0%)
  are exceptions to examine.
- **Other methods** found 15 included studies no database search returned.

## Deviations from the protocol (to report)

1. The review-version date rule was amended before any run (protocol amendment, 18 Nov 2026).
2. Selection protocol 2.5's ranking change was measured offline on the development set and not confirmed with the
   model before these runs.
3. One review (CD006715) was re-run after its first run lost two screening batches to dropped connections (the
   batch runner fails runs with more than 5% unscreened records). In the final runs, CD006715 and CD009642 each
   have 12 records (one batch) that the model did not screen, under the 5% threshold.
4. The development set's 5d runs used the paid endpoint of the same model for 9 of 11 reviews; the test runs used
   the free endpoint.

## Still to do

- Standard-mode runs of the 22 test reviews (`--mode standard --out runs/6_test_standard`), then re-run
  `python -m eval.session6 --prisma runs/6_test_prisma --standard runs/6_test_standard --set test --name 6_test`.
- Look at the two parse failures and the low-sensitivity reviews for the paper's error analysis (no changes to the
  measured pipeline).
