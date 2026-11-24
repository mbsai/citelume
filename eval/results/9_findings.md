# Session 9 findings: component checks (in progress)

## Study-type classifier (`9_study_types.md`; `python -m eval.study_type_check runs/5b_dev runs/5c_dev
## runs/5c_dev_full runs/5d_dev runs/6_test_prisma runs/7_search_r1 runs/7_search_r2`)

Gold labels: the publication types PubMed indexers assigned to 26,114 records (one per PubMed ID) from the raw
search output of the evaluation runs, mapped to eight classes. The classifier (frozen, protocol 2.5) reads a
record's publication types first, so it was tested where it cannot lean on PubMed's labels:

- **Wording only** (types removed, as for most arXiv, bioRxiv and some Semantic Scholar records): accuracy 0.67.
  Trials: precision 0.88, recall 0.62. Systematic reviews and meta-analyses: precision 0.82, recall 0.58. Case
  reports (recall 0.17) and narrative reviews (0.32) are mostly called "other primary research"; editorials and
  letters are never recognised from wording (by design, that rule reads publication types only).
- **With Semantic Scholar's own types** (2,819 papers both sources returned): accuracy 0.78. Semantic Scholar labels
  225 papers that PubMed indexes as primary research "Review" (review precision 0.67): the error analysis's "case
  series typed as reviews", confirmed in bulk.
- **How much it matters:** 69% of the records in the test corpora, and 80% of the included studies, carry PubMed's
  own types, which the classifier uses first; the figures above apply to the rest (31% and 20%).

## Duplicate merging (awaiting a hand check)

`python -m eval.merge_check build runs/6_test_prisma --n 100` replays the frozen merge on the test runs' raw
output: 1,351 merges matched on the title alone and 99 merges of records that share an identifier but differ in
title (merges of identical records are not counted). 50 of each are in `eval/annotation/merge_sheet.csv` (kept
local); fill "same_study" with yes, no or unsure (about 30 minutes), then `python -m eval.merge_check score`.

## Risk of bias: reported as exploratory (decided 24 Nov 2026)

The CLEF benchmark does not include the reviews' risk-of-bias tables, and CiteLume assesses risk of bias only for
studies with open full text (a minority), so a comparison with Cochrane's judgements would rest on few trials and
need them extracted by hand. Decision: no agreement figure is claimed. The paper describes the risk-of-bias module
as exploratory: every judgement is a suggestion for the reviewer to confirm, each rests on a quote checked against
the study's text, randomised trials follow the coded RoB 2 algorithm (checked branch by branch against the official
cribsheet), and abstract-only studies are marked "Not assessed". A comparison with published judgements is listed
as future work.
