# Session 5d findings (development set only)

Run `runs/5d_dev`, code 57b5935 (two reviews) and 702073c (nine reviews), Deep Research, systematic-style review
type, PubMed core search 500 deep, citation chasing and review reference lists on. Model: Nemotron 3 Ultra
(`nvidia/nemotron-3-ultra-550b-a55b`); the first two reviews ran on its free endpoint before the free daily quota
ran out, the other nine on the paid endpoint (same model; cost 1.63 USD). The 702073c change affects only records
the model did not screen; every record in these runs was screened by the model.

| | 5b baseline | 5c full | 5d |
|---|---|---|---|
| Found by the searches | 39 (28%) | 78 (55%) | 87 (62%) |
| Eligible | 9 (6%) | 25 (18%) | 77 (55%) |
| Screened in | 8 (6%) | 24 (17%) | 57 (40%) |
| Selected | 8 (6%) | 24 (17%) | 55 (39%) |
| Studies selected in all | (see 5b runs) | 399 | 883 |
| Precision of the selection | | 6% | 6% |

Of 141 included studies. Table: `5d_loss_by_stage.md`.

## What the changes did

- **Eligibility no longer loses most of what is found:** losses at eligibility fell from 53 to 10. Reading PubMed's
  MeSH indexing, screening partial matches instead of excluding them, and keeping title-only records together
  admitted 77 of the 87 included studies found.
- **Precision held:** the selection grew from 399 to 883 studies and the share that the reviews included stayed at
  6%. The model's screening absorbed the extra records: it excluded most partial matches and kept the included
  studies (only 3 screened out).
- **Recall at the selection more than doubled:** 24 to 55 of 141.

## Where studies are lost now

| Stage | Lost | Share |
|---|---|---|
| Search | 54 | 38% |
| Eligibility | 10 | 7% |
| Screening and limit | 22 | 16% |

- **The screening cap is the new bottleneck after search.** 17 included studies were eligible but never screened:
  the model screens at most 300 records per review (`MAX_SCREENED`, or twice the study limit), in score order, and
  these ranked below that. Only 29 of the 77 eligible included studies ranked in the top 60. Better ranking, or a
  higher cap at a model cost, would recover them.
- **Model cost rose** with the number of records screened: about 0.18 USD per review on the paid endpoint, and the
  free daily quota no longer covers a full development run.
- **Search** still loses 54; 25 of them are matched by CiteLume's queries beyond the first 500 records.

## Decisions for the plan

1. Session 6 can start: the pipeline is fixed for the test set (selection protocol 2.4, depth 500 for evaluation).
2. Ranking (where included studies fall in the eligible list) and the screening cap are worth a small session
   before the paper's final runs; they are measured on the development set only.
3. Report the model cost per review in the paper.

Development-set findings only; they decide what to build, not what is reported as performance.
