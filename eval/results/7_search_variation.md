# Session 7, E2: records found on repeated searches (test set)

22 questions. Runs: 7_search_r1, 7_search_r2, stored (model-planned). The repeats used no model (rule-based planning); the stored run was planned by the model. Overlap is Jaccard of the record sets after duplicate merging (records keyed by PubMed ID, DOI or title).

| Pair | Median overlap | Lowest |
|---|---|---|
| 7_search_r1 vs 7_search_r2 | 0.63 | 0.24 |
| 7_search_r1 vs stored (model-planned) | 0.10 | 0.00 |
| 7_search_r2 vs stored (model-planned) | 0.12 | 0.00 |

| Run | Mean pool recall | Median records |
|---|---|---|
| 7_search_r1 | 34% | 738 |
| 7_search_r2 | 37% | 654 |
| stored (model-planned) | 54% | 782 |
