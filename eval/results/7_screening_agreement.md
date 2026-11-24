# Session 7, E2: screening agreement on the same records (test set)

Same records (a fixed sample of up to 96 screened records per review, every screened included study first), same question and criteria, frozen screening prompt. 'stored' is the decision in the Session 6 run.

| A | B | Records | Agreement | Cohen's kappa |
|---|---|---|---|---|
| stored | nemotron-r1 | 1638 | 91% | 0.82 |
| stored | nemotron-r2 | 1638 | 91% | 0.82 |
| stored | nemotron-r3 | 1638 | 90% | 0.80 |
| stored | qwen3.5-9b-local | 192 | 77% | 0.52 |
| nemotron-r1 | nemotron-r2 | 1638 | 96% | 0.92 |
| nemotron-r1 | nemotron-r3 | 1638 | 96% | 0.91 |
| nemotron-r1 | qwen3.5-9b-local | 192 | 75% | 0.49 |
| nemotron-r2 | nemotron-r3 | 1638 | 97% | 0.94 |
| nemotron-r2 | qwen3.5-9b-local | 192 | 76% | 0.51 |
| nemotron-r3 | qwen3.5-9b-local | 192 | 74% | 0.48 |

| Screened by | Included studies screened | Sensitivity |
|---|---|---|
| stored | 117 | 92% |
| nemotron-r1 | 117 | 93% |
| nemotron-r2 | 117 | 93% |
| nemotron-r3 | 117 | 93% |
| qwen3.5-9b-local | 11 | 100% |
