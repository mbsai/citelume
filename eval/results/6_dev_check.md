# Session 6, recall (development set): 6_dev_check

11 reviews analysed. Measures and summaries as pre-registered in eval/PROTOCOL.md section 5; 95% CIs are bootstrap intervals of the mean (10,000 resamples of reviews, seed 20261118).

| Measure | Reviews | Median | Mean (95% CI) | Mean, reviews with <= 60 included (95% CI) |
|---|---|---|---|---|
| pool recall | 11 | 77% | 69% (50%-86%) | 69% (50%-86%) |
| pool recall, databases only | 11 | 50% | 58% (40%-75%) | 58% (40%-75%) |
| top 20 | 11 | 14% | 16% (8%-25%) | 16% (8%-25%) |
| top 60 | 11 | 20% | 28% (16%-41%) | 28% (16%-41%) |
| selected | 11 | 38% | 42% (24%-61%) | 42% (24%-61%) |
| screening sensitivity | 10 | 100% | 95% (90%-100%) | 95% (90%-100%) |
| screening precision | 11 | 9% | 9% (5%-13%) | 9% (5%-13%) |
| BM25 top 20 | 11 | 0% | 4% (1%-9%) | 4% (1%-9%) |
| BM25 top 60 | 11 | 11% | 14% (6%-23%) | 14% (6%-23%) |
| Citations top 20 | 11 | 0% | 5% (0%-12%) | 5% (0%-12%) |
| Citations top 60 | 11 | 0% | 10% (1%-21%) | 10% (1%-21%) |
| Random top 20 | 11 | 0% | 2% (0%-6%) | 2% (0%-6%) |
| Random top 60 | 11 | 0% | 7% (2%-13%) | 7% (2%-13%) |

## Per review

| Review | Included | pool recall | pool recall, databases only | top 20 | top 60 | selected | screening sensitivity | screening precision | standard pool recall | standard selected |
|---|---|---|---|---|---|---|---|---|---|---|
| CD007868 | 3 | 100% | 33% | 0% | 0% | 0% |  | 0% |  |  |
| CD008018 | 13 | 77% | 46% | 23% | 23% | 38% | 100% | 12% |  |  |
| CD010038 | 9 | 11% | 11% | 0% | 11% | 11% | 100% | 1% |  |  |
| CD010355 | 6 | 100% | 83% | 33% | 67% | 100% | 100% | 21% |  |  |
| CD010526 | 8 | 100% | 100% | 0% | 12% | 88% | 88% | 5% |  |  |
| CD010753 | 20 | 50% | 50% | 5% | 30% | 50% | 100% | 9% |  |  |
| CD011571 | 6 | 100% | 100% | 50% | 67% | 67% | 80% | 3% |  |  |
| CD011977 | 37 | 51% | 49% | 14% | 16% | 14% | 83% | 22% |  |  |
| CD012223 | 5 | 80% | 80% | 20% | 20% | 20% | 100% | 6% |  |  |
| CD012455 | 5 | 20% | 20% | 20% | 20% | 20% | 100% | 11% |  |  |
| CD012551 | 29 | 66% | 62% | 14% | 38% | 52% | 100% | 10% |  |  |
