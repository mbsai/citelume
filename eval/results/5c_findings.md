# Session 5c findings (development set only)

Three runs of the 11 development reviews, same settings as 5b (Deep Research, systematic-style review type,
Nemotron, citation chasing on, each review's title as the question, its year as the upper date limit):

| Run | Code | What changed | Found | Eligible | Selected |
|---|---|---|---|---|---|
| 5b baseline (`runs/5b_dev`) | ce2bc39 | - | 39 (28%) | 9 (6%) | 8 (6%) |
| MeSH (`runs/5c_dev`) | ed30e47 | MeSH headings and pharmacological actions in the PubMed queries; question parsing (dashes, slashes, purpose words, class labels, bracketed abbreviations) | 51 (36%) | 12 (9%) | 12 (9%) |
| Full (`runs/5c_dev_full`) | a5a27e9 | the above, plus PubMed core search 500 deep (`--pubmed-depth 500`) and the reference lists of excluded reviews | 78 (55%) | 25 (18%) | 24 (17%) |

Of 141 included studies. Tables: `5b_loss_by_stage.md`, `5c_mesh_loss_by_stage.md`, `5c_full_loss_by_stage.md`.

## Losses by stage, full run

| Stage | Lost | Share |
|---|---|---|
| Search | 63 | 45% (was 72%) |
| Duplicate merging | 0 | 0% |
| Eligibility | 53 | 38% (was 21%) |
| Screening and limit | 1 | 1% |

## What the changes did

- **MeSH headings** carried the drug-class question: *Antidepressants for insomnia* went from 0 to 10 of 20
  studies found (`"antidepressive agents"[Pharmacological Action]`).
- **The slash fix** carried the prostatitis review: 4 to 17 of 29 found.
- **MeSH broadened the queries, so depth started to matter.** After the MeSH run, 43 of the 90 missed studies were
  matched by CiteLume's own queries without a result limit; the core search had fetched only 100 of up to 42,810
  matches. At 500 deep, 28 of the remaining 63 are still matched but not fetched.
- **Other methods** found 6 included studies no database search returned: 3 by citation chasing, 3 from the
  reference lists of excluded reviews (which added up to 150 records in the reviews where earlier reviews existed).
- **"Retrieved N of M"** is now logged for every PubMed search and reported in the Methods (for example, core
  search: retrieved 500 of 12,642 matching records).
- One review lost ground in the MeSH run (fluoride toothpastes, 1 to 0 found) and recovered at depth 500.

## What is left

1. **Eligibility now loses almost as much as search** (53 of the 78 found): "not about the topic" 34, "does not
   address every part of the question" 12, no abstract 7. More records found means more records judged by word
   matching, which rejects trials that name things differently. This is Session 5d.
2. **Search depth is still a cap:** 28 missed studies are matched by the queries beyond the first 500. Deeper
   paging costs time and model screening; it is a setting (`pubmed_depth`), not the default. The default depth
   (100 for the systematic-style core search) and the golden test are unchanged.
3. **Query sensitivity:** 35 missed studies are matched by no CiteLume query. Examples remain where the question's
   concepts are narrower than the review's searches (a review title is not its search strategy).

## Decision for the plan

5d next: concept matching on MeSH headings and synonyms in eligibility (records already carry PubMed's MeSH
indexing once parsed), outcome phrases not required, title-only records judged on their titles. Then consider
whether systematic-style reviews on local installs should default to a deeper PubMed search (for example 500),
reported as a setting in the Methods.

Development-set findings only; they decide what to build, not what is reported as performance.
