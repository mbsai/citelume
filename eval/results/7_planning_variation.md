# Session 7, E2: variation of the model's search plan (test set)

22 questions, 66 new plans. Overlap is Jaccard (1 = identical) of the concept names (case-folded) and of the PubMed search strings, against the plan stored in the Session 6 run.

- Concept names, new plan vs stored: median overlap 1.00
- Concept names, between new plans: median overlap 1.00
- Core PubMed search identical to the stored one (MeSH alternatives removed): 14 of 66 plans
- All PubMed strings (core, reviews and aspect searches), new plan vs stored: median overlap 0.00
- Questions whose every new plan named exactly the stored concepts: 14 of 22

The concept names say what the model takes the question to be about; the PubMed strings also carry the synonyms it adds and the aspects it chooses, which vary more.

| Review | Plans | Concepts vs stored | Concepts between repeats | Core search identical | All PubMed strings vs stored |
|---|---|---|---|---|---|
| CD000996 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD001261 | 3 | 1.00 | 1.00 | 100% | 0.34 |
| CD004414 | 3 | 0.61 | 0.78 | 0% | 0.00 |
| CD005139 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD005253 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD006468 | 3 | 0.33 | 0.33 | 0% | 0.00 |
| CD006715 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD007867 | 3 | 1.00 | 1.00 | 33% | 0.05 |
| CD008170 | 3 | 1.00 | 1.00 | 100% | 0.17 |
| CD008201 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD009069 | 3 | 1.00 | 1.00 | 100% | 0.35 |
| CD009642 | 3 | 0.83 | 0.67 | 0% | 0.00 |
| CD010239 | 3 | 0.50 | 0.30 | 0% | 0.00 |
| CD010558 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD010778 | 3 | 0.47 | 0.47 | 33% | 0.05 |
| CD011380 | 3 | 0.50 | 1.00 | 0% | 0.00 |
| CD011436 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD011768 | 3 | 0.20 | 1.00 | 0% | 0.00 |
| CD012069 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD012164 | 3 | 1.00 | 1.00 | 0% | 0.00 |
| CD012347 | 3 | 1.00 | 1.00 | 33% | 0.11 |
| CD012930 | 3 | 1.00 | 1.00 | 67% | 0.30 |
