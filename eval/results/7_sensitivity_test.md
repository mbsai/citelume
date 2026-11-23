# Session 7, E3: sensitivity of the selection to its weights and caps (test set)

22 Standard and 22 PRISMA corpora; each weight or cap set to half and one and a half times its value, one at a time, in memory; compared with the frozen selection (protocol citelume-selection/2.5). Overlap is Jaccard (1 = identical); tau is Kendall's rank correlation of the studies selected both times; recall change is the mean change in the share of the reviews' included studies, in percentage points.

| Change | Standard: overlap of the 20 (median, min) | Standard: rank tau (median) | Standard: recall change | PRISMA: overlap of the first 300 (median, min) | PRISMA: recall change in the first 300 |
|---|---|---|---|---|---|
| concept-match points x0.5 | 0.95, 0.74 | 0.92 | +0.0 pts | 1.00, 0.90 | +0.0 pts |
| wording-overlap points x0.5 | 0.90, 0.74 | 0.85 | +0.1 pts | 1.00, 0.85 | +0.0 pts |
| study-type points x0.5 | 0.90, 0.67 | 0.87 | +0.1 pts | 1.00, 0.89 | +0.0 pts |
| citation-impact points x0.5 | 0.90, 0.74 | 0.91 | +0.0 pts | 1.00, 0.86 | +0.0 pts |
| recency points x0.5 | 0.90, 0.74 | 0.91 | +0.0 pts | 1.00, 0.91 | +0.0 pts |
| preprint penalty x0.5 | 1.00, 1.00 | 1.00 | +0.0 pts | 1.00, 1.00 | +0.0 pts |
| review share caps x0.5 | 0.57, 0.48 | 1.00 | +0.1 pts | 1.00, 1.00 | +0.0 pts |
| same author-year cap x0.5 | 1.00, 0.82 | 1.00 | -0.3 pts | 1.00, 1.00 | +0.0 pts |
| concept-match points x1.5 | 0.90, 0.74 | 0.95 | +0.1 pts | 1.00, 0.90 | +0.0 pts |
| wording-overlap points x1.5 | 0.90, 0.74 | 0.91 | +0.1 pts | 1.00, 0.86 | -0.0 pts |
| study-type points x1.5 | 1.00, 0.54 | 0.92 | +0.2 pts | 1.00, 0.91 | +0.0 pts |
| citation-impact points x1.5 | 0.90, 0.82 | 0.90 | +0.2 pts | 1.00, 0.89 | +0.0 pts |
| recency points x1.5 | 0.95, 0.74 | 0.91 | +0.1 pts | 1.00, 0.88 | -0.0 pts |
| preprint penalty x1.5 | 1.00, 1.00 | 1.00 | +0.0 pts | 1.00, 1.00 | +0.0 pts |
| review share caps x1.5 | 0.90, 0.54 | 1.00 | +0.0 pts | 1.00, 1.00 | +0.0 pts |
| same author-year cap x1.5 | 1.00, 1.00 | 1.00 | +0.0 pts | 1.00, 1.00 | +0.0 pts |
