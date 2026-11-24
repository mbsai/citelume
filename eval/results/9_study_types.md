# Session 9: study-type classifier against PubMed publication types

26114 PubMed records with a usable publication type (from 7 evaluation runs' raw search output, one per PubMed ID); 2819 of them also returned by Semantic Scholar. Gold: the indexers' publication types. The classifier is frozen (selection protocol 2.5).

## Wording only (publication types removed)

Accuracy 0.67 on 26114 records.

| Class | Records | Precision | Recall | F1 |
|---|---|---|---|---|
| correction | 19 | 0.33 | 0.16 | 0.21 |
| protocol | 15 | 0.14 | 0.80 | 0.24 |
| synthesis | 1275 | 0.82 | 0.58 | 0.68 |
| rct | 2576 | 0.88 | 0.62 | 0.72 |
| case report | 2032 | 0.84 | 0.17 | 0.28 |
| commentary | 792 |  | 0.00 |  |
| review | 5629 | 0.74 | 0.32 | 0.45 |
| other primary | 13776 | 0.64 | 0.95 | 0.77 |

Confusion (rows: gold, columns: predicted):

| Gold \ predicted | correction | protocol | synthesis | rct | case report | commentary | review | other primary |
|---|---|---|---|---|---|---|---|---|
| correction | 3 |  | 1 | 6 |  |  | 1 | 8 |
| protocol |  | 12 |  | 2 |  |  |  | 1 |
| synthesis | 4 |  | 740 | 16 |  |  | 201 | 314 |
| rct |  | 55 | 7 | 1585 | 1 |  | 6 | 922 |
| case report | 1 |  | 2 | 2 | 347 |  | 62 | 1618 |
| commentary |  |  | 20 | 13 | 1 |  | 17 | 741 |
| review |  |  | 49 | 17 | 3 |  | 1816 | 3744 |
| other primary | 1 | 17 | 80 | 161 | 63 |  | 339 | 13115 |

## As Semantic Scholar returned the record (its own types)

Accuracy 0.78 on 2819 records.

| Class | Records | Precision | Recall | F1 |
|---|---|---|---|---|
| correction | 4 | 0.50 | 0.25 | 0.33 |
| protocol | 3 | 0.18 | 1.00 | 0.30 |
| synthesis | 349 | 0.91 | 0.80 | 0.85 |
| rct | 343 | 0.89 | 0.46 | 0.61 |
| case report | 114 | 0.91 | 0.76 | 0.83 |
| commentary | 92 | 0.97 | 0.71 | 0.82 |
| review | 648 | 0.67 | 0.93 | 0.78 |
| other primary | 1266 | 0.80 | 0.79 | 0.79 |

Confusion (rows: gold, columns: predicted):

| Gold \ predicted | correction | protocol | synthesis | rct | case report | commentary | review | other primary |
|---|---|---|---|---|---|---|---|---|
| correction | 1 |  |  | 1 |  |  |  | 2 |
| protocol |  | 3 |  |  |  |  |  |  |
| synthesis | 1 |  | 278 |  |  |  | 59 | 11 |
| rct |  | 12 |  | 159 |  | 2 | 7 | 163 |
| case report |  |  | 1 |  | 87 |  | 2 | 24 |
| commentary |  |  | 3 | 2 |  | 65 | 4 | 18 |
| review |  |  | 7 |  | 1 |  | 601 | 39 |
| other primary |  | 2 | 15 | 17 | 8 |  | 225 | 999 |
