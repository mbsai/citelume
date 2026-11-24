# Manuscript: CiteLume for the Journal of Biomedical Informatics

Draft of paper 1 (Session 14 of docs/PUBLICATION_PLAN.md), started 24 November 2026. Elsevier `elsarticle` template.

    latexmk -pdf main.tex                      # the manuscript -> main.pdf
    (cd supplement && latexmk -pdf prisma_s.tex)   # S1, the PRISMA-S checklist

- `sections/`: abstract, introduction, methods (design and evaluation), results, discussion, statements.
- `figures/`: Fig. 2 and Fig. 3 are copies of `eval/results/6_test.png` and `7_sensitivity_test.png`; re-copy them if
  those are regenerated. Fig. 1 (the pipeline) is drawn in TikZ in `sections/methods.tex`.
- `references.bib`: entries marked `note = {VERIFY ...}` were written from memory or from a URL alone; check each
  against its source before submission.

## Open items

Red `[TODO: ...]` marks are the owner's (name, affiliation, ORCID, DOIs, funding, interests, CRediT, the AI-use
statement, the journal's limits). Orange `[PENDING: ...]` marks are results not yet available: claim accuracy
(Session 8 annotation) and the duplicate-merge check (Session 9). List them with:

    grep -n "TODO\|PENDING" main.tex sections/*.tex supplement/*.tex

Every number in the text comes from `eval/results/` (6_findings.md, 6_test.md, 6_error_analysis.md,
6_test_loss_by_stage.md, 7_findings.md, 8_findings.md, 9_findings.md); change them there first and then here.
Not yet done: the graphical abstract, the reference check, and confirming JBI's format (structured abstract and its
limit, highlights, article type) from the Guide for Authors.
