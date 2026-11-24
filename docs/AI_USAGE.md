# Use of AI in building CiteLume

DRAFT, prepared with AI assistance for the owner to check and correct before release. Replace anything that does
not match your own account.

## Who did what

CiteLume was conceived, directed and owned by its author (GitHub: mbsai). Most of the code, documentation, analysis
scripts and the text of the evaluation reports was written by an AI coding assistant (Anthropic's Claude, used
through Claude Code) working in the repository under the author's direction, session by session, from a written
plan (docs/PUBLICATION_PLAN.md).

**Decided by the author** (each recorded in the plan or the evaluation protocol with its date):
- the purpose of the tool and the claim of the paper (protocol-first, auditable, repeatable study selection);
- the architecture choices put to them (for example the shared corpus layer, the guided review, study limits by
  review type, imports from other databases, risk of bias from full text only);
- the evaluation design and its pre-registration, the benchmark and keeping it at 33 reviews, freezing search and
  selection before using the test results, and every change to the evaluation design (for example measuring claim
  accuracy instead of comparing gates on and off);
- which runs to pay for, which models to use, and what to report as exploratory (risk of bias);
- whether and when to publish, push and deploy.

**Done by the AI assistant, under that direction**: writing and changing code and tests; running searches,
evaluation runs and analyses; drafting documentation, the protocol text and findings; proposing options and
recommending one.

**Done by people, not by AI**: the annotation of claims and merged records (Sessions 8 and 9), and the final review
of everything submitted.

## How the work was checked

- Every change is covered by an automated test suite (270 tests at the time of writing) that runs offline and in CI.
- Selection is deterministic and replayable: every evaluation corpus reproduces its result from its stored records.
- The evaluation protocol was committed before any benchmark data were loaded; test-set results were produced only
  after the measured code was frozen (git tag eval-freeze-2.5, checked by a test).
- Numbers in the reports are produced by scripts in eval/ from stored runs, not written by hand.

## The tool's own use of AI

Inside CiteLume, a language model (chosen by the user; any provider, or none) may plan searches, screen titles and
abstracts against the protocol's criteria, extract study details, suggest risk-of-bias judgements and write the
synthesis. Its role and limits are described in docs/RANKING.md and in docs/RAISE.md.
