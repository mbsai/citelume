# CiteLume against RAISE recommendations for AI tool development teams

Source: Responsible use of AI in evidence SynthEsis (RAISE 2026) 1: Living recommendations for practice, section 2
"AI tool development teams", recommendations 2.1 to 2.27. Version 4, 23 September 2026 (draft for consultation and
revision). Cochrane, Campbell Collaboration, JBI and Collaboration for Environmental Evidence with others; OSF
project https://osf.io/fwaud/, DOI 10.17605/OSF.IO/FWAUD. Recommendations are paraphrased; see the source for the
full text. Assessed 24 November 2026 against the repository at that date; revisit when RAISE is revised.

CiteLume does not train or fine-tune models: it calls general-purpose models (any provider, or none) for planning
searches, screening, extraction and writing, and its selection is rule-based. Recommendations about training data
are read as applying to the evaluation data.

Status: **Met**, **Partly**, **Not yet** (planned or needs the owner), **Not applicable**.

| # | Recommendation (paraphrased) | Status | CiteLume |
|---|---|---|---|
| 2.1 | Understand the principles of rigorous evidence synthesis and design AI that supports them | Met | Protocol-first design: a review protocol drives searches and selection; selection is rule-based, stored and replayable; PRISMA 2020 reporting; the model's role is limited and stated (docs/RANKING.md). |
| 2.2 | Use suitable datasets; no overlap between training, testing and validation | Met | No model is trained. The evaluation used a seeded development / test split fixed before any run; changes were made on development reviews only, and search and selection were frozen before the test results were used (eval/PROTOCOL.md, FREEZE.json). |
| 2.3 | Minimise bias in data, parameters and prompts | Partly | Prompts are fixed and versioned; screening criteria come from the protocol, not the model. No formal assessment of bias in the prompts or the models used. |
| 2.4 | Evaluate with appropriate methods, metrics and protocols | Met | Pre-registered protocol (eval/PROTOCOL.md): recall at each stage, screening sensitivity and precision, baselines, bootstrap intervals; amendments dated and explained. |
| 2.5 | Report evaluations completely and publicly, including null results, preferably peer-reviewed | Partly | All results, including low recall and failures, are in eval/results/ (error analysis, deviations). Peer-reviewed publication in preparation. |
| 2.6 | Share datasets and code openly (FAIR, FAIR4RS); state access and licence | Partly | Code under MIT licence; CITATION.cff; an evaluation archive with checksums is built (eval/build_archive.py). Not yet deposited with a DOI. |
| 2.7 | Follow national and international rules (e.g. EU AI Act) on technical information about models built or fine-tuned | Not applicable | No model is built or fine-tuned. The models called are named in every run record. Not formally assessed against the EU AI Act. |
| 2.8 | Report name, version and purpose; how it has changed; planned updates | Partly | Every corpus records the selection protocol version, prompt versions, model and code commit. No public changelog or roadmap yet. |
| 2.9 | Report how the tool works | Met | Open source; docs/RANKING.md (search and selection, generated from the code), docs/GAPS.md, docs/FRAMEWORKS.md. |
| 2.10 | Say which data are used to train, test and validate, and where they are stored | Met | No training data. Evaluation data: CLEF TAR 2019 intervention reviews (eval/benchmark/, built by a script from public sources), corpora and results in the evaluation archive. |
| 2.11 | Give users clear guidance and help them take accountability | Partly | In-page explanations: suggested judgements "for the reviewer to confirm", screening decisions the user can override with a reason, sources shown for claims. No complete user guide. |
| 2.12 | Report strengths and limitations, and when not to use it | Partly | Limits stated in docs/RANKING.md and docs/GAPS.md; the evaluation states that the quick answer is not a review search (4% of a review's included studies) and that recall is limited mainly by search. To add to the README and the page. |
| 2.13 | Report equity issues and biases, including limits to generalisability | Partly | Stated: citation counts favour older and English-language work; the evaluation covers Cochrane intervention reviews only. No systematic equity assessment. |
| 2.14 | Declare funding and interests | Not yet | To declare in the paper and the README (owner). |
| 2.15 | Be honest about all aspects of use and development | Met | Protocol deviations, failed runs, design changes and costs are recorded with dates in eval/results/. |
| 2.16 | Do not market or recommend the tool for evidence synthesis without robust evaluation | Met | The paper reports measured recall (pool 54%, selected 25% of review-included studies on held-out reviews) and does not claim the tool replaces a systematic search. |
| 2.17 | Label tools without robust evaluation as testing or evaluation versions | Partly | The public demo should be labelled as a research version with the evaluation's limits. To add. |
| 2.18 | Do not promote tools that amplify publication bias or give biased, misleading summaries | Partly | Citations and figures are checked against the sources; studies without full text are not judged for risk of bias. Claim accuracy is being measured by human annotation (Session 8). |
| 2.19 | Involve interest holders in development and standards | Not yet | No formal engagement; a usability study with reviewers is optional future work. |
| 2.20 | Assess and mitigate systemic risks; track incidents; secure the system | Partly | Demo limits and usage caps; no keys or query text in logs; keys never written to outputs. No incident log. |
| 2.21 | Keep humans in control (human oversight) | Met | The guided review: the user confirms the protocol; screening and risk-of-bias decisions can be overridden with a recorded reason; the corpus records every decision. |
| 2.22 | Responsible handover; do not require users to give up rights to their data | Met | User data are not reused: the demo keeps corpora in memory for 24 hours and logs no query text; local installs keep everything local; imports and PDFs are disabled on the demo. |
| 2.23 | Provide accessible appeals and feedback pathways | Partly | GitHub issues. No feedback route shown in the app. |
| 2.24 | Monitor deployed models for changes in performance | Partly | A golden test and CI guard selection; the frozen evaluation can be re-run on stored corpora. No monitoring of the models' performance over time. |
| 2.25 | Do not treat the tool as an author; avoid anthropomorphising it | Met | Presented as a tool; no human-like persona; outputs are labelled as suggestions where judgement is involved. |
| 2.26 | Follow ethical, legal and regulatory standards (copyright, privacy) and tell users | Partly | Only open-access full text is fetched; the evaluation archive notes that abstracts remain the publishers' copyright; privacy as in 2.22. No formal legal review. |
| 2.27 | Minimise environmental and societal impact; use AI only where suited | Partly | Search, selection and checks run without a model; small or free models are supported; model cost per review is reported. Energy use not measured. |

## Gaps to close before submission

1. Declare funding and interests (2.14).
2. Label the public demo as a research version and state its limits on the page and in the README (2.12, 2.17).
3. Deposit the code and the evaluation archive with DOIs (2.6).
4. Add a changelog and a feedback route (2.8, 2.23).
