# Session 8 findings: verification gates and claim accuracy (awaiting annotation)

Runs, 24 Nov 2026, Nemotron 3 Ultra (free endpoint), no cost: the analysis of the 22 frozen Standard test corpora
re-run (`runs/8_gates`, `python -m eval.gates_runs`), and the 11 development questions run in Standard mode
(`runs/8_dev_standard`). Each result keeps the narrative as the model wrote it before the section quality gate
(`ungated_narrative`) as well as the one shown.

## The gates rarely act (`8_gate_activity.md`)

Over 33 model-written analyses and 881 cited sentences shown to readers:
- the section quality gate changed no narrative (no figure removed, no narrative replaced);
- the claim gate excluded none of 117 structured claims;
- one narrative needed a corrective retry during generation;
- ten narratives carried a soft quality note and were kept, nine of them "evidence cluster N has no cited studies".
Not counted here: the check of figures in the per-study extraction, which retries an extraction inside the model
call and is not recorded per study.

## Design change (decided 24 Nov 2026, before any annotation)

The plan was to annotate claims with the gates on and off. Since the gates changed nothing in these runs, the two
narratives are the same text, and annotating both would cost hours to confirm what the counts show. The annotation
measures instead **whether the claims shown to readers are supported by the studies they cite**: 200 cited sentences
drawn at random (seed 20261118) from the 881 shown in the 33 narratives (every question represented, 2 to 11 claims
each), and 50 of them for a second annotator. The tooling is unchanged and would report the two narratives
separately if a gate had changed one.

## To do (the owner's step)

1. Annotate `eval/annotation/sheet.csv` (200 claims, about 4-6 hours, can be split) following
   `eval/annotation/GUIDE.md`; a second annotator does `sheet_second.csv` (50 claims, 1-2 hours), independently.
2. `python -m eval.annotation_sheet score` writes `8_gates.md`: the share of shown claims judged not supported, and
   partly or not supported, with 95% bootstrap intervals over questions, for all questions and for test and
   development separately; and Cohen's kappa between annotators. `judgements.csv` keeps the judgements without
   the abstract text.
