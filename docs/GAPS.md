# How CiteLume finds knowledge gaps ("Open questions")

This describes the gap module as it works today (the *Open questions* view and the gap matrix in the exports). It
is exploratory: it reports what the included studies show or say is unresolved, not what is missing from the whole
literature. The limits section says why, and what is planned instead.

Code: `backend/services/gap_derivation.py` (gaps read from the studies) and `backend/services/gap_matrix_engine.py`
(the grid, the checks and the assembled matrix); the model's part is `build_knowledge_gap_matrix` in
`backend/services/synthesis_engine.py`. Endpoint: `POST /api/corpus/{id}/gaps` returns the gap matrix of a stored
corpus's analysis.

## What it starts from

- The **included studies** of a corpus (20 for a quick answer, up to 60 or more for a guided review), with what the
  extraction step read from each: design, population or system, results and stated limitations; full text where
  it was available.
- The **evidence table** and the **synthesis claims**, which say for each claim which studies support it and which
  contradict it.
- The **candidate pool**: the records the searches found but the selection left out, used only to check gaps.

## What each mode reports

- **Quick answer (20 studies): only what holds at any number of studies.** Its studies were chosen to answer the
  question, not to map what has been studied, so they cannot show that something is missing. It reports the open
  questions the authors of these studies state (3 below) and disagreements among them (2 below), headed "Open
  questions stated by the authors, and where the studies disagree". No grid, design-profile gaps, sub-question
  coverage or research opportunities; the model is not asked for a grid.
- **Deep review and guided review: the full analysis below**, labelled exploratory.

## Where the gaps come from

Up to six gaps are reported. Each one names the studies it rests on and the kind of evidence behind it.

1. **Design profile** (rules, `derive_design_gaps`). It looks at which kinds of study the included set contains and
   lacks, for example an intervention question with no randomised trial, only cross-sectional data for a causal
   question, only cell or animal models, or prediction models with no external validation.
2. **Disagreements between studies** (from the synthesis). When the synthesis finds studies that support a claim
   and others that contradict it, the disagreement is a gap: what would resolve it is a head-to-head comparison or
   a meta-analysis that explains it.
3. **Open questions the authors state** (rules, `find_author_open_questions`). Sentences in which authors say
   something is still unknown or needs further study ("further prospective studies are needed to ...", "... remains
   unclear"), read from the abstract and, when full text is available, the discussion, limitations and conclusion.
   They are ranked by how closely they relate to the question, and near-duplicates across studies are merged,
   keeping all citations.
4. **Thin cells of a question-specific grid.** With a model, the model proposes a 4 by 4 grid for the question
   (rows: methods or interventions; columns: outcomes, populations, settings or contexts) and assigns studies to
   cells; cells with few or no studies become candidate gaps. Without a model, the grid uses fixed axes for the
   field, and studies are placed by word matching. Each cell is classed by how much evidence it holds.

Nothing uses fixed per-field wording that could appear unchanged for an unrelated question.

## Why the evidence cannot answer it

Every gap names one of the four reasons of Robinson et al. (2011), *Frameworks for determining research gaps during
systematic reviews* (AHRQ Methods Future Research Needs Report 2), shown on the page and in the exports:

| Reason | Used for |
|---|---|
| Too little or imprecise evidence | no or one study in a grid cell; evidence absent; open mechanistic or theoretical questions |
| Evidence at risk of bias | methodological limitations; an intervention question with no randomised trial |
| Inconsistent results | studies that disagree; results on incompatible measures (consistency unknown) |
| Not the population, setting or outcome asked about | only cell or animal models; no external validation; a single site or population |

The reason comes from the gap's kind by fixed rules (`ROBINSON_BY_CATEGORY` and `ROBINSON_BY_DESIGN_KIND` in
`gap_matrix_engine.py`). Risk-of-bias judgements, which run alongside the gap step, are not yet used for it.

## How each gap is checked

- **Verification against the candidate pool** (`execute_targeted_gap_verification`). The gap's own search terms are
  compared with every record that was found but not included, to see whether any of them already addresses it; the
  three best matches are shown. (Before 24 November 2026 only the first ten such records, in the order the searches
  returned them, were read.) This does not run a new database search: it is bounded by what the searches returned.
- **Adversarial check** (`run_adversarial_gap_check`). Each gap is challenged by fixed questions, for example
  whether an included study already answers it, or whether a domain-specific result (such as an established proof)
  contradicts it. A gap that fails is marked as resolved by the retrieved evidence.
- **Priority** (`evaluate_gap_priority`): a qualitative category, with no invented percentages. Unresolved
  disagreements, insufficient validation and open theoretical questions rank higher; a gap whose verification was
  inconclusive (for example because only abstracts were available) is marked "insufficient information to
  prioritize".
- **Validation** (`validate_knowledge_gap_matrix`): every gap must cite studies that exist in the corpus; gaps and
  cells without support are flagged.

Every gap carries a scope note ("Reflects coverage within the selected study set; absence in selected papers does
not prove universal field-wide absence") and says whether it rests on full text or abstracts only.

## Limits

- **A gap drawn from 20 to 150 studies is not a gap in the literature** (which is why a quick answer reports none). "No study here did X" is often a sampling
  artefact: the selection was built to answer the question, not to map everything done on it.
- **Verification does not search.** Checking a gap against the leftover candidate records is weaker than asking
  the databases whether the gap is real.
- **The model's grid varies** between runs and models; without a model the grid is generic.
- **Reading author-stated open questions is by sentence patterns**; a remark can be missed or misread.

For these reasons the gap view is labelled exploratory, and the publication plan (`docs/PUBLICATION_PLAN.md`)
describes gaps as exploratory in the first paper.

## What is planned instead

Sessions 11 and 12 of the publication plan rebuild coverage gaps on counts: a grid of population by intervention
or exposure by outcome by design, with one count query per cell against PubMed, Europe PMC and OpenAlex, reported
with its query string and date and no model in that step; then a validation against published evidence gap maps.
Author-stated open questions and disagreements would stay, labelled as what the included studies say.
