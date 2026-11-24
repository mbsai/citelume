# Annotating claims (Session 8)

You will judge whether each claim is supported by the studies it cites. The sheet does not say which version of the
narrative a claim came from; please do not open `key.json` until you have finished.

## The sheet

`sheet.csv` (first annotator) and `sheet_second.csv` (second annotator, a subset). One row per claim:

- **question**: the review question the narrative answers.
- **claim**: one sentence of the narrative, with its citation numbers.
- **cited_studies_and_evidence**: for each cited study, its title and year, then the two sentences of its abstract
  that share the most words with the claim (in the abstract's own order). This is a reading aid, not the whole
  evidence: if it is not enough, look the study up by its title.
- **judgement**: fill in one of the four words below, exactly.
- **notes**: optional; say why when you choose *partly* or *not supported*.

## Judgements

- **supported**: everything the claim says about the cited studies is stated or directly shown by them: the
  direction of the effect, the population, the comparison, any figure, and how strongly it is put.
- **partly**: the claim is broadly right but overstates or loses something: a figure is rounded or slightly off, a
  subgroup is generalised to everyone, a trend is called a significant effect, or only some of the cited studies
  support it.
- **not supported**: the cited studies do not say this, say the opposite, or are about something else; or a figure
  does not appear in them.
- **cannot tell**: the abstract does not settle it and the full text is not to hand. Use this sparingly; it is left
  out of the error rates.

Judge each claim against its cited studies only, not against what you know of the field. A true statement that its
citations do not support is *not supported*.

## Time

About 1 to 2 minutes a claim: about 200 claims in 4 to 6 hours, which can be split over several sittings. The second
annotator's 50 claims take about 1 to 2 hours. Work independently; do not compare judgements before both sheets are
done.

## When you are done

Save the sheets in place (CSV, same columns) and run `python -m eval.annotation_sheet score`.
