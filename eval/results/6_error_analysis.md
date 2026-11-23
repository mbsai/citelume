# Session 6 error analysis (test set, frozen pipeline)

Every included study of the 22 test reviews that the systematic-style run did not select, classified by where and
why it was lost. Computed by `python -m eval.error_analysis runs/6_test_prisma` (counts) from the loss-by-stage
output; the case notes below come from reading the stored corpora. Nothing in the measured pipeline was changed
(search and selection are frozen at protocol 2.5); these are findings for the paper and for work after it.

## Where the lost studies went

538 of the 644 included studies were not selected.

| Where and why | Studies | Share of those lost |
|---|---|---|
| Search: depth (a CiteLume PubMed query matches it, beyond the 500 records fetched) | 266 | 49% |
| Search: query sensitivity (no CiteLume query matches it) | 169 | 31% |
| Eligibility: the concept words did not match | 55 | 10% |
| Screening cap or study limit (eligible, never screened) | 27 | 5% |
| Eligibility: excluded as a review article by publication type | 10 | 2% |
| Screening: the model answered "no" to a criterion | 9 | 2% |
| Eligibility: other (not a research report) | 2 | 0% |

Concentration: one review (CD012069, 245 included studies) accounts for 134 of the 266 depth losses and 25 of the
27 cap losses; one (CD008170) for 81 search losses; one (CD005139) for 66 losses. Per-review counts are in
`6_error_analysis.json`.

## The failure modes, with examples

### 1. Selection reads a weaker parse of the question than the search does (eligibility, 55; part of search)

By design, selection reads a rule-based parse of the question, so that the included studies do not depend on which
model planned the searches (tested in `test_golden_search`). The model's parse is often much better:

| Review | The model's search concepts | The rule-based selection concepts |
|---|---|---|
| CD008170 First-line drugs inhibiting the renin angiotensin system versus other first-line antihypertensive drug classes | renin-angiotensin system inhibitors; first-line antihypertensive drugs | "first-line drugs inhibiting"; "renin angiotensin"; "first-line antihypertensive drug classes" |
| CD010239 Lower versus higher oxygen concentrations titrated to target oxygen saturations during resuscitation of preterm infants | oxygen therapy; resuscitation; oxygen saturation targeting | "oxygen concentrations titrated"; "target oxygen saturations"; resuscitation |
| CD011436 Ultrasound guidance for perioperative neuraxial and peripheral nerve blocks in children | ultrasound guidance; neuraxial block; peripheral nerve block | ultrasound guidance; "perioperative neuraxial" (or peripheral nerve blocks) |

Rule-parse failures seen: a verb glued to a concept ("drugs inhibiting"), a participle glued to a measure
("concentrations titrated"), an adjective glued without its noun ("perioperative neuraxial"), and "age-related"
reduced to "age" ("neovascular age macular degeneration", CD005139), which also blocked its MeSH lookup.

This is the clearest trade-off the evaluation exposes: model independence of the selection (a reproducibility
property the paper claims) costs recall when the rule parse is poor.

### 2. A class named in the question, its members named in the studies (eligibility; search)

- "Psychological therapies" (CD010558, CD012347): the trials say cognitive behavioural therapy, dialectical behaviour
  therapy, interpersonal psychotherapy, problem-solving therapy. The concept has no MeSH alternative, so word
  matching decided: 9 of 9 found studies in CD010558 and 8 of 8 in CD012347 were judged off topic.
- "Inhaled corticosteroids" (CD000996): PubMed maps "corticosteroids" to the heading Adrenal Cortex Hormones;
  fluticasone is indexed under the pharmacological action Glucocorticoids instead, and the abstracts say "inhaled
  steroids". 4 of 6 found studies excluded.
- "Anti-vascular endothelial growth factor" (CD005139): PubMed maps it to Endothelial Growth Factors (the growth
  factor, not the drugs that block it); ranibizumab and pegaptanib trials match neither the heading nor the words.
- "Ultrasound guidance for ... nerve blocks" (CD011436): trials name the specific block (rectus sheath,
  transversus abdominis plane, ilioinguinal).

### 3. The question's structure turned into required screening criteria (screening, 8 of 9)

- "Intralesional treatment **versus** wide resection" (CD010778): the comparator became the criterion "Outcome:
  wide resection"; case series of curettage alone, which the review included, were screened out (5).
- "Interventions for infantile seborrhoeic dermatitis **(including cradle cap)**" (CD011380): the bracketed synonym
  became a separate required criterion (1).
- "Continuous intravenous perioperative lidocaine **infusion**" (CD009642): "infusion" became a separate required
  topic; trials of lidocaine injection were screened out (2).

### 4. Publication type "Review" on case series (eligibility, 10)

In CD010778 (a review that included case series), 8 included studies carry the publication type "Review" in their
PubMed or Semantic Scholar metadata (for example "Surgical management of grade I chondrosarcoma of the long bones")
and were excluded as review articles.

### 5. Depth and the screening cap in large reviews (search 266, cap 27)

Reviews with many included studies lose most by depth: CD012069 (245 included) had 134 included studies matched by
CiteLume's queries beyond the 500 records fetched, and 25 eligible ones beyond the 300-record screening cap.
CD008170 (81 included) had 55 matched beyond the 500. A deeper search and a higher cap would recover many of these,
at a cost in time and model calls; both are settings.

### 6. Query sensitivity (search 169)

Concentrated in a few reviews (CD005139 40, CD012069 56, CD008170 26, CD009642 24). Same causes as 1 and 2 at the
search stage: the question's words and the PubMed term mapping do not reach studies named by drug or procedure.

## What the paper should say

- Most loss is at search (80%), half of it depth. A Cochrane search is broad and hand-tuned; a search built from the
  review's title alone is not, and CiteLume makes the gap visible ("retrieved N of M") rather than hiding it.
- The reproducibility rule that selection does not depend on the model has a measurable recall cost on poorly
  parsed questions; reporting it is part of the reproducibility claim.
- Model screening was not the weak point: 9 included studies screened out, 8 of them by criteria built from the
  question's structure (comparators, bracketed synonyms), not by the model misreading an abstract.

## Work after the paper (post-freeze, new protocol version, reported separately)

1. Let the model's concept parse inform selection while keeping a rule-based record of it (store the model's
   concepts in the corpus, so selection replays without the model).
2. Class concepts: expand by MeSH tree (children of the heading) and by pharmacological action; fall back to the
   model's member list stored in the corpus.
3. Screening criteria: comparators ("versus") and bracketed synonyms not made required.
4. Publication type "Review" from Semantic Scholar not trusted alone when PubMed says "Journal Article" with a study
   design in the abstract.
5. Depth and screening cap scaled to the number of matches, with the cost stated.
