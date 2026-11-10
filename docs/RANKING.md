# How CiteLume chooses its studies

Protocol version: `citelume-selection/2.0`. This file is generated from the code (`backend/services/paper_selection.py` and `backend/services/query_planning.py`); do not edit it by hand.

CiteLume answers a question from at most **20 studies (Standard)** or **60 studies (Deep Research)**. A topic can have thousands of papers, so which ones are chosen matters. The choice follows the fixed steps below. Nothing in steps 2 to 5 uses an AI model: the same records always give the same result, and every result shows the counts at each step and the score of each included study.

## Step 1. Search

1. **Concepts.** The question is reduced to one to three concepts that every relevant paper must be about, each with synonyms and alternative spellings (for example *gentamicin* / *gentamycin*). A place or population named in the question (for example *India*, *older adults*) is required as well. When an AI model is configured it proposes the concepts and synonyms; they are accepted only if the leading concept is anchored in the question's own words. Without a model the concepts are taken from the question's wording.
2. **Breadth.** The question is classed as a *broad topic* to survey ("gut microbiome") or a *focused question* relating concepts ("Does vitamin D prevent COVID-19?"). A question that relates concepts is always treated as focused.
3. **Aspects.** The question is split into 4 aspects (Standard) or 6 (Deep Research), such as *Mechanisms*, *Treatment* or *Null or conflicting findings*. The model proposes them when available; otherwise a template for the field and type of question is used.
4. **Searches.** Each database (PubMed with its Europe PMC mirror, Semantic Scholar with OpenAlex as backup, bioRxiv/medRxiv, and arXiv) receives a *core* search (the concepts only), a *reviews* search (the concepts plus review and meta-analysis filters) and one search per aspect (the concepts plus that aspect's terms). The core search requests 30 records per database and every other search 15 (Deep Research: 50 and 25). The search strings are compiled by fixed rules, never written by the model, and are shown with the results.

## Step 2. Merge duplicates

Records for the same paper from different databases, and a preprint with its published version, are merged by DOI, PubMed ID, arXiv ID or matching title.

## Step 3. Eligibility

A record is removed, and counted under its reason, if it is:

- **outside the place or population asked about** (matched on whole words: *Indiana* and *American Indian* do not count as India);
- **not about the topic**: the main concept, and its synonyms, appear nowhere in the title or abstract;
- **not a research report**: a correction, retraction notice or retracted article, a study protocol with no results, or a meeting abstract;
- **without an abstract**, when at least as many assessable records remain as are needed.

The first two rules apply only when at least 5 records pass them; otherwise the records are kept but flagged as off topic and left out of the written synthesis.

## Step 4. Score (100 points)

| Criterion | Points | Rule |
|---|---|---|
| **Relevance to the question** | 50 | Up to 30 points for the question's concepts (or their synonyms), and any place or population it names, appearing in the study: full marks for the title, 70% for two or more mentions in the abstract, 40% for one mention; the main concept counts double. Up to 20 points for overall wording overlap with the question (BM25), ranked against the other candidate studies. |
| **Strength of study type** | 20 | Points by study type: systematic review, meta-analysis or guideline 20; randomised trial 18; cohort, case-control or other controlled study 14; cross-sectional, laboratory, computational or other primary study 10-11; narrative review 8 (16 when the question is a broad topic to survey); case report 5; commentary 2. Preprints that are not peer reviewed lose 3 points. |
| **Citation impact** | 15 | Citations per year since publication, ranked against the other candidate studies. Studies from the last two calendar years, and studies with no citation data, get at least the middle value so new work is not penalised. |
| **Recency** | 10 | 10 points for the last two years, then one point less for each year older (0 from 12 years). |
| **Found by several searches** | 5 | 5 points if three or more independent searches or databases returned the study, 3 points for two. |

### Points by study type

Study design decides how far a result can be trusted in health, life and social sciences, so those fields use the left column. Computing, mathematics, physics, chemistry and engineering have no such hierarchy, so primary work there is not ranked by design (right column).

| Study type | Health, life and social sciences | Technical fields |
|---|---|---|
| Systematic review / meta-analysis / guideline | 20 | 16 |
| Randomised controlled trial | 18 | 14 |
| Cohort / case-control / controlled study | 14 | 14 |
| Cross-sectional / survey study | 11 | 14 |
| Laboratory study (cell / animal / assay) | 11 | 14 |
| Computational / modelling study | 11 | 14 |
| Theoretical / formal study | 10 | 14 |
| Qualitative study | 10 | 12 |
| Primary research article | 10 | 14 |
| Narrative review | 8 | 10 |
| Case report / case series | 5 | 6 |
| Editorial / commentary / letter | 2 | 2 |

A narrative review scores 16 when the question is a broad topic, because reviews are what best answer a survey question. A preprint loses 3 points. The study type comes from the database's own publication type where it gives one (for example PubMed's *Randomized Controlled Trial*), otherwise from the wording of the title and abstract.

### Details

- **Concept match** (30 of the relevance points): each concept scores 1.0 if it, or a synonym, is in the title, 0.7 for two or more mentions in the abstract, 0.4 for one, 0 for none. An abbreviation on its own (for example *CHD*) scores 0.4, because abbreviations are ambiguous. The scores are averaged with the main concept counted twice, and multiplied by 30.
- **Wording overlap** (20 of the relevance points): the standard BM25 text-retrieval score of the title (counted twice) and abstract against the question and its concepts. The study's rank among the eligible studies, as a fraction, is multiplied by 20.
- **Citation impact**: citations divided by years since publication, ranked among the eligible studies and multiplied by 15. Counts come from NIH iCite for PubMed records and from Semantic Scholar, OpenAlex or Europe PMC for the rest.
- **Ranked against the other candidates** means the scale adapts to the topic: a study is compared with the other studies found for this question, not with all of science.

## Step 5. Fill the places

1. Studies that match every part of the question are placed ahead of studies that match only part of it; within each group the order is by score.
2. Coverage: each aspect of the question first receives its highest-scoring study (two in Deep Research), provided that study is on topic.
3. Null results: for questions about an effect or association, up to two studies (four in Deep Research) that report null or contrary findings are kept in if any are eligible.
4. Remaining places are filled in order of score.
5. Narrative reviews are limited to 30% of the included studies (50% for broad topics), so primary evidence is not crowded out.
6. No more than two studies with the same first author and year, to favour independent evidence.
7. Ties are broken by newer publication year, then citations per year, then title.

The included studies are then numbered [1], [2], ... in this order.

## What you see with every result

- How many records were found, merged, removed for each reason, eligible and included.
- The concepts and synonyms every search required, and the exact search strings.
- For each included study: its total score, the five component scores, its study type, where its citation count came from, and the rule that included it.
- How many included studies cover each aspect of the question.

## After selection

When an AI model is configured it reads each included study and can mark one as not addressing the question. Such a study keeps its number and stays in the reference list, but is left out of the written synthesis, and the results say so.

## Limits to state when presenting this

- It is a ranked sample, not a systematic search. The candidate pool is what the databases return for these searches (typically 150 to 300 unique records in Standard), not every paper on the topic.
- Topic matching is by words and synonyms. A relevant paper that uses different vocabulary can be scored low or missed.
- Citation counts favour older and English-language work; the neutral value for recent papers and the recency points offset this only partly.
- Study type recognised from wording can be wrong; the type shown for each study makes this checkable.
- The weights are choices, not laws. They are constants at the top of `backend/services/paper_selection.py` and can be changed in one place.
