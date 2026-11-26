"""Writes docs/RANKING.md from the selection protocol's own constants, so the document always matches the code.
Run from backend/:  python -m services.ranking_doc"""
from pathlib import Path

from services import paper_selection as ps
from services import study_limits as sl
from services import query_planning as qp

DOC_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "RANKING.md"


def render() -> str:
    std_core, std_other = qp.RESULTS_PER_SEARCH["standard"]
    deep_core, deep_other = qp.RESULTS_PER_SEARCH["deep_research"]
    n_std, n_deep = qp.ASPECTS_BY_MODE["standard"], qp.ASPECTS_BY_MODE["deep_research"]
    prisma_core, prisma_other = qp.RESULTS_PER_SEARCH_PRISMA
    from services import citation_chasing as cc
    from services import mesh_terms as mt
    from services.question_parsing import ACTION_GERUNDS
    lines = [
        "# How CiteLume chooses its studies",
        "",
        f"Protocol version: `{ps.PROTOCOL_VERSION}`. This file is generated from the code "
        "(`backend/services/paper_selection.py` and `backend/services/query_planning.py`); do not edit it by hand.",
        "",
        f"A **quick answer** uses at most **{sl.QUICK_LIMIT} studies**. A **guided review** sets its study limit in "
        f"the review protocol: a narrative review uses up to **{sl.NARRATIVE_LIMIT}** by default, and systematic-"
        "style and scoping reviews include **every study that meets the criteria**, up to a ceiling "
        f"({sl.DEFAULT_CEILING} for systematic-style and {sl.DEFAULT_SCOPING_CEILING} for scoping reviews in a "
        f"local install, {sl.DEMO_CEILING} on the public demo). The ceiling is a cost limit, not a method choice: "
        "the model reads every included study, and eligible studies beyond it are listed. After the search, the "
        "user may choose a smaller number from the same records without searching again. "
        "A topic can have thousands of papers, so which ones are chosen matters. The choice follows the fixed "
        "steps below. Nothing in steps 2 to 5 uses an AI model: the same records always give the same result, and "
        "every result shows the counts at each step and the score of each included study.",
        "",
        "An AI model, when one is configured, may widen the *searches* in step 1 by proposing synonyms and "
        "aspects. Steps 2 to 5 do not read the model's plan: studies are matched against the concepts parsed from "
        "the question's own words by rule, so the same records give the same included studies with any model or "
        "none.",
        "",
        "## Step 0. The review protocol",
        "",
        "Before searching, the question is turned into a review protocol: the review type and why, the parts of "
        "the question (concepts) with their synonyms and which of them is the main topic, the populations or places "
        "every study must name, the aspects that get their own searches, and the eligibility criteria (study "
        "designs, publication years, whether reviews are included). The tool proposes a default from the question's "
        "own words; in the guided review the user confirms or edits each part. An AI model may suggest extra "
        "synonyms or aspects, but they are used only if the user accepts them. The protocol is stored with the "
        "results, and running a search with the default protocol gives the same studies as running it without "
        "one. When the user supplies a protocol, no model is involved in steps 1 to 5.",
        "",
        "## Step 1. Search",
        "",
        "1. **Concepts.** The question is reduced to one to three concepts that every relevant paper must be "
        "about, each with synonyms and alternative spellings (for example *gentamicin* / *gentamycin*). A place "
        "or population named in the question (for example *India*, *older adults*) is required as well. When an "
        "AI model is configured it proposes the concepts and synonyms; they are accepted only if the leading "
        "concept is anchored in the question's own words. Without a model the concepts are taken from the "
        f"question's wording. At most {qp.MAX_REQUIRED_CONCEPTS} concepts are required in every search (more makes "
        "searches too narrow); a further concept (for example the organism in *atrazine biotransformation by "
        "Aspergillus hortai*) gets its own core search together with the main concept.",
        "   Reading the question's wording follows fixed rules, built for questions written as review titles: "
        "typographic dashes count as hyphens; a slash between two words names alternatives (*chronic "
        "prostatitis/chronic pelvic pain syndrome*); purpose words frame a concept rather than join it (*for the "
        "prevention of*, *treating*, *protecting*; the full list has "
        f"{len(ACTION_GERUNDS)} verbs); the research action a title names (*assessment of*, *evaluation of*, "
        "*analysis of*, *characterisation of*) frames the question and is not a concept; a label for a class of "
        "interventions (*non-pharmacological interventions*, "
        "*... strategies*, *... programmes*) is not required when the question names something else; and a term "
        "in brackets after a concept is another name for it (*intraocular lenses (IOLs)*).",
        "2. **Breadth.** The question is classed as a *broad topic* to survey (\"gut microbiome\") or a "
        "*focused question* relating concepts (\"Does vitamin D prevent COVID-19?\"). A question that relates "
        "concepts is always treated as focused.",
        f"3. **Aspects.** The question is split into {n_std} aspects (Standard) or {n_deep} (Deep Research), such "
        "as *Mechanisms*, *Treatment* or *Null or conflicting findings*. The model proposes them when available; "
        "otherwise a template for the field and type of question is used.",
        "4. **Searches.** Each database (PubMed with its Europe PMC mirror, Semantic Scholar with OpenAlex as "
        "backup, bioRxiv/medRxiv, and arXiv) receives a *core* search (the concepts only), a *reviews* search "
        "(the concepts plus review and meta-analysis filters) and one search per aspect (the concepts plus that "
        f"aspect's terms). The core search requests {std_core} records per database and every other search "
        f"{std_other} (Deep Research: {deep_core} and {deep_other}; a systematic-style review: {prisma_core} and "
        f"{prisma_other}). A local install can set a deeper PubMed core search (`pubmed_depth`, up to 2,000 "
        "records, fetched in batches of 200); the public demo keeps its limits. The search strings are compiled by "
        "fixed rules, never written by the model, and are shown with the results, with the number of records each "
        "PubMed search matched and how many were retrieved (\"retrieved N of M\").",
        "5. **MeSH headings.** For each concept, PubMed is asked once how it maps the concept's words to its "
        "controlled vocabulary, and the MeSH headings and pharmacological actions it returns become one more "
        "alternative in the PubMed query (for example *antidepressants* also searches *Antidepressive Agents* "
        "[Pharmacological Action], which covers trials of doxepin or trazodone). Only parts of two or more words, "
        "and the word the concept ends with, are mapped, so a single leading word does not pull in another sense; "
        f"at most {mt.MAX_TAGS_PER_WORD} headings per part are used. When only the final word is mapped and a word "
        "before it is a substance (PubMed maps it to a Supplementary Concept), that word stays required with the "
        "heading (*atrazine biotransformation* searches *Biotransformation* [MeSH] together with *atrazine*, not any "
        "biotransformation study). The headings are stored with the plan, so the "
        "query is the same on every run; if PubMed does not answer, the query is used without them.",
        "6. **Other methods** (Deep Research). After a first selection, the reference lists and the citing papers "
        f"of up to {cc.MAX_SEEDS} selected studies are fetched from Semantic Scholar; a paper linked to at least "
        f"{cc.MIN_LINKS} of them and not already found is added (up to {cc.MAX_ADDED}), as a record identified by "
        "citation searching. The reference lists of reviews that the eligibility rules set aside (on the topic, "
        f"excluded only for being reviews; up to {cc.MAX_REVIEW_SEEDS}) are searched the same way, and every study "
        f"they cite is added (up to {cc.REVIEW_REFS_MAX_ADDED}). Missing abstracts and publication types of the "
        "added records are filled from PubMed. Selection then runs again over the enlarged pool, so the added "
        "records meet the same rules; the flow diagram counts them separately, as identified by other methods.",
        "",
        "## Step 2. Merge duplicates",
        "",
        "Records for the same paper from different databases, and a preprint with its published version, are "
        "merged by DOI, PubMed ID, arXiv ID or matching title.",
        "",
        "## Step 3. Eligibility",
        "",
        "A record is removed, and counted under its reason, if it is:",
        "",
        "- **outside the place or population asked about** (matched on whole words: *Indiana* and *American "
        "Indian* do not count as India);",
        "- **not about the topic**: the main concept, and its synonyms, appear nowhere in the title or abstract;",
        "- **not a research report**: a correction, retraction notice or retracted article, a study protocol "
        "with no results, or a meeting abstract;",
        f"- **without an abstract**, when at least as many assessable records remain as are needed.",
        "",
        f"The first two rules apply only when at least {ps.MIN_CONSTRAINT_MATCHES} records pass them; otherwise "
        "the records are kept but flagged as off topic and left out of the written synthesis.",
        "",
        "The review protocol's eligibility criteria then apply, each counted under its own reason:",
        "",
        f"- **{ps.REASON_YEARS.lower()}** (when years are set);",
        f"- **{ps.REASON_DESIGN.lower()}** (when study designs are set; every design is eligible otherwise);",
        f"- **{ps.REASON_REVIEW_BY_PROTOCOL.lower()}** (when the protocol excludes reviews; in a systematic-style "
        "review reviews are excluded unless the protocol includes them).",
        "",
        "## Step 4. Score (100 points)",
        "",
        "| Criterion | Points | Rule |",
        "|---|---|---|",
    ]
    for c in ps.CRITERIA:
        lines.append(f"| **{c.label}** | {c.max_points:g} | {c.rule} |")
    lines += [
        "",
        "### Points by study type",
        "",
        "Study design decides how far a result can be trusted in health, life and social sciences, so those "
        "fields use the left column. Computing, mathematics, physics, chemistry and engineering have no such "
        "hierarchy, so primary work there is not ranked by design (right column).",
        "",
        "| Study type | Health, life and social sciences | Technical fields |",
        "|---|---|---|",
    ]
    for key, pts in ps.EVIDENCE_POINTS["empirical"].items():
        lines.append(f"| {ps.STUDY_TYPE_LABELS[key]} | {pts:g} | {ps.EVIDENCE_POINTS['technical'][key]:g} |")
    lines += [
        "",
        f"A narrative review scores {ps.REVIEW_POINTS_FOR_BROAD_TOPICS:g} when the question is a broad topic, "
        f"because reviews are what best answer a survey question. A preprint loses {ps.PREPRINT_PENALTY:g} points. "
        "The study type comes from the database's own publication type where it gives one (for example PubMed's "
        "*Randomized Controlled Trial*), otherwise from the wording of the title and abstract.",
        "",
        "### Details",
        "",
        "- **Concept match** (30 of the relevance points): each concept scores 1.0 if it, or a synonym, is in the "
        "title, 0.7 for two or more mentions in the abstract, 0.4 for one, 0 for none. An abbreviation on its "
        "own (for example *CHD*) scores 0.4, because abbreviations are ambiguous. The scores are averaged with "
        "the main concept counted twice, and multiplied by 30.",
        "- **Wording overlap** (20 of the relevance points): the standard BM25 text-retrieval score of the title "
        "(counted twice) and abstract against the question and its concepts. The study's rank among the eligible "
        "studies, as a fraction, is multiplied by 20.",
        "- **Citation impact**: citations divided by years since publication, ranked among the eligible studies "
        "and multiplied by 15. Counts come from NIH iCite for PubMed records and from Semantic Scholar, OpenAlex "
        "or Europe PMC for the rest.",
        "- **Ranked against the other candidates** means the scale adapts to the topic: a study is compared "
        "with the other studies found for this question, not with all of science.",
        "",
        "## Step 5. Fill the places",
        "",
    ]
    lines += [f"{i}. {rule}" for i, rule in enumerate(ps.SELECTION_RULES, start=1)]
    lines += [
        "",
        "The included studies are then numbered [1], [2], ... in this order.",
        "",
        "### Systematic-style (PRISMA 2020) reviews",
        "",
        "A guided review of the systematic-style type replaces this step with eligibility criteria, so inclusion "
        "does not depend on quotas:",
        "",
    ]
    lines += [f"{i}. {rule}" for i, rule in enumerate(ps.PRISMA_RULES, start=1)]
    lines += ["", "### Scoping reviews", "",
              "A scoping review also includes every eligible study, but maps reviews as well and does not judge "
              "risk of bias:", ""]
    lines += [f"{i}. {rule}" for i, rule in enumerate(ps.SCOPING_RULES, start=1)]
    lines += [
        "",
        "Every exclusion is counted at its PRISMA stage (identification, screening, eligibility) and shown in the "
        "flow diagram. Each included study also receives a risk-of-bias judgement on the domains of a tool matched "
        "to its design (RoB 2, ROBINS-I/E, JBI, SYRCLE, CASP, AMSTAR 2 or a technical validity checklist); the "
        "overall judgement is computed from the domains by a fixed rule.",
        "",
        "## What you see with every result",
        "",
        "- How many records were found, merged, removed for each reason, eligible and included.",
        "- The concepts and synonyms every search required, and the exact search strings.",
        "- For each included study: its total score, the five component scores, its study type, where its "
        "citation count came from, and the rule that included it.",
        "- How many included studies cover each aspect of the question.",
        "",
        "## After selection",
        "",
        "When an AI model is configured it reads each included study and can mark one as not addressing the "
        "question. Such a study keeps its number and stays in the reference list, but is left out of the written "
        "synthesis, and the results say so.",
        "",
        "In a systematic-style (PRISMA) review the model also screens titles and abstracts before the study limit "
        "is applied. That screening is a separate step after the rule-based selection, not part of it: it can "
        "differ between models and runs, so every decision is stored with its reason and the model's name, and "
        "the flow diagram counts its exclusions on their own line.",
        "",
        "## Limits to state when presenting this",
        "",
        "- It is a ranked sample, not a systematic search. The candidate pool is what the databases return for "
        "these searches (typically 150 to 300 unique records in Standard), not every paper on the topic.",
        "- Topic matching is by words and synonyms, and by MeSH indexing for records PubMed indexes. A relevant "
        "paper that names a member of a class (a drug, a therapy) when the question names the class can still be "
        "scored low or missed, especially when PubMed maps the class to a narrower or different heading.",
        "- Selection reads a rule-based parse of the question, not the model's, so that the same records give the "
        "same studies with any model. When the question's wording parses poorly (a verb or participle joined to a "
        "concept), this costs recall; the concepts used are shown with every result.",
        "- Measured on 22 held-out Cochrane intervention reviews (eval/results/6_test.md): a systematic-style run's "
        "pool held 54% of the reviews' included studies on average and its selection 25%; most of the loss was at "
        "search. Changing any weight to half or one and a half times its value changed about one of 20 selected "
        "studies and recall by under half a point (eval/results/7_sensitivity_test.md).",
        "- Citation counts favour older and English-language work; the neutral value for recent papers and the "
        "recency points offset this only partly.",
        "- Study type recognised from wording can be wrong; the type shown for each study makes this checkable.",
        "- The weights are choices, not laws. They are constants at the top of "
        "`backend/services/paper_selection.py` and can be changed in one place.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"Wrote {DOC_PATH}")
