"""
Lays out a systematic-style review in the PRISMA 2020 reporting order:

  Abstract (structured)  1 Introduction  2 Methods  3 Results (3.1 study selection with the flow diagram,
  3.2 study characteristics, 3.3 risk of bias, 3.4 individual studies, 3.5 syntheses, 3.6 reporting biases,
  3.7 certainty)  4 Discussion  5 Other information.

The Introduction, the syntheses, certainty and the Discussion are written by the model (synthesis_engine). The
Methods and the descriptive parts of the Results are computed here from what the pipeline actually did, so they
cannot misreport the method. Item numbers refer to the PRISMA 2020 checklist (Page et al., BMJ 2021;372:n71).
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from frameworks.review_types import ORGANIZATIONS
from frameworks.risk_of_bias import OVERALL_RULE, TOOLS, tool_for
from models import PaperRecord, ReviewProtocolReport, SelectionReport, SynthesisSection
from services.paper_selection import STUDY_TYPE_LABELS, reports_null_result

ROB_LEVELS = ("Low", "Some concerns", "High", "No information")


def _n(count: int, word: str, plural: Optional[str] = None) -> str:
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def _labelled(label: str, text: str) -> str:
    return f"**{label}.** {text}"


def _kind(heading: str) -> str:
    h = heading.strip()
    if h.startswith("1."):
        return "introduction"
    if h.startswith("4.") or h.startswith("E."):
        return "discussion"
    if h.startswith("C."):
        return "heterogeneity"
    if h.startswith("D."):
        return "certainty"
    return "synthesis"


def _sub_heading(heading: str) -> str:
    """'B.2 Dose and fracture risk' -> 'Dose and fracture risk'; 'B. Thematic Evidence Synthesis' -> same text."""
    return re.sub(r"^[A-Z](?:\.\d+)?\.?\s+", "", heading.strip()) or heading


# ---------------------------------------------------------------------------------------------------------------
# Computed parts
# ---------------------------------------------------------------------------------------------------------------
def _objective(query: str) -> str:
    q = query.strip().rstrip("?.")
    return f"To identify, appraise and synthesise the primary studies that address the question: {q}?"


def _methods(protocol: ReviewProtocolReport, sel: SelectionReport, papers: Sequence[PaperRecord], model: str,
             organization_id: str, search_date: str) -> str:
    flow = protocol.flow
    sources = list((flow.identified_by_source if flow else {}).keys())
    citation_links = any("Citation links" in " ".join(p.retrieved_by or []) for p in papers)
    tools_used = sorted({tool_for(p.study_type).tool_id for p in papers}, key=list(TOOLS).index)
    organization = ORGANIZATIONS.get(organization_id, ORGANIZATIONS["thematic"]).label.lower()
    n_searches = max((len(v) for v in protocol.search_strategies.values()), default=0)
    full_text = flow.full_text_retrieved if flow else 0
    criteria = [c for c in protocol.eligibility_criteria if not c.startswith("Sources searched")]
    parts = [
        _labelled("Eligibility criteria (item 5)", " ".join(criteria)),
        _labelled("Information sources (item 6)",
                  f"{', '.join(sources) or 'The enabled databases'} were searched on {search_date}"
                  + ("; the reference lists and citing articles of highly cited included studies were also searched"
                     if citation_links else "") + "."),
        _labelled("Search strategy (item 7)",
                  f"Each database was searched with {_n(n_searches, 'search string')} built from the question's "
                  "concepts and their synonyms by fixed rules (core concepts, reviews and one search per aspect of "
                  "the question). The exact strings for every database are listed under Review Method."),
        _labelled("Selection process (item 8)",
                  "Records were de-duplicated by DOI, PubMed ID and normalised title, then screened on title and "
                  "abstract in two passes: matching of the question's concepts, then screening by the model "
                  f"({model}) in order of score until the study limit was filled. The model checked each included "
                  "study's relevance again during data extraction. There was no independent dual screening."),
        _labelled("Data collection process (item 9)",
                  f"Data were extracted by a language model ({model}) from the abstract of every included study and "
                  f"from the open-access full text where it could be retrieved ({full_text} of {len(papers)}). Every "
                  "figure was checked by code against the source text, and unsupported figures were removed."),
        _labelled("Data items (item 10)",
                  "Study design, population or system, sample size, intervention or exposure, outcomes and the "
                  "results reported for them, the authors' stated limitations, and the direction of the finding "
                  "relative to the question (supporting, mixed, contradicting or neutral)."),
        _labelled("Study risk of bias assessment (item 11)",
                  "Each study was judged by the model on the domains of a tool matched to its design: "
                  + "; ".join(TOOLS[t].label for t in tools_used) + ". Each domain was rated Low, Some concerns, "
                  "High or No information from what the study reports. " + OVERALL_RULE),
        _labelled("Effect measures (item 12)",
                  "Effect measures are reported as each study gave them (for example risk ratios, odds ratios, "
                  "hazard ratios or mean differences); none were recalculated."),
        _labelled("Synthesis methods (item 13)",
                  "Narrative synthesis without meta-analysis, following SWiM: studies were grouped "
                  f"{'by theme' if organization == 'thematic' else 'in ' + organization + ' order'}, results were "
                  "compared within groups, agreement was summarised by direction of effect (vote counting by "
                  "direction, not by statistical significance), and differences in population, design and outcome "
                  "were examined to explain heterogeneity. No results were pooled statistically."),
        _labelled("Reporting bias assessment (item 14)",
                  "Not assessed formally: with no pooled estimate, funnel plots and statistical tests for small-study "
                  "effects were not possible. The searches included preprint servers to reduce publication bias."),
        _labelled("Certainty assessment (item 15)",
                  "Certainty was considered against the five GRADE domains (risk of bias, inconsistency, "
                  "indirectness, imprecision and publication bias) using indicators computed from the included "
                  "studies; this informs, but is not, a formal GRADE rating."),
    ]
    return "\n\n".join(parts)


def _study_selection(protocol: ReviewProtocolReport, sel: SelectionReport) -> str:
    f = protocol.flow
    if not f:
        return "The record flow was not available."
    by_source = "; ".join(f"{k} {v}" for k, v in f.identified_by_source.items())
    sentences = [
        f"The searches identified {_n(f.identified_total, 'record')}" + (f" ({by_source})" if by_source else "")
        + f". After {_n(f.duplicates_removed, 'duplicate')} were removed"
        + (f" and {sum(f.removed_before_screening.values())} non-research records (corrections, protocols, meeting "
           "abstracts)" if f.removed_before_screening else "")
        + f", {_n(f.screened, 'record')} were screened on title and abstract and {f.excluded_at_screening} excluded"
        + (" (" + "; ".join(f"{r.lower()}: {n}" for r, n in f.excluded_at_screening_reasons.items()) + ")"
           if f.excluded_at_screening_reasons else "") + ".",
        f"Of {_n(f.assessed_for_eligibility, 'record')} assessed for eligibility, "
        + ("; ".join(f"{n} {'was' if n == 1 else 'were'} excluded as {r[0].lower() + r[1:]}"
                     for r, n in f.excluded_at_eligibility.items()) or "none was excluded")
        + f", leaving {_n(f.eligible, 'eligible study', 'eligible studies')}.",
    ]
    if f.beyond_study_limit:
        sentences.append(f"The {f.study_limit} highest-scoring eligible studies were synthesised; the other "
                         f"{f.beyond_study_limit} are listed in the appendix as eligible but not synthesised.")
    if f.excluded_at_extraction:
        sentences.append(f"During data extraction {_n(f.excluded_at_extraction, 'study', 'studies')} "
                         f"{'was' if f.excluded_at_extraction == 1 else 'were'} found not to address the question and "
                         "left out of the synthesis.")
    sentences.append(f"{_n(f.included_in_synthesis or f.included, 'study', 'studies')} "
                     f"{'was' if (f.included_in_synthesis or f.included) == 1 else 'were'} included in the synthesis"
                     f" (full text retrieved for {f.full_text_retrieved}). The flow is shown in the diagram.")
    sentences += [n for n in sel.notes if n.startswith("Eligibility criteria relaxed")]
    return " ".join(sentences)


def _characteristics(papers: Sequence[PaperRecord]) -> str:
    years = sorted(p.year for p in papers if p.year)
    designs = Counter(STUDY_TYPE_LABELS.get(p.study_type, p.study_type or "Other") for p in papers).most_common()
    with_n = sum(1 for p in papers if p.elicit.sample_reporting_status == "Reported")
    preprints = sum(1 for p in papers if p.version_status == "preprint")
    text = (f"The {_n(len(papers), 'included study', 'included studies')}"
            + (f" were published between {years[0]} and {years[-1]}" if years else "")
            + ". Designs: " + "; ".join(f"{d} ({c})" for d, c in designs) + ". "
            + f"{with_n} reported a sample size or scale; {_n(preprints, 'was a preprint', 'were preprints') if preprints else 'none was a preprint'}. "
            + "The design, population or system, sample and setting of each study are given in the Evidence Table.")
    return text


def _risk_of_bias_summary(papers: Sequence[PaperRecord]) -> str:
    judged = [p for p in papers if p.elicit.risk_of_bias]
    if not judged:
        return ("Risk of bias could not be judged for the included studies in this run (the model did not return "
                "the judgements); see the Evidence Table for each study's reported limitations.")
    overall = Counter(p.elicit.risk_of_bias.overall for p in judged)
    domain_concerns = Counter(d.domain for p in judged for d in p.elicit.risk_of_bias.domains
                              if d.judgement in ("High", "Some concerns"))
    tools = Counter(p.elicit.risk_of_bias.tool for p in judged).most_common()
    text = (f"Risk of bias was judged for {len(judged)} of {_n(len(papers), 'study', 'studies')} "
            f"({'; '.join(f'{t}: {c}' for t, c in tools)}). Overall: "
            + "; ".join(f"{lvl.lower()} {overall[lvl]}" for lvl in ROB_LEVELS if overall.get(lvl)) + ".")
    if domain_concerns:
        text += (" The domains most often rated High or Some concerns were "
                 + "; ".join(f"{d.lower()} ({c})" for d, c in domain_concerns.most_common(3)) + ".")
    high = [p for p in judged if p.elicit.risk_of_bias.overall == "High"]
    if high:
        text += " Studies at high risk: " + ", ".join(f"[{p.citation_index}]" for p in high) + "."
    no_info = overall.get("No information", 0)
    if no_info:
        text += (f" For {no_info} studies most domains could not be judged because the abstract does not report "
                 "them; this reflects reporting, not necessarily bias.")
    return text + " Judgements for each study and domain are in the table."


def _reporting_biases(papers: Sequence[PaperRecord]) -> str:
    nulls = sum(1 for p in papers if reports_null_result(p) or p.stance == "contradicting")
    preprints = sum(1 for p in papers if p.version_status == "preprint")
    return (f"Reporting bias was not assessed statistically. {nulls} of {len(papers)} included studies report null "
            f"or contrary findings, and {preprints} {'is a preprint' if preprints == 1 else 'are preprints'}; studies "
            "that are unpublished, not indexed in the searched databases, or in languages the databases do not "
            "cover may be missing.")


def _other_information(protocol: ReviewProtocolReport, sel: SelectionReport) -> str:
    parts = [
        _labelled("Registration and protocol (item 24)",
                  "This review was not registered and no protocol was prepared in advance; the methods are fixed by "
                  "the CiteLume software (selection protocol " + (sel.protocol_version or "n.a.") + ")."),
        _labelled("Support (item 25)", "None."),
        _labelled("Competing interests (item 26)", "None declared."),
        _labelled("Availability of data and methods (item 27)",
                  "The extracted data, search strings, record flow and risk-of-bias judgements can be exported from "
                  "this page (CSV, JSON, Markdown, PDF); the selection rules are published in docs/RANKING.md."),
    ]
    if protocol.previous_reviews:
        parts.append(_labelled(
            "Earlier reviews identified (excluded from the synthesis, cited in the Introduction and Discussion)",
            " ".join(f"[{r.label}] {r.authors or 'n.a.'} ({r.year or 'n.d.'}). {r.title}. {r.venue}."
                     + (f" {r.url}" if r.url else "") for r in protocol.previous_reviews)))
    return "\n\n".join(parts)


def _appendix(sel: SelectionReport) -> Optional[SynthesisSection]:
    if not sel.eligible_over_limit:
        return None
    lines = [f"These {len(sel.eligible_over_limit)} studies met the eligibility criteria but were not synthesised "
             f"because of the {sel.target_count}-study limit (they had lower scores):"]
    lines += [f"- {r.authors or 'n.a.'} ({r.year or 'n.d.'}). {r.title}. {r.venue}." for r in sel.eligible_over_limit]
    return SynthesisSection(heading="Appendix: Eligible Studies Not Synthesised", content="\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------
def compose_prisma_report(
    query: str,
    executive_answer: str,
    sections: Sequence[SynthesisSection],
    papers: Sequence[PaperRecord],
    selection_report: SelectionReport,
    protocol: ReviewProtocolReport,
    model: str,
    organization_id: str = "thematic",
    search_date: Optional[str] = None,
) -> Tuple[str, List[SynthesisSection]]:
    """Returns the structured abstract and the sections in PRISMA 2020 order."""
    search_date = search_date or datetime.now(timezone.utc).strftime("%d %B %Y")
    synth_papers = [p for p in papers if p.relevance != "off_topic"] or list(papers)
    flow = protocol.flow

    intro = [s for s in sections if _kind(s.heading) == "introduction"]
    syntheses = [s for s in sections if _kind(s.heading) == "synthesis"]
    heterogeneity = [s for s in sections if _kind(s.heading) == "heterogeneity"]
    certainty = [s for s in sections if _kind(s.heading) == "certainty"]
    discussion = [s for s in sections if _kind(s.heading) == "discussion"]

    out: List[SynthesisSection] = []
    intro_text = intro[0].content if intro else ""
    out.append(SynthesisSection(
        heading="1. Introduction",
        content=(intro_text + "\n\n" if intro_text else "") + _labelled("Objective (item 4)", _objective(query)),
        cited_indices=intro[0].cited_indices if intro else []))
    out.append(SynthesisSection(
        heading="2. Methods",
        content=_methods(protocol, selection_report, synth_papers, model, organization_id, search_date)))
    out.append(SynthesisSection(heading="3.1 Study Selection", content=_study_selection(protocol, selection_report),
                                figure="prisma_flow"))
    out.append(SynthesisSection(heading="3.2 Study Characteristics", content=_characteristics(synth_papers)))
    out.append(SynthesisSection(heading="3.3 Risk of Bias in Studies", content=_risk_of_bias_summary(synth_papers),
                                figure="risk_of_bias"))
    out.append(SynthesisSection(
        heading="3.4 Results of Individual Studies",
        content="The findings of each included study, with the effect estimates as reported, are in the Evidence "
                "Table (one row per study)."))
    k = 0
    for s in syntheses + heterogeneity:
        k += 1
        label = "Heterogeneity and Conflicting Results" if _kind(s.heading) == "heterogeneity" else _sub_heading(s.heading)
        out.append(s.model_copy(update={"heading": f"3.5.{k} {label}"}))
    if k == 0:
        out.append(SynthesisSection(heading="3.5 Results of Syntheses", content="No synthesis could be written."))
    out.append(SynthesisSection(heading="3.6 Reporting Biases", content=_reporting_biases(synth_papers)))
    for s in certainty[:1]:
        out.append(s.model_copy(update={"heading": "3.7 Certainty of Evidence"}))
    for s in discussion[:1]:
        out.append(s.model_copy(update={"heading": "4. Discussion"}))
    out.append(SynthesisSection(heading="5. Other Information", content=_other_information(protocol, selection_report)))
    appendix = _appendix(selection_report)
    if appendix:
        out.append(appendix)

    sources = ", ".join((flow.identified_by_source if flow else {}).keys()) or "the enabled databases"
    designs = Counter(STUDY_TYPE_LABELS.get(p.study_type, p.study_type or "other") for p in synth_papers).most_common(3)
    abstract = "\n\n".join([
        _labelled("Objective", _objective(query)),
        _labelled("Methods",
                  f"{sources} were searched on {search_date}. Records were screened against pre-set eligibility "
                  "criteria (primary studies addressing every part of the question) by concept matching and then by "
                  "a language model on title and abstract; data were "
                  "extracted and risk of bias judged with design-matched tools with AI assistance, and results were "
                  "synthesised narratively without meta-analysis."),
        _labelled("Results",
                  (f"Of {flow.identified_total} records identified ({flow.screened} screened), {flow.eligible} met "
                   f"the criteria and {flow.included_in_synthesis or flow.included} were synthesised"
                   + (f" ({'; '.join(f'{d.lower()} {c}' for d, c in designs)})" if designs else "") + ". "
                   if flow else "") + executive_answer.strip()),
        _labelled("Limitations",
                  "Automated single-pass screening and AI-assisted extraction; no protocol registration; "
                  "certainty informed by, but not formally rated with, GRADE."),
    ])
    return abstract, out
