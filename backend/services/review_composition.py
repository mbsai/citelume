"""
Review-type composition for the rule-based synthesis: organises the findings section (B) chronologically,
methodologically or theoretically, and adapts sections C-E to the selected review framework (narrative,
PRISMA-style systematic, or PRISMA-ScR scoping). Thematic organisation is the composer's own default and is not
rebuilt here.

Everything is derived from the retrieved study records, so these sections are available with any model, or none.
"""
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from frameworks.review_types import ORGANIZATIONS, REVIEW_FRAMEWORKS, automation_disclosure
from models import AtomicFinding, PaperRecord, SynthesisClaim, SynthesisSection

_NOT_REPORTED = {"", "not reported", "unavailable", "not applicable"}
_THEORY_TERMS = r"(?:theory|theorem|framework|hypothesis|principle|law|paradigm|equation|model)"
_THEORY_PHRASE = re.compile(rf"\b((?:[a-z][\w'-]*\s+){{1,3}}){_THEORY_TERMS}s?\b", re.I)
_THEORY_STOPWORDS = {
    "the", "a", "an", "our", "this", "that", "these", "their", "its", "we", "new", "novel", "proposed", "present",
    "existing", "current", "simple", "general", "same", "such", "each", "both", "of", "in", "on", "for", "and",
    "with", "to", "by", "is", "are", "be", "using", "based", "trained", "learning", "prediction", "predictive",
    "language", "large", "machine", "deep", "statistical", "regression", "mathematical", "computational",
}
# Minimum number of studies that must name a theory before it becomes its own group
MIN_STUDIES_PER_THEORY = 2


def _finding_text(p: PaperRecord, findings_by_idx: Dict[int, AtomicFinding]) -> str:
    kf = (p.elicit.key_findings or "").strip()
    if kf.lower() in _NOT_REPORTED:
        f = findings_by_idx.get(p.citation_index)
        kf = f.observed_result if f else ""
    return kf.rstrip(".")[:260] or "No result stated in the abstract"


def _lead_author(p: PaperRecord) -> str:
    if not p.authors:
        return p.source
    tokens = p.authors[0].replace(",", " ").split() or [p.authors[0]]
    # PubMed lists "Surname Initials" (e.g. "Chen L"); most other sources list "Given Surname"
    surname = tokens[0] if len(tokens) > 1 and tokens[-1].isupper() and len(tokens[-1]) <= 3 else tokens[-1]
    return f"{surname} et al." if len(p.authors) > 1 else surname


def _cites(papers: Sequence[PaperRecord]) -> str:
    return ", ".join(str(p.citation_index) for p in papers)


def _stance_summary(papers: Sequence[PaperRecord]) -> str:
    c = Counter(p.stance for p in papers)
    parts = [f"{c[s]} {s}" for s in ("supporting", "mixed", "contradicting") if c.get(s)]
    return ", ".join(parts) if parts else "direction not determinable"


def _representative(papers: Sequence[PaperRecord], n: int = 3) -> List[PaperRecord]:
    return sorted(papers, key=lambda p: (-(p.citation_count or 0), p.citation_index))[:n]


def _group_paragraph(label: str, papers: Sequence[PaperRecord], findings_by_idx: Dict[int, AtomicFinding]) -> str:
    reps = "; ".join(
        f"{_lead_author(p)} ({p.year or 'n.d.'}) report that {_finding_text(p, findings_by_idx)} [{p.citation_index}]"
        for p in _representative(papers)
    )
    return f"{label} - {len(papers)} stud{'y' if len(papers) == 1 else 'ies'} [{_cites(papers)}]; direction of effect: " \
           f"{_stance_summary(papers)}. {reps}."


def _year_periods(papers: Sequence[PaperRecord], max_periods: int = 4) -> List[Tuple[str, List[PaperRecord]]]:
    """Consecutive periods with roughly equal numbers of studies; a year is never split across periods."""
    dated = sorted((p for p in papers if p.year), key=lambda p: p.year)
    if not dated:
        return []
    years = sorted({p.year for p in dated})
    if len(years) <= max_periods:
        periods: List[List[int]] = [[y] for y in years]
    else:
        # Assign each year to the period containing the midpoint of its studies in publication order
        target = len(dated) / max_periods
        periods = [[] for _ in range(max_periods)]
        before = 0
        for y in years:
            in_year = sum(1 for p in dated if p.year == y)
            periods[min(max_periods - 1, int((before + in_year / 2) / target))].append(y)
            before += in_year
        periods = [ys for ys in periods if ys]
    out = []
    for ys in periods:
        label = str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]}-{ys[-1]}"
        out.append((label, [p for p in dated if ys[0] <= p.year <= ys[-1]]))
    return out


def _chronological(papers, findings_by_idx) -> Tuple[List[str], List[str]]:
    periods = _year_periods(papers)
    paragraphs = [
        _group_paragraph(f"Period {label}", ps, findings_by_idx)
        + f" Predominant designs: {', '.join(d for d, _ in Counter(p.elicit.study_design for p in ps).most_common(2))}."
        for label, ps in periods
    ]
    notes = []
    if len(periods) >= 2:
        first = Counter(p.elicit.study_design for p in periods[0][1]).most_common(1)[0][0]
        last = Counter(p.elicit.study_design for p in periods[-1][1]).most_common(1)[0][0]
        if first != last:
            notes.append(f"Methodological shift: the earliest period is dominated by {first}, the most recent by {last}.")
    undated = [p for p in papers if not p.year]
    if undated:
        notes.append(f"{len(undated)} record(s) without a publication year are not placed on the timeline [{_cites(undated)}].")
    return paragraphs, notes


def _methodological(papers, findings_by_idx) -> Tuple[List[str], List[str]]:
    groups: Dict[str, List[PaperRecord]] = defaultdict(list)
    for p in papers:
        groups[(p.elicit.study_design or "Design not reported").strip()].append(p)
    paragraphs = []
    for design, ps in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lims = [p.elicit.limitations for p in ps if (p.elicit.limitations or "").strip().lower() not in _NOT_REPORTED]
        lim = f" Typical limitation noted: {lims[0].rstrip('.')[:200]}." if lims else ""
        paragraphs.append(_group_paragraph(design, ps, findings_by_idx) + lim)
    return paragraphs, []


def _named_theories(papers: Sequence[PaperRecord]) -> Dict[str, List[PaperRecord]]:
    by_phrase: Dict[str, List[PaperRecord]] = defaultdict(list)
    for p in papers:
        text = f"{p.title}. {p.abstract or ''}"
        seen = set()
        for m in _THEORY_PHRASE.finditer(text):
            words = [w.lower() for w in m.group(0).split()]
            while words and words[0] in _THEORY_STOPWORDS:
                words = words[1:]
            if len(words) < 2 or all(w in _THEORY_STOPWORDS for w in words[:-1]):
                continue
            phrase = " ".join(words)
            if phrase not in seen:
                seen.add(phrase)
                by_phrase[phrase].append(p)
    shared = {ph: ps for ph, ps in by_phrase.items() if len(ps) >= MIN_STUDIES_PER_THEORY}
    return dict(sorted(shared.items(), key=lambda kv: -len(kv[1]))[:5])


def _theoretical(papers, findings_by_idx, claims: Sequence[SynthesisClaim]) -> Tuple[List[str], List[str]]:
    theories = _named_theories(papers)
    if theories:
        placed = {p.citation_index for ps in theories.values() for p in ps}
        paragraphs = [_group_paragraph(f"Framework: {ph}", ps, findings_by_idx) for ph, ps in theories.items()]
        rest = [p for p in papers if p.citation_index not in placed]
        if rest:
            paragraphs.append(_group_paragraph("Studies not tied to a shared named framework", rest, findings_by_idx))
        return paragraphs, []
    # No theory is named by enough studies: fall back to the kind of evidence each claim rests on
    groups: Dict[str, List[PaperRecord]] = defaultdict(list)
    by_idx = {p.citation_index: p for p in papers}
    for c in claims:
        for i in c.relevant_study_indices:
            if i in by_idx and by_idx[i] not in groups[c.evidence_type]:
                groups[c.evidence_type].append(by_idx[i])
    paragraphs = [_group_paragraph(t.replace("_", " ").capitalize(), ps, findings_by_idx) for t, ps in groups.items()]
    note = (f"No theory, model or framework was named by {MIN_STUDIES_PER_THEORY} or more of the retrieved studies, so "
            "studies are grouped by the kind of evidence they provide instead.")
    return paragraphs, [note]


def organized_groups(
    organization: str,
    papers: Sequence[PaperRecord],
    claims: Sequence[SynthesisClaim],
    max_groups: int = 4,
) -> List[Tuple[str, List[PaperRecord]]]:
    """
    The body subsections of a Deep Research review as (label, studies): one per theme (verified claim), period,
    method or theory, depending on the chosen organisation. Small leftover groups are merged into the last one.
    """
    by_idx = {p.citation_index: p for p in papers}
    if organization == "chronological":
        groups = [(f"Period {label}", ps) for label, ps in _year_periods(papers, max_periods=max_groups)]
    elif organization == "methodological":
        buckets: Dict[str, List[PaperRecord]] = defaultdict(list)
        for p in papers:
            buckets[(p.elicit.study_design or "Design not reported").strip()].append(p)
        groups = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    elif organization == "theoretical":
        theories = _named_theories(papers)
        groups = [(f"Framework: {ph}", ps) for ph, ps in theories.items()]
        placed = {p.citation_index for _, ps in groups for p in ps}
        rest = [p for p in papers if p.citation_index not in placed]
        if rest:
            groups.append(("Other theoretical and empirical perspectives", rest))
    else:
        groups = []
        for c in claims:
            ps = [by_idx[i] for i in c.relevant_study_indices if i in by_idx]
            if ps:
                groups.append((c.theme_label, ps))
    groups = [(label, list(ps)) for label, ps in groups if ps]
    if len(groups) > max_groups:
        head, tail = groups[: max_groups - 1], groups[max_groups - 1:]
        merged = [p for _, ps in tail for p in ps]
        groups = head + [("Further evidence: " + "; ".join(lbl for lbl, _ in tail[:3]), list({p.citation_index: p for p in merged}.values()))]
    return groups or [("Synthesis of the evidence", list(papers))]


def compose_organized_body(
    organization: str,
    papers: Sequence[PaperRecord],
    findings: Sequence[AtomicFinding],
    claims: Sequence[SynthesisClaim],
) -> SynthesisSection:
    org = ORGANIZATIONS[organization]
    findings_by_idx = {f.citation_index: f for f in findings}
    if organization == "chronological":
        paragraphs, notes = _chronological(papers, findings_by_idx)
    elif organization == "methodological":
        paragraphs, notes = _methodological(papers, findings_by_idx)
    else:
        paragraphs, notes = _theoretical(papers, findings_by_idx, claims)
    content = "\n\n".join([f"B.{i} {para}" for i, para in enumerate(paragraphs, 1)] + notes)
    return SynthesisSection(heading=org.body_heading, content=content,
                            cited_indices=sorted(p.citation_index for p in papers))


def grade_informed_indicators(papers: Sequence[PaperRecord], findings: Sequence[AtomicFinding]) -> str:
    """
    Signals for the five GRADE reasons to rate down certainty, computed from abstracts. They inform, but are not, a
    GRADE certainty rating (which needs outcome-level full-text assessment).
    """
    n = len(papers) or 1
    preprints = sum(1 for p in papers if p.version_status == "preprint")
    abstract_only = sum(1 for f in findings if f.evidence_access != "full_text")
    stances = Counter(p.stance for p in papers)
    conflicting = stances.get("contradicting", 0) + stances.get("mixed", 0)
    with_uncertainty = sum(1 for f in findings if f.uncertainty not in ("Not reported", "Not applicable"))
    with_sample = sum(1 for p in papers if p.elicit.sample_reporting_status == "Reported")
    return "\n".join([
        "GRADE-informed indicators (signals from abstracts; not a GRADE certainty rating):",
        f"• Risk of bias: {preprints} of {n} studies are preprints and {abstract_only} were appraised from abstracts "
        "only; design-matched reporting gaps are listed per study in the Evidence Table.",
        f"• Inconsistency: {conflicting} of {n} studies report mixed or contradicting results.",
        f"• Imprecision: {with_uncertainty} of {len(findings) or n} findings report an uncertainty interval; "
        f"{with_sample} of {n} studies report a sample size or scale.",
        "• Indirectness: see section C for differences in population, system and outcome between studies.",
        "• Publication bias: cannot be assessed from a single automated search; absence of evidence is not evidence "
        "of absence.",
    ])


def _evidence_base_characteristics(papers: Sequence[PaperRecord]) -> str:
    years = [p.year for p in papers if p.year]
    designs = Counter(p.elicit.study_design for p in papers).most_common(5)
    sources = Counter(p.source for p in papers).most_common()
    status = Counter(p.version_status for p in papers)
    return "\n".join([
        f"• Volume and period: {len(papers)} studies" + (f", published {min(years)}-{max(years)}." if years else "."),
        "• Study designs: " + "; ".join(f"{d} ({c})" for d, c in designs) + ".",
        "• Sources: " + "; ".join(f"{s} ({c})" for s, c in sources) + ".",
        f"• Publication status: {status.get('published', 0) + status.get('preprint_with_published_link', 0)} "
        f"peer-reviewed, {status.get('preprint', 0)} preprint(s).",
    ])


def apply_review_framework(
    framework_id: str,
    sections: List[SynthesisSection],
    papers: Sequence[PaperRecord],
    findings: Sequence[AtomicFinding],
) -> List[SynthesisSection]:
    """Renames and extends sections C-E (indices 1-3 after body B) for the selected review framework."""
    framework = REVIEW_FRAMEWORKS[framework_id]
    if framework_id == "narrative" or len(sections) < 4:
        return sections
    body, conflict, reliability, conclusion = sections[0], sections[1], sections[2], sections[3]
    c_head, d_head, e_head = framework.section_plan
    disclosure = automation_disclosure(framework)
    if framework_id == "systematic_prisma":
        return [
            body,
            conflict.model_copy(update={"heading": c_head}),
            reliability.model_copy(update={
                "heading": d_head,
                "content": f"{reliability.content}\n\n{grade_informed_indicators(papers, findings)}",
            }),
            conclusion.model_copy(update={"heading": e_head, "content": f"{conclusion.content}\n\n{disclosure}"}),
        ] + sections[4:]
    # Scoping: describe and map the evidence rather than appraise it
    return [
        body,
        SynthesisSection(heading=c_head, content=_evidence_base_characteristics(papers),
                         cited_indices=sorted(p.citation_index for p in papers)),
        SynthesisSection(
            heading=d_head,
            content=("Under-studied combinations of methods and outcomes are mapped in the Knowledge Gap Matrix tab. "
                     "Divergent or conditional findings that point to open questions:\n" + conflict.content),
            cited_indices=conflict.cited_indices,
        ),
        conclusion.model_copy(update={"heading": e_head, "content": f"{conclusion.content}\n\n{disclosure}"}),
    ] + sections[4:]


def finalize_llm_sections(
    framework_id: str,
    sections: List[SynthesisSection],
    papers: Sequence[PaperRecord],
    findings: Sequence[AtomicFinding],
) -> List[SynthesisSection]:
    """
    Adds the framework's computed parts to an LLM-written synthesis: GRADE-informed indicators for systematic-style
    reviews and the automation disclosure for systematic and scoping reviews. These are never left to the model.
    """
    if framework_id == "narrative" or not sections:
        return sections
    framework = REVIEW_FRAMEWORKS[framework_id]
    out = list(sections)
    if framework_id == "systematic_prisma":
        indicators = grade_informed_indicators(papers, findings)
        d_idx = next((i for i, s in enumerate(out) if s.heading.strip().upper().startswith("D")), None)
        if d_idx is None:
            out.append(SynthesisSection(heading=framework.section_plan[1], content=indicators,
                                        cited_indices=[p.citation_index for p in papers][:8]))
        else:
            out[d_idx] = out[d_idx].model_copy(update={"content": f"{out[d_idx].content}\n\n{indicators}"})
    out[-1] = out[-1].model_copy(update={"content": f"{out[-1].content}\n\n{automation_disclosure(framework)}"})
    return out
