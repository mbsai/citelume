import datetime
from typing import List, Optional, Tuple
from models import ExportRequest, ReviewProtocolReport, SelectionReport


def selection_report_lines(report: Optional[SelectionReport], papers=None) -> List[str]:
    """Plain-text lines on how the included studies were chosen (shared by all export formats)."""
    if not report or not report.included:
        return []
    lines = [
        f"Selection protocol {report.protocol_version} (full description: docs/RANKING.md).",
        f"Records found by the searches: {report.records_found}; after merging duplicates: {report.unique_records}; "
        f"eligible and scored: {report.eligible}; included: {report.included} (limit {report.target_count}).",
    ]
    lines += [f"- Removed ({n}): {reason}" for reason, n in report.excluded.items()]
    lines.append("Score out of 100 for each eligible study:")
    lines += [f"- {c.label} (up to {c.max_points:g}): {c.rule}" for c in report.criteria]
    lines.append("Rules for filling the places:")
    lines += [f"- {r}" for r in report.selection_rules]
    if report.citation_data:
        lines.append(report.citation_data)
    lines += list(report.notes)
    scored = [p for p in (papers or []) if p.selection_breakdown.get("total") is not None]
    if scored:
        lines.append("Scores of the included studies (relevance / study type / citations / recency / searches = total):")
        for p in scored:
            b = p.selection_breakdown
            lines.append(
                f"- [{p.citation_index}] {b.get('relevance', 0):g} / {b.get('evidence', 0):g} / {b.get('impact', 0):g} / "
                f"{b.get('recency', 0):g} / {b.get('agreement', 0):g} = {b.get('total', 0):g}; "
                f"{report.study_type_labels.get(p.study_type, p.study_type)}; {p.selection_reason}")
    return lines


def _is_prisma(req: ExportRequest) -> bool:
    return bool(req.review_protocol and req.review_protocol.framework_id == "systematic_prisma")


def prisma_flow_lines(f) -> List[str]:
    """The PRISMA 2020 flow as text lines, one per box (shared by the text exports)."""
    def reasons(d):
        return f" ({'; '.join(f'{k}: {v}' for k, v in d.items())})" if d else ""
    by_source = ", ".join(f"{k} {v}" for k, v in f.identified_by_source.items())
    lines = [
        f"Identification: {f.identified_total} records identified" + (f" ({by_source})" if by_source else "")
        + f"; {f.duplicates_removed} duplicates removed" + (
            f"; {sum(f.removed_before_screening.values())} removed before screening{reasons(f.removed_before_screening)}"
            if f.removed_before_screening else ""),
        f"Screening: {f.screened} records screened on title and abstract; {f.excluded_at_screening} excluded"
        + reasons(f.excluded_at_screening_reasons),
        f"Eligibility: {f.assessed_for_eligibility} records assessed; "
        f"{sum(f.excluded_at_eligibility.values())} excluded{reasons(f.excluded_at_eligibility)}",
        f"Included: {f.eligible} eligible; {f.included_in_synthesis or f.included} included in the synthesis "
        f"(full text retrieved for {f.full_text_retrieved})"
        + (f"; {f.beyond_study_limit} eligible beyond the {f.study_limit}-study limit" if f.beyond_study_limit else "")
        + (f"; {f.excluded_at_extraction} found not to address the question during extraction"
           if f.excluded_at_extraction else ""),
    ]
    return lines


def risk_of_bias_rows(papers) -> List[Tuple[str, List[Tuple[str, List[str], str, str]]]]:
    """Risk-of-bias judgements grouped by tool: (tool, [(study label, domain judgements, overall, reason)]), with
    the tool's domain names as the first row's header via `risk_of_bias_domains`."""
    groups: dict = {}
    for p in papers:
        rob = p.elicit.risk_of_bias
        if not rob:
            continue
        who = (p.authors[0].split()[-1] if p.authors else p.source) or ""
        groups.setdefault(rob.tool, []).append((
            f"[{p.citation_index}] {who} {p.year or ''}".strip(),
            [d.judgement for d in rob.domains], rob.overall, f"{rob.reason} ({rob.basis})".strip()))
    return list(groups.items())


def risk_of_bias_domains(papers, tool: str) -> List[str]:
    for p in papers:
        rob = p.elicit.risk_of_bias
        if rob and rob.tool == tool:
            return [d.domain for d in rob.domains]
    return []


def _section_figure_lines(sec, req, markdown: bool) -> List[str]:
    out: List[str] = []
    if sec.figure == "prisma_flow" and req.review_protocol and req.review_protocol.flow:
        out += ["", "PRISMA 2020 flow of records:" if not markdown else "**PRISMA 2020 flow of records**", ""]
        out += [f"- {ln}" for ln in prisma_flow_lines(req.review_protocol.flow)]
    if sec.figure == "risk_of_bias":
        for tool, rows in risk_of_bias_rows(req.papers):
            domains = risk_of_bias_domains(req.papers, tool)
            out += ["", f"**{tool}**" if markdown else tool]
            if markdown:
                out.append("| Study | " + " | ".join(f"D{i + 1}" for i in range(len(domains))) + " | Overall | Main concern |")
                out.append("|---" * (len(domains) + 3) + "|")
                for label, js, overall, reason in rows:
                    out.append(f"| {label} | " + " | ".join(js) + f" | **{overall}** | {reason.replace('|', '/')} |")
            else:
                for label, js, overall, reason in rows:
                    out.append(f"  {label}: " + "; ".join(f"D{i + 1} {j}" for i, j in enumerate(js))
                               + f"; overall {overall}. {reason}")
            out.append(("Domains: " if not markdown else "_Domains: ")
                       + "; ".join(f"D{i + 1} {d}" for i, d in enumerate(domains)) + ("" if not markdown else "_"))
    return out


def review_protocol_lines(protocol: Optional[ReviewProtocolReport]) -> List[str]:
    """Plain-text lines for the review method, PRISMA-style flow and source-cited rules (shared by all formats)."""
    if protocol is None:
        return []
    lines = [f"Review type: {protocol.framework_label}. Findings organisation: {protocol.organization_label}."]
    if protocol.disclosure:
        lines.append(protocol.disclosure)
    if protocol.flow and protocol.flow.assessed_for_eligibility:
        lines.append("Flow of records (PRISMA 2020):")
        lines.extend(f"- {ln}" for ln in prisma_flow_lines(protocol.flow))
    elif protocol.flow:
        f = protocol.flow
        by_source = ", ".join(f"{k} {v}" for k, v in f.identified_by_source.items())
        lines.append(
            f"Flow of records: identified {f.identified_total} ({by_source}); duplicates removed {f.duplicates_removed}; "
            f"screened {f.screened}; excluded by relevance ranking {f.excluded_at_screening}; included {f.included}."
        )
    lines.extend(f"Eligibility: {c}" for c in protocol.eligibility_criteria)
    for src, queries in protocol.search_strategies.items():
        lines.extend(f"Search ({src}): {q}" for q in queries)
    lines.append("Methodology rules applied:")
    for r in protocol.rules_applied:
        check = " [checked by code]" if r.enforcement != "prompt" else ""
        lines.append(f"- {r.stage.capitalize()}: {r.text} ({'; '.join(r.sources)}){check}")
    if protocol.framework_citations:
        lines.append("Framework sources:")
        lines.extend(f"- {c}" for c in protocol.framework_citations)
    return lines


def generate_markdown_report(req: ExportRequest) -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_label = "Deep Research Literature Review" if req.mode == "deep_research" else "Standard Evidence Synthesis"

    lines: List[str] = [
        f"# {req.query}",
        f"**Report Type:** {mode_label} | **Generated:** {now_str} | **Papers Evaluated:** {len(req.papers)}",
        "",
        "---",
        "",
        "## Abstract" if _is_prisma(req) else "## 1. Evidence Agreement & Executive Synthesis",
        f"- **Agreement Verdict:** {req.consensus.consensus_verdict or 'Synthesized'} "
        f"(Supporting: {req.consensus.supporting_pct}% | Mixed: {req.consensus.mixed_pct}% | Contradicting: {req.consensus.contradicting_pct}%)",
        f"- **Agreement Summary:** {req.consensus.consensus_summary}",
        "",
        req.executive_answer,
        "",
    ]

    if req.sections:
        lines.append("## Systematic Review (PRISMA 2020 order)" if _is_prisma(req) else "## 2. Thematic Synthesis & Detailed Analysis")
        lines.append("")
        for sec in req.sections:
            lines.append(f"### {sec.heading}")
            lines.append(sec.content)
            lines.extend(_section_figure_lines(sec, req, markdown=True))
            lines.append("")

    if req.include_evidence_table and req.papers:
        lines.append("## Structured Evidence Table (Universal Scientific Evidence Matrix)")
        lines.append("")
        if req.evidence_matrix:
            ep = req.evidence_matrix.extraction_plan
            lines.append(
                f"- **Active Domain Profile:** {ep.primary_domain} (`{ep.domain_module_activated}`) | "
                f"**Records:** {len(req.evidence_matrix.documents)} Documents, {len(req.evidence_matrix.evidence_units)} Studies, {len(req.evidence_matrix.granular_findings)} Granular Results"
            )
            lines.append("")
        lines.append("| Ref | Paper & Year | Source & Status | Stance | Methodology & Design | Dataset / System & Scale | Key Findings & Metrics | Limitations & Comparability |")
        lines.append("|---|---|---|---|---|---|---|---|")
        row_by_idx = (
            {r.citation_index: r for r in req.evidence_matrix.rows}
            if req.evidence_matrix and req.evidence_matrix.rows
            else {}
        )
        for p in req.papers:
            auth = f"{p.authors[0]} et al." if p.authors else p.source
            title_clean = p.title.replace("|", "/")[:75]
            r_obj = row_by_idx.get(p.citation_index)
            if r_obj:
                pub_st = r_obj.document.publication_status
                meth = f"{r_obj.evidence_unit.experimental_or_analytical_design}: {r_obj.evidence_unit.methodological_approach}".replace("|", "/")
                ds = f"{r_obj.evidence_unit.context_or_system} (Scale: {r_obj.evidence_unit.sample_or_scale})".replace("|", "/")
                kf = f"{r_obj.primary_finding.normalized_summary} ({len(r_obj.evidence_unit.results)} result(s))".replace("|", "/")
                lim_src = r_obj.evidence_unit.appraisal.source_reported_limitations[0] if r_obj.evidence_unit.appraisal.source_reported_limitations else p.elicit.limitations
                comp_st = r_obj.evidence_unit.comparability.status
                lim = f"{lim_src} [Comparability: {comp_st}]".replace("|", "/")
                lines.append(
                    f"| [{p.citation_index}] | **{title_clean}** ({auth}, {p.year or 'n.d.'}) | {p.source} ({pub_st}) | {p.stance} | {meth} | {ds} | {kf} | {lim} |"
                )
            else:
                meth = (p.elicit.methodology or "").replace("|", "/")
                ds = (p.elicit.dataset_or_system or "").replace("|", "/")
                kf = (p.elicit.key_findings or "").replace("|", "/")
                lim = (p.elicit.limitations or "").replace("|", "/")
                lines.append(
                    f"| [{p.citation_index}] | **{title_clean}** ({auth}, {p.year or 'n.d.'}) | {p.source} | {p.stance} | {meth} | {ds} | {kf} | {lim} |"
                )
        lines.append("")

    if req.include_gap_matrix and req.gap_matrix and (req.gap_matrix.structured_gaps or req.gap_matrix.top_opportunities):
        lines.append("## Knowledge Gap Matrix & Verified Research Priorities")
        lines.append("")
        if req.gap_matrix.corpus_adequacy_explanation:
            lines.append(f"> {req.gap_matrix.corpus_adequacy_explanation}")
            lines.append("")
        if req.gap_matrix.structured_gaps:
            lines.append("| Knowledge Gap | What Is Known | What Remains Unresolved | Evidence Basis | Verification Status | Research Direction |")
            lines.append("|---|---|---|---|---|---|")
            for sg in req.gap_matrix.structured_gaps:
                g_title = f"**[{sg.gap_id}] {sg.gap_title}** ({sg.taxonomy_code}: {sg.primary_gap_type})".replace("|", "/")
                known = (sg.current_knowledge or "").replace("|", "/")
                unres = (sg.unresolved_issue or "").replace("|", "/")
                ev_basis = (
                    f"Refs: {', '.join(f'[{i}]' for i in sg.supporting_study_indices) or '0 in selected set'} "
                    f"| Units: {', '.join(sg.supporting_evidence_unit_ids) or 'None'}"
                ).replace("|", "/")
                v_stat = f"{sg.verification_status} ({sg.priority_assessment})".replace("|", "/")
                r_dir = (sg.recommended_study_or_approach or "").replace("|", "/")
                lines.append(f"| {g_title} | {known} | {unres} | {ev_basis} | {v_stat} | {r_dir} |")
            lines.append("")
        for opp in ([] if req.gap_matrix.structured_gaps else req.gap_matrix.top_opportunities[:6]):
            badge = opp.coverage_classification
            lines.append(f"### [{badge}] {opp.method_row} × {opp.domain_col}")
            lines.append(f"- **Why Gap Exists:** {opp.why_gap_exists}")
            lines.append(f"- **Proposed Hypothesis:** {opp.proposed_hypothesis}")
            if opp.suggested_hf_models:
                lines.append(f"- **Recommended Hugging Face Models:** {', '.join(opp.suggested_hf_models)}")
            lines.append("")

    if req.include_network_summary and req.graph and req.graph.nodes:
        lines.append("## Literature Network Map Summary (Multilayer Knowledge-Relationship Engine)")
        lines.append("")
        cs = req.graph.coverage_stats
        lines.append(
            f"- **Network Topology:** {len(req.graph.nodes)} Total Nodes ({cs.included_studies_count} Included Studies, "
            f"{cs.context_publications_count} Citation-Context Papers) | {len(req.graph.edges)} Multilayer Edges"
        )
        lines.append(
            f"- **Layer Breakdown:** Layer A Bibliographic: {cs.direct_citation_edges_count} direct citations, "
            f"{cs.bibliographic_coupling_edges_count} coupling, {cs.co_citation_edges_count} co-citation | "
            f"Layer B Semantic: {cs.semantic_edges_count} | "
            f"Layer C Evidence/Claims/Gaps: {cs.evidence_edges_count} | "
            f"Layer D Evolution: {cs.evolution_edges_count}"
        )
        if req.graph.clusters:
            cluster_str = ", ".join(f"{c.get('label')} ({c.get('count')} papers)" for c in req.graph.clusters)
            lines.append(f"- **Thematic Clusters:** {cluster_str}")
        interp = req.graph.interpretation
        if interp and interp.research_themes_summary:
            lines.append(f"- **Major Research Themes:** {interp.research_themes_summary}")
        if interp and interp.intellectual_foundations_summary:
            lines.append(f"- **Foundational & Bridge Works:** {interp.intellectual_foundations_summary}")
        if interp and interp.scientific_evidence_relationships_summary:
            lines.append(f"- **Evidence Convergence & Divergence:** {interp.scientific_evidence_relationships_summary}")
        if interp and interp.network_limitations_and_disclosures:
            lines.append(f"- **Coverage & Sparsity Note:** {' '.join(interp.network_limitations_and_disclosures)}")
        paper_nodes = [n for n in req.graph.nodes if n.node_type == "paper"]
        top_hubs = sorted(paper_nodes, key=lambda x: (x.pagerank, x.citations), reverse=True)[:5]
        if top_hubs:
            lines.append("- **Key Network Hub Papers:**")
            for h in top_hubs:
                roles_str = ", ".join(h.importance_roles) if h.importance_roles else "standard_included_study"
                lines.append(
                    f"  - {h.label}: *{h.full_title}* ({h.citations} citations, Roles: `{roles_str}`, Cluster: {h.cluster_label})"
                )
        lines.append("")

    if req.include_hf_models and req.hf_resources:
        lines.append("## Matched Hugging Face Models & Datasets")
        lines.append("")
        for hf in req.hf_resources[:12]:
            lines.append(
                f"- **[{hf.id}]({hf.url})** ({hf.resource_type.upper()} | `{hf.pipeline_tag}` | Downloads: {hf.downloads:,}): {hf.description}"
            )
        lines.append("")

    if req.selection_report and req.selection_report.included:
        lines.append("## How the Studies Were Chosen")
        lines.append("")
        lines.extend(f"{ln}  " for ln in selection_report_lines(req.selection_report, req.papers))
        lines.append("")

    if req.review_protocol:
        lines.append("## Review Method & Frameworks Applied")
        lines.append("")
        lines.extend(f"{ln}  " for ln in review_protocol_lines(req.review_protocol))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## References & Bibliography")
    lines.append("")
    for bib in req.bibliography:
        lines.append(f"{bib.apa_formatted}")
        lines.append("")

    return "\n".join(lines)


def generate_txt_report(req: ExportRequest) -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_label = "DEEP RESEARCH LITERATURE REVIEW" if req.mode == "deep_research" else "STANDARD EVIDENCE SYNTHESIS"
    sep = "=" * 84
    subsep = "-" * 84

    lines: List[str] = [
        sep,
        f"CITELUME — {mode_label}",
        f"QUERY: {req.query}",
        f"GENERATED: {now_str} | PAPERS EVALUATED: {len(req.papers)}",
        sep,
        "",
        "EVIDENCE AGREEMENT & EXECUTIVE SYNTHESIS",
        subsep,
        f"Agreement Verdict : {req.consensus.consensus_verdict}",
        f"Distribution      : Supporting {req.consensus.supporting_pct}% | Mixed {req.consensus.mixed_pct}% | Contradicting {req.consensus.contradicting_pct}%",
        f"Agreement Summary : {req.consensus.consensus_summary}",
        "",
        ("ABSTRACT\n" if _is_prisma(req) else "") + req.executive_answer.replace("**", ""),
        "",
    ]

    if req.sections:
        lines.extend(["SYSTEMATIC REVIEW (PRISMA 2020 ORDER)" if _is_prisma(req) else "THEMATIC SYNTHESIS & DETAILED ANALYSIS", subsep, ""])
        for sec in req.sections:
            lines.append(f"[{sec.heading}]")
            lines.append(sec.content.replace("**", ""))
            lines.extend(_section_figure_lines(sec, req, markdown=False))
            lines.append("")

    if req.include_evidence_table and req.papers:
        lines.extend(["STRUCTURED EVIDENCE TABLE (STUDY COMPARISON MATRIX)", subsep, ""])
        for p in req.papers:
            auth = f"{p.authors[0]} et al." if p.authors else p.source
            lines.append(f"[{p.citation_index}] {p.title} ({auth}, {p.year or 'n.d.'}) [{p.source}] — Stance: {p.stance.upper()}")
            lines.append(f"    • Methodology : {p.elicit.methodology}")
            lines.append(f"    • Dataset     : {p.elicit.dataset_or_system}")
            lines.append(f"    • Key Finding : {p.elicit.key_findings}")
            lines.append(f"    • Limitations : {p.elicit.limitations}")
            lines.append("")

    if req.include_gap_matrix and req.gap_matrix and (req.gap_matrix.structured_gaps or req.gap_matrix.top_opportunities):
        lines.extend(["KNOWLEDGE GAPS", subsep, ""])
        for item in _gap_text_items(req.gap_matrix):
            lines.append(f"* {item[0]}")
            lines.extend(f"  {label:<20}: {value}" for label, value in item[1])
            lines.append("")

    if req.include_network_summary and req.graph and req.graph.nodes:
        lines.extend(["LITERATURE NETWORK MAP SUMMARY (MULTILAYER KNOWLEDGE ENGINE)", subsep, ""])
        cs = req.graph.coverage_stats
        lines.append(
            f"Nodes: {len(req.graph.nodes)} ({cs.included_studies_count} Included Studies, "
            f"{cs.context_publications_count} Context Papers) | Multilayer Edges: {len(req.graph.edges)}"
        )
        if req.graph.interpretation and req.graph.interpretation.research_themes_summary:
            lines.append(f"Streams : {req.graph.interpretation.research_themes_summary}")
        if req.graph.interpretation and req.graph.interpretation.network_limitations_and_disclosures:
            lines.append(f"Coverage: {' '.join(req.graph.interpretation.network_limitations_and_disclosures)}")
        paper_nodes = [n for n in req.graph.nodes if n.node_type == "paper"]
        top_hubs = sorted(paper_nodes, key=lambda x: (x.pagerank, x.citations), reverse=True)[:5]
        for h in top_hubs:
            lines.append(f"  • Hub {h.label}: {h.full_title} ({h.citations} citations, Cluster: {h.cluster_label})")
        lines.append("")

    if req.include_hf_models and req.hf_resources:
        lines.extend(["HUGGING FACE MODELS & DATASETS", subsep, ""])
        for hf in req.hf_resources[:12]:
            lines.append(f"* {hf.id} ({hf.resource_type.upper()} | {hf.pipeline_tag} | {hf.downloads:,} downloads): {hf.url}")
        lines.append("")

    if req.selection_report and req.selection_report.included:
        lines.extend(["HOW THE STUDIES WERE CHOSEN", subsep, ""])
        lines.extend(selection_report_lines(req.selection_report, req.papers))
        lines.append("")

    if req.review_protocol:
        lines.extend(["REVIEW METHOD & FRAMEWORKS APPLIED", subsep, ""])
        lines.extend(review_protocol_lines(req.review_protocol))
        lines.append("")

    lines.extend([sep, "REFERENCES & BIBLIOGRAPHY", sep, ""])
    for bib in req.bibliography:
        lines.append(bib.apa_formatted)
        lines.append("")

    return "\n".join(lines)


def _gap_text_items(gm) -> list:
    """Knowledge gaps as (title, [(label, value), ...]) for the text and PDF reports: the evidence-derived gap
    records, or - when there are none - the thin cells of the question-specific grid."""
    items = []
    for sg in gm.structured_gaps:
        refs = ", ".join(f"[{i}]" for i in sg.supporting_study_indices) or "none in the selected studies"
        items.append((
            f"[{sg.gap_id}] {sg.gap_title} ({sg.taxonomy_code}: {sg.primary_gap_type})",
            [("What is known", sg.current_knowledge), ("What is unresolved", sg.unresolved_issue),
             ("Why it matters", sg.why_it_matters), ("Studies", refs),
             ("Verification", f"{sg.verification_status}; {sg.priority_assessment}"),
             ("Research direction", sg.recommended_study_or_approach)],
        ))
    if not items:
        for opp in gm.top_opportunities[:8]:
            items.append((
                f"{opp.method_row} x {opp.domain_col} ({opp.paper_count} stud{'y' if opp.paper_count == 1 else 'ies'})",
                [("Why it is a gap", opp.why_gap_exists), ("Research direction", opp.proposed_hypothesis)],
            ))
    return [(t, [(k, v) for k, v in f if v]) for t, f in items]
