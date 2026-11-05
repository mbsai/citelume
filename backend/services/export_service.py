import datetime
import textwrap
from typing import List, Optional, Tuple
from models import ExportRequest, ReviewProtocolReport


def review_protocol_lines(protocol: Optional[ReviewProtocolReport]) -> List[str]:
    """Plain-text lines for the review method, PRISMA-style flow and source-cited rules (shared by all formats)."""
    if protocol is None:
        return []
    lines = [f"Review type: {protocol.framework_label}. Findings organisation: {protocol.organization_label}."]
    if protocol.disclosure:
        lines.append(protocol.disclosure)
    if protocol.flow:
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
        "## 1. Evidence Agreement & Executive Synthesis",
        f"- **Agreement Verdict:** {req.consensus.consensus_verdict or 'Synthesized'} "
        f"(Supporting: {req.consensus.supporting_pct}% | Mixed: {req.consensus.mixed_pct}% | Contradicting: {req.consensus.contradicting_pct}%)",
        f"- **Agreement Summary:** {req.consensus.consensus_summary}",
        "",
        req.executive_answer,
        "",
    ]

    if req.sections:
        lines.append("## 2. Thematic Synthesis & Detailed Analysis")
        lines.append("")
        for sec in req.sections:
            lines.append(f"### {sec.heading}")
            lines.append(sec.content)
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
        for opp in req.gap_matrix.top_opportunities[:6]:
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
        req.executive_answer,
        "",
    ]

    if req.sections:
        lines.extend(["THEMATIC SYNTHESIS & DETAILED ANALYSIS", subsep, ""])
        for sec in req.sections:
            lines.append(f"[{sec.heading}]")
            lines.append(sec.content)
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

    if req.include_gap_matrix and req.gap_matrix and req.gap_matrix.top_opportunities:
        lines.extend(["KNOWLEDGE GAP MATRIX — RESEARCH OPPORTUNITIES", subsep, ""])
        for opp in req.gap_matrix.top_opportunities[:8]:
            lines.append(f"* {opp.method_row}  x  {opp.domain_col} ({opp.status.upper()}: {opp.paper_count} papers)")
            lines.append(f"  Why Gap Exists      : {opp.why_gap_exists}")
            lines.append(f"  Proposed Hypothesis : {opp.proposed_hypothesis}")
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

    if req.review_protocol:
        lines.extend(["REVIEW METHOD & FRAMEWORKS APPLIED", subsep, ""])
        lines.extend(review_protocol_lines(req.review_protocol))
        lines.append("")

    lines.extend([sep, "REFERENCES & BIBLIOGRAPHY", sep, ""])
    for bib in req.bibliography:
        lines.append(bib.apa_formatted)
        lines.append("")

    return "\n".join(lines)


def _pdf_escape(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u00d7": "x",
        "\u2192": "->",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_pdf_report(req: ExportRequest) -> bytes:
    """
    Generates a clean, multi-page PDF 1.4 document with Helvetica & Helvetica-Bold fonts,
    honoring the user's selected section checkboxes and ending with the full Bibliography.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_label = "Deep Research Literature Review" if req.mode == "deep_research" else "Standard Evidence Synthesis"

    styled_lines: List[Tuple[str, int, int, str]] = []

    def add_heading(title: str, size: int = 13):
        styled_lines.append(("F1", 6, 8, ""))
        for wline in textwrap.wrap(title, width=78):
            styled_lines.append(("F2", size, size + 5, wline))
        styled_lines.append(("F1", 4, 6, ""))

    def add_paragraph(text: str, font: str = "F1", size: int = 10, width: int = 92):
        for raw_para in (text or "").split("\n"):
            raw_para = raw_para.strip()
            if not raw_para:
                styled_lines.append(("F1", 5, 7, ""))
                continue
            for wline in textwrap.wrap(raw_para, width=width):
                styled_lines.append((font, size, size + 4, wline))

    styled_lines.append(("F2", 15, 20, "CITELUME - SYNTHESIS REPORT"))
    for qline in textwrap.wrap(f"Query: {req.query}", width=76):
        styled_lines.append(("F2", 12, 16, qline))
    styled_lines.append(
        ("F1", 9, 14, f"Mode: {mode_label} | Generated: {now_str} | Corpus: {len(req.papers)} Cited Papers")
    )
    styled_lines.append(("F1", 9, 13, "-" * 96))

    add_heading("Evidence Agreement & Executive Synthesis", 12)
    add_paragraph(
        f"Agreement Verdict: {req.consensus.consensus_verdict} "
        f"(Supporting: {req.consensus.supporting_pct}% | Mixed: {req.consensus.mixed_pct}% | Contradicting: {req.consensus.contradicting_pct}%)",
        font="F2",
        size=10,
    )
    add_paragraph(f"Summary: {req.consensus.consensus_summary}", font="F1", size=10)
    styled_lines.append(("F1", 5, 7, ""))
    add_paragraph(req.executive_answer, font="F1", size=10)

    if req.sections:
        add_heading("Thematic Analysis & Literature Review", 12)
        for sec in req.sections:
            add_heading(sec.heading, 11)
            add_paragraph(sec.content, font="F1", size=10)

    if req.include_evidence_table and req.papers:
        add_heading("Structured Evidence Table (Study Comparison Matrix)", 12)
        for p in req.papers:
            auth = f"{p.authors[0]} et al." if p.authors else p.source
            add_paragraph(
                f"[{p.citation_index}] {p.title} ({auth}, {p.year or 'n.d.'}, {p.source}) - Stance: {p.stance.upper()}",
                font="F2",
                size=9,
                width=96,
            )
            add_paragraph(
                f"    Methodology: {p.elicit.methodology} | Dataset: {p.elicit.dataset_or_system}",
                font="F1",
                size=9,
                width=96,
            )
            add_paragraph(f"    Key Finding: {p.elicit.key_findings}", font="F1", size=9, width=96)
            add_paragraph(f"    Limitations: {p.elicit.limitations}", font="F1", size=9, width=96)
            styled_lines.append(("F1", 3, 5, ""))

    if req.include_gap_matrix and req.gap_matrix and req.gap_matrix.top_opportunities:
        add_heading("Knowledge Gap Matrix & White-Space Opportunities", 12)
        for opp in req.gap_matrix.top_opportunities[:8]:
            add_paragraph(
                f"* [{opp.status.upper()}] {opp.method_row} x {opp.domain_col} ({opp.paper_count} papers)",
                font="F2",
                size=9,
                width=95,
            )
            add_paragraph(f"  Why Gap Exists: {opp.why_gap_exists}", font="F1", size=9, width=95)
            add_paragraph(f"  Proposed Hypothesis: {opp.proposed_hypothesis}", font="F1", size=9, width=95)
            styled_lines.append(("F1", 3, 5, ""))

    if req.include_network_summary and req.graph and req.graph.nodes:
        add_heading("Literature Network Map Summary", 12)
        add_paragraph(
            f"Network Topology: {len(req.graph.nodes)} Total Nodes | {len(req.graph.edges)} Citation & Semantic Similarity Edges",
            font="F2",
            size=9,
        )
        paper_nodes = [n for n in req.graph.nodes if n.node_type == "paper"]
        top_hubs = sorted(paper_nodes, key=lambda x: (x.pagerank, x.citations), reverse=True)[:5]
        for h in top_hubs:
            add_paragraph(
                f"  * Hub {h.label}: {h.full_title} ({h.citations} citations, Cluster: {h.cluster_label})",
                font="F1",
                size=9,
                width=95,
            )
        styled_lines.append(("F1", 3, 5, ""))

    if req.include_hf_models and req.hf_resources:
        add_heading("Matched Hugging Face Models & Datasets", 12)
        for hf in req.hf_resources[:10]:
            add_paragraph(
                f"* {hf.id} ({hf.resource_type.upper()} - {hf.pipeline_tag}, {hf.downloads:,} downloads): {hf.description}",
                font="F1",
                size=9,
                width=95,
            )

    if req.review_protocol:
        add_heading("Review Method & Frameworks Applied", 12)
        for ln in review_protocol_lines(req.review_protocol):
            add_paragraph(ln, font="F1", size=9, width=96)
        styled_lines.append(("F1", 3, 5, ""))

    add_heading("Complete References & Bibliography", 12)
    for bib in req.bibliography:
        add_paragraph(bib.apa_formatted, font="F1", size=9, width=96)
        styled_lines.append(("F1", 3, 5, ""))

    pages_streams: List[str] = []
    y_top = 742
    y_bottom = 52
    curr_y = y_top
    curr_cmds: List[str] = []

    for font_key, fsize, spacing, text in styled_lines:
        if curr_y - spacing < y_bottom:
            pages_streams.append("\n".join(curr_cmds))
            curr_cmds = []
            curr_y = y_top
        curr_y -= spacing
        if text:
            escaped = _pdf_escape(text)
            curr_cmds.append(f"BT /{font_key} {fsize} Tf 48 {curr_y} Td ({escaped}) Tj ET")

    if curr_cmds or not pages_streams:
        pages_streams.append("\n".join(curr_cmds))

    objects: List[bytes] = []
    num_pages = len(pages_streams)
    page_obj_ids = [5 + 2 * i for i in range(num_pages)]

    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n".encode("latin-1"))
    objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n")
    objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>\nendobj\n")

    for i, stream_str in enumerate(pages_streams):
        page_id = 5 + 2 * i
        content_id = page_id + 1
        footer_cmd = f"\nBT /F1 8 Tf 245 28 Td (Page {i + 1} of {num_pages} - CiteLume) Tj ET"
        stream_bytes = (stream_str + footer_cmd).encode("latin-1", errors="replace")
        page_obj = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {content_id} 0 R >>\nendobj\n"
        ).encode("latin-1")
        content_obj = (
            f"{content_id} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )
        objects.append(page_obj)
        objects.append(content_obj)

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets: List[int] = []
    body = bytearray(header)
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    total_objs = len(objects) + 1
    xref_lines = [f"xref\n0 {total_objs}\n0000000000 65535 f \n"]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n")
    trailer = (
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    body.extend("".join(xref_lines).encode("latin-1"))
    body.extend(trailer.encode("latin-1"))
    return bytes(body)
