"""
PDF export built with ReportLab: justified body text, the evidence table and knowledge-gap tables as real tables on
landscape pages, the gap coverage grid, and a drawing of the literature map.

Everything shown comes from the export request: the PDF shows the same results as the web page.
"""
from __future__ import annotations

import datetime
import html
import io
import math
import os
import re
from typing import List, Optional

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from models import ExportRequest
from services.fulltext import repair_ligatures
from services.export_service import _gap_text_items, review_protocol_lines, selection_report_lines

# ---------------------------------------------------------------------------------------------------------------
# Fonts: a Unicode TrueType font when one is installed (so Greek letters, ≥, · and accented names print), otherwise
# the built-in Helvetica with characters it cannot show replaced.
# ---------------------------------------------------------------------------------------------------------------
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", None),
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", None),
]
_FONTS: Optional[tuple] = None


def _fonts() -> tuple:
    """(regular, bold, unicode_capable) font names, registered once."""
    global _FONTS
    if _FONTS is not None:
        return _FONTS
    for regular, bold in _FONT_CANDIDATES:
        if not os.path.exists(regular):
            continue
        try:
            pdfmetrics.registerFont(TTFont("CLSans", regular))
            if bold and os.path.exists(bold):
                pdfmetrics.registerFont(TTFont("CLSans-Bold", bold))
            else:
                pdfmetrics.registerFont(TTFont("CLSans-Bold", regular))
            pdfmetrics.registerFontFamily("CLSans", normal="CLSans", bold="CLSans-Bold",
                                          italic="CLSans", boldItalic="CLSans-Bold")
            _FONTS = ("CLSans", "CLSans-Bold", True)
            return _FONTS
        except Exception:
            continue
    _FONTS = ("Helvetica", "Helvetica-Bold", False)
    return _FONTS


_LATIN1_REPLACEMENTS = {
    "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "•": "*",
    "→": "->", "≤": "<=", "≥": ">=", "−": "-", "·": ".", " ": " ", " ": " ",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "μ": "u", "…": "...",
}


def _clean(text) -> str:
    """Plain text made safe for a Paragraph (escaped markup, characters the font cannot show replaced)."""
    s = str(text or "")
    s = repair_ligatures(s)  # ligatures copied from PDFs are missing from most fonts
    if not _fonts()[2]:
        for k, v in _LATIN1_REPLACEMENTS.items():
            s = s.replace(k, v)
        s = s.encode("latin-1", errors="replace").decode("latin-1")
    return html.escape(s, quote=False)


def _rich(text) -> str:
    """Model-written text: escape it, then keep **bold** and line breaks."""
    s = _clean(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"<b>\1</b>", s)
    return s


# ---------------------------------------------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------------------------------------------
INK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
ACCENT = colors.HexColor("#4338ca")
RULE = colors.HexColor("#cbd5e1")
HEAD_BG = colors.HexColor("#eef2ff")
ZEBRA = colors.HexColor("#f8fafc")

CLUSTER_COLOURS = ["#4f46e5", "#0891b2", "#16a34a", "#d97706", "#db2777", "#7c3aed", "#dc2626", "#0d9488",
                   "#65a30d", "#c2410c"]


def _styles() -> dict:
    reg, bold, _ = _fonts()
    base = dict(fontName=reg, textColor=INK)
    return {
        "title": ParagraphStyle("title", fontName=bold, fontSize=17, leading=21, textColor=ACCENT, spaceAfter=4),
        "query": ParagraphStyle("query", fontName=bold, fontSize=12.5, leading=16, textColor=INK, spaceAfter=4),
        "meta": ParagraphStyle("meta", fontName=reg, fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=8),
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=13.5, leading=17, textColor=ACCENT,
                             spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontSize=10, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "small": ParagraphStyle("small", fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY, spaceAfter=3, **base),
        "note": ParagraphStyle("note", fontName=reg, fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=4,
                               alignment=TA_JUSTIFY),
        "bullet": ParagraphStyle("bullet", fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY, leftIndent=10,
                                 bulletIndent=0, spaceAfter=2, **base),
        "ref": ParagraphStyle("ref", fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY, leftIndent=14,
                              firstLineIndent=-14, spaceAfter=4, **base),
        "cell": ParagraphStyle("cell", fontSize=7.3, leading=9.2, alignment=TA_LEFT, **base),
        # Table cells are left-aligned: justifying narrow columns leaves wide gaps between words
        "cellj": ParagraphStyle("cellj", fontSize=7.3, leading=9.2, alignment=TA_LEFT, **base),
        "cellh": ParagraphStyle("cellh", fontName=bold, fontSize=7.5, leading=9.4, textColor=INK),
        "cellc": ParagraphStyle("cellc", fontName=bold, fontSize=7.5, leading=9.4, textColor=INK, alignment=1),
    }


def _table_style(header_rows: int = 1) -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, ACCENT),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, ZEBRA]),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])


# ---------------------------------------------------------------------------------------------------------------
# Page templates: portrait for text, landscape for the wide tables and the map
# ---------------------------------------------------------------------------------------------------------------
MARGIN = 16 * mm


def _footer(canvas, doc):
    canvas.saveState()
    reg = _fonts()[0]
    canvas.setFont(reg, 7.5)
    canvas.setFillColor(MUTED)
    w, _h = canvas._pagesize
    canvas.drawString(MARGIN, 9 * mm, "CiteLume literature synthesis")
    canvas.drawRightString(w - MARGIN, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _doc(buffer) -> BaseDocTemplate:
    doc = BaseDocTemplate(buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                          bottomMargin=MARGIN + 4 * mm, title="CiteLume literature synthesis", author="CiteLume")
    pw, ph = A4
    lw, lh = landscape(A4)
    portrait = Frame(MARGIN, MARGIN + 4 * mm, pw - 2 * MARGIN, ph - 2 * MARGIN - 4 * mm, id="p",
                     leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    wide = Frame(MARGIN, MARGIN + 4 * mm, lw - 2 * MARGIN, lh - 2 * MARGIN - 4 * mm, id="l",
                 leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="portrait", frames=[portrait], pagesize=A4, onPage=_footer),
        PageTemplate(id="landscape", frames=[wide], pagesize=landscape(A4), onPage=_footer),
    ])
    return doc


PORTRAIT_W = A4[0] - 2 * MARGIN
LANDSCAPE_W = landscape(A4)[0] - 2 * MARGIN
LANDSCAPE_H = landscape(A4)[1] - 2 * MARGIN - 4 * mm


# ---------------------------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------------------------
def _paragraphs(text: str, style) -> list:
    out = []
    for block in re.split(r"\n\s*\n", str(text or "").strip()):
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if all(re.match(r"^([-*•]|\d+[.)])\s+", ln) for ln in lines):
            for ln in lines:
                out.append(Paragraph(_rich(re.sub(r"^([-*•]|\d+[.)])\s+", "", ln)), style, bulletText="•"))
        else:
            out.append(Paragraph(_rich(" ".join(lines)), style))
    return out


def _agreement_table(req: ExportRequest, st: dict) -> Table:
    c = req.consensus
    data = [
        [Paragraph("Verdict", st["cellh"]), Paragraph("Supporting", st["cellh"]), Paragraph("Mixed", st["cellh"]),
         Paragraph("Contradicting", st["cellh"])],
        [Paragraph(_clean(c.consensus_verdict or "Synthesised"), st["cell"]),
         Paragraph(f"{c.supporting_pct}%", st["cell"]), Paragraph(f"{c.mixed_pct}%", st["cell"]),
         Paragraph(f"{c.contradicting_pct}%", st["cell"])],
    ]
    t = Table(data, colWidths=[PORTRAIT_W * 0.46] + [PORTRAIT_W * 0.18] * 3)
    t.setStyle(_table_style())
    return t


def _evidence_rows(req: ExportRequest, st: dict) -> list:
    row_by_idx = {r.citation_index: r for r in (req.evidence_matrix.rows if req.evidence_matrix else [])}
    rows = []
    for p in req.papers:
        auth = f"{p.authors[0]} et al." if len(p.authors) > 1 else (p.authors[0] if p.authors else "")
        study = f"<b>{_clean(p.title)}</b><br/>{_clean(auth)} ({p.year or 'n.d.'})"
        venue = p.venue or p.source
        if venue:
            study += f"<br/><font color='#64748b'>{_clean(venue)}</font>"
        r = row_by_idx.get(p.citation_index)
        if r is not None:
            eu = r.evidence_unit
            design = f"{eu.experimental_or_analytical_design}"
            if eu.methodological_approach:
                design += f": {eu.methodological_approach}"
            sample = " ".join(x for x in [eu.context_or_system, f"(n/scale: {eu.sample_or_scale})" if eu.sample_or_scale else ""] if x)
            # The written finding and limitation (as on the web page's study cards) read better than the
            # machine summary of extracted numbers, which is used only when they are missing.
            finding = p.elicit.key_findings or r.primary_finding.normalized_summary
            lims = eu.appraisal.source_reported_limitations
            limitation = p.elicit.limitations or (lims[0] if lims else "")
        else:
            design = p.elicit.methodology
            sample = p.elicit.dataset_or_system
            finding = p.elicit.key_findings
            limitation = p.elicit.limitations
        stype = (req.selection_report.study_type_labels.get(p.study_type, p.study_type)
                 if req.selection_report and p.study_type else (p.study_type or ""))
        if stype:
            design = f"<b>{_clean(stype)}</b><br/>{_clean(design)}"
        else:
            design = _clean(design)
        rows.append([
            Paragraph(f"[{p.citation_index}]", st["cellh"]),
            Paragraph(study, st["cell"]),
            Paragraph(design, st["cell"]),
            Paragraph(_clean(sample), st["cell"]),
            Paragraph(_rich(finding), st["cellj"]),
            Paragraph(_rich(limitation), st["cellj"]),
            Paragraph(_clean((p.stance or "").capitalize()), st["cell"]),
        ])
    return rows


def _evidence_table(req: ExportRequest, st: dict) -> list:
    head = ["Ref", "Study", "Study type & design", "Population / system & scale", "Key findings",
            "Limitations", "Stance"]
    data = [[Paragraph(h, st["cellh"]) for h in head]] + _evidence_rows(req, st)
    fr = [0.04, 0.19, 0.14, 0.13, 0.26, 0.165, 0.075]
    t = Table(data, colWidths=[LANDSCAPE_W * f for f in fr], repeatRows=1)
    t.setStyle(_table_style())
    return [Paragraph("Evidence Table", st["h1"]),
            Paragraph(f"{len(req.papers)} studies. Findings and limitations are taken from each study's abstract "
                      f"or, where it could be read, its full text.", st["note"]), t]


def _coverage_grid(gm, st: dict) -> list:
    if not (gm.rows and gm.cols and gm.cells):
        return []
    n_cols = len(gm.cols)
    first = 0.22
    col_w = (1 - first) / max(n_cols, 1)
    data = [[Paragraph(_clean(gm.row_dimension_title or ""), st["cellh"])] +
            [Paragraph(_clean(c), st["cellc"]) for c in gm.cols]]
    styles = _table_style()
    for i, row_label in enumerate(gm.rows):
        line = [Paragraph(f"<b>{_clean(row_label)}</b>", st["cell"])]
        for j in range(n_cols):
            cell = gm.cells[i][j] if i < len(gm.cells) and j < len(gm.cells[i]) else None
            n = cell.paper_count if cell else 0
            refs = ", ".join(str(x) for x in (cell.paper_indices if cell else [])[:8])
            txt = f"<b>{n}</b>" + (f"<br/><font size='6.3' color='#475569'>[{_clean(refs)}]</font>" if refs else "")
            line.append(Paragraph(txt, st["cellc"]))
            shade = "#fee2e2" if n == 0 else ("#fef3c7" if n == 1 else ("#dcfce7" if n < 4 else "#bbf7d0"))
            styles.add("BACKGROUND", (j + 1, i + 1), (j + 1, i + 1), colors.HexColor(shade))
        data.append(line)
    t = Table(data, colWidths=[LANDSCAPE_W * first] + [LANDSCAPE_W * col_w] * n_cols, repeatRows=1)
    t.setStyle(styles)
    return [CondPageBreak(130), Paragraph("Evidence coverage grid", st["h2"]),
            Paragraph(f"Rows: {_clean(gm.row_dimension_title)}. Columns: {_clean(gm.col_dimension_title)}. Each cell "
                      f"shows the number of included studies and their reference numbers; red cells have none, amber "
                      f"cells only one.", st["note"]), t]


def _gap_section(req: ExportRequest, st: dict) -> list:
    gm = req.gap_matrix
    out = [Paragraph("Knowledge Gaps", st["h1"])]
    if gm.corpus_adequacy_explanation:
        out.append(Paragraph(_rich(gm.corpus_adequacy_explanation), st["note"]))
    if gm.structured_gaps:
        head = ["Gap", "What is known", "What remains unresolved", "Why it matters", "Studies", "Suggested research"]
        data = [[Paragraph(h, st["cellh"]) for h in head]]
        for sg in gm.structured_gaps:
            refs = ", ".join(f"[{i}]" for i in sg.supporting_study_indices) or "none in the selected studies"
            title = f"<b>{_clean(sg.gap_title)}</b><br/><font color='#64748b'>{_clean(sg.primary_gap_type)}; " \
                    f"{_clean(sg.verification_status)}</font>"
            data.append([
                Paragraph(title, st["cell"]),
                Paragraph(_rich(sg.current_knowledge), st["cellj"]),
                Paragraph(_rich(sg.unresolved_issue), st["cellj"]),
                Paragraph(_rich(sg.why_it_matters), st["cellj"]),
                Paragraph(_clean(refs), st["cell"]),
                Paragraph(_rich(sg.recommended_study_or_approach), st["cellj"]),
            ])
        fr = [0.17, 0.19, 0.19, 0.16, 0.08, 0.21]
        t = Table(data, colWidths=[LANDSCAPE_W * f for f in fr], repeatRows=1)
        t.setStyle(_table_style())
        out.append(t)
    else:
        for title, fields in _gap_text_items(gm):
            out.append(Paragraph(f"<b>{_clean(title)}</b>", st["small"]))
            for label, value in fields:
                out.append(Paragraph(f"<b>{_clean(label)}:</b> {_rich(value)}", st["bullet"], bulletText="•"))
    grid = _coverage_grid(gm, st)
    if grid:
        out.append(Spacer(1, 8))
        out.extend(grid)
    return out


def _map_drawing(req: ExportRequest, width: float, height: float) -> Optional[Drawing]:
    """The literature map: studies placed by a force layout of their links, coloured by cluster."""
    g = req.graph
    nodes = [n for n in g.nodes if n.node_type == "paper"]
    if len(nodes) < 2:
        return None
    try:
        import networkx as nx
    except Exception:
        return None
    ids = {n.id for n in nodes}
    G = nx.Graph()
    G.add_nodes_from(ids)
    edges = [e for e in g.edges if e.source in ids and e.target in ids and e.source != e.target]
    edges = sorted(edges, key=lambda e: e.normalized_weight or e.weight or 0, reverse=True)[:400]
    for e in edges:
        w = float(e.normalized_weight or e.weight or 0.3)
        if G.has_edge(e.source, e.target):
            G[e.source][e.target]["weight"] = max(G[e.source][e.target]["weight"], w)
        else:
            G.add_edge(e.source, e.target, weight=w)
    pos = nx.spring_layout(G, seed=7, weight="weight", k=1.6 / math.sqrt(max(len(ids), 1)), iterations=200)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 22
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = (width - 2 * pad) / ((x1 - x0) or 1)
    sy = (height - 2 * pad) / ((y1 - y0) or 1)

    def at(nid):
        x, y = pos[nid]
        return pad + (x - x0) * sx, pad + (y - y0) * sy

    reg, bold, _ = _fonts()
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#fbfcfe"), strokeColor=RULE, strokeWidth=0.5))
    for u, v, data in G.edges(data=True):
        (ax, ay), (bx, by) = at(u), at(v)
        d.add(Line(ax, ay, bx, by, strokeColor=colors.HexColor("#94a3b8"),
                   strokeWidth=0.3 + 1.2 * float(data.get("weight", 0.3)), strokeOpacity=0.45))
    max_c = max((n.citations for n in nodes), default=0) or 1
    for n in nodes:
        x, y = at(n.id)
        r = 4 + 7 * math.sqrt(max(n.citations, 0) / max_c)
        colour = CLUSTER_COLOURS[(n.cluster_id or 0) % len(CLUSTER_COLOURS)]
        context = n.is_citation_context_only or n.citation_index is None
        d.add(Circle(x, y, r * (0.7 if context else 1), fillColor=colors.HexColor("#cbd5e1" if context else colour),
                     strokeColor=colors.white, strokeWidth=0.8))
        if not context:
            d.add(String(x, y - 2.4, str(n.citation_index), fontName=bold, fontSize=6.3, fillColor=colors.white,
                         textAnchor="middle"))
    return d


def _map_section(req: ExportRequest, st: dict) -> list:
    g = req.graph
    cs = g.coverage_stats
    out = [Paragraph("Literature Map", st["h1"]),
           Paragraph(f"{len(g.nodes)} nodes ({cs.included_studies_count} included studies, "
                     f"{cs.context_publications_count} citation-context papers) and {len(g.edges)} links. Circles are "
                     f"studies numbered as in the reference list, sized by citations and coloured by cluster; lines "
                     f"join studies that cite each other, share references or address similar questions. Grey "
                     f"circles are papers outside the included set.", st["note"])]
    drawing = _map_drawing(req, LANDSCAPE_W, LANDSCAPE_H * 0.66)
    if drawing is not None:
        out.append(drawing)
    clusters = {}
    for n in g.nodes:
        if n.node_type == "paper" and n.citation_index is not None and not n.is_citation_context_only:
            clusters.setdefault((n.cluster_id or 0, n.cluster_label or f"Cluster {n.cluster_id}"), []).append(
                n.citation_index)
    if clusters:
        cells = []
        for (cid, label), refs in sorted(clusters.items()):
            colour = CLUSTER_COLOURS[cid % len(CLUSTER_COLOURS)]
            refs_txt = ", ".join(str(r) for r in sorted(refs))
            cells.append(Paragraph(f"<font color='{colour}'>●</font> <b>{_clean(label)}</b>: [{refs_txt}]",
                                   st["cell"]))
        per_row = 3
        rows = [cells[i:i + per_row] + [""] * (per_row - len(cells[i:i + per_row]))
                for i in range(0, len(cells), per_row)]
        t = Table(rows, colWidths=[LANDSCAPE_W / per_row] * per_row)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        out += [Spacer(1, 4), t]
    return out


def _map_text(req: ExportRequest, st: dict) -> list:
    g = req.graph
    interp = g.interpretation
    out = []
    for label, value in [
        ("Research themes", interp.research_themes_summary if interp else ""),
        ("Foundational and bridging works", interp.intellectual_foundations_summary if interp else ""),
        ("Where the evidence agrees and differs", interp.scientific_evidence_relationships_summary if interp else ""),
    ]:
        if value:
            out.append(Paragraph(f"<b>{label}.</b> {_rich(value)}", st["small"]))
    paper_nodes = [n for n in g.nodes if n.node_type == "paper"]
    hubs = sorted(paper_nodes, key=lambda x: (x.pagerank, x.citations), reverse=True)[:5]
    if hubs:
        out.append(Paragraph("<b>Most connected studies</b>", st["small"]))
        for h in hubs:
            ref = f"[{h.citation_index}] " if h.citation_index is not None else ""
            out.append(Paragraph(f"{ref}{_clean(h.full_title)} ({h.citations} citations; {_clean(h.cluster_label)})",
                                 st["bullet"], bulletText="•"))
    if interp and interp.network_limitations_and_disclosures:
        out.append(Paragraph(_clean(" ".join(interp.network_limitations_and_disclosures)), st["note"]))
    return out


def _lines_section(title: str, lines: List[str], st: dict) -> list:
    out = [Paragraph(title, st["h1"])]
    for ln in lines:
        if ln.startswith("- "):
            out.append(Paragraph(_rich(ln[2:]), st["bullet"], bulletText="•"))
        else:
            out.append(Paragraph(_rich(ln), st["small"]))
    return out


# ---------------------------------------------------------------------------------------------------------------
def generate_pdf_report(req: ExportRequest) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = _doc(buf)
    mode_label = "Deep Research literature review" if req.mode == "deep_research" else "Standard evidence synthesis"
    now_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

    story: list = [
        Paragraph("CiteLume Literature Synthesis", st["title"]),
        Paragraph(_clean(req.query), st["query"]),
        Paragraph(f"{mode_label} · {len(req.papers)} studies · generated {now_str}", st["meta"]),
        Paragraph("Summary of the Evidence", st["h1"]),
        _agreement_table(req, st),
        Spacer(1, 6),
    ]
    if req.consensus.consensus_summary:
        story.append(Paragraph(_rich(req.consensus.consensus_summary), st["small"]))
    story += _paragraphs(req.executive_answer, st["body"])

    if req.sections:
        story.append(Paragraph("Synthesis", st["h1"]))
        for sec in req.sections:
            story.append(CondPageBreak(60))
            story.append(Paragraph(_clean(sec.heading), st["h2"]))
            story += _paragraphs(sec.content, st["body"])

    wide_parts = []
    if req.include_evidence_table and req.papers:
        wide_parts.append(_evidence_table(req, st))
    if req.include_gap_matrix and req.gap_matrix and (
            req.gap_matrix.structured_gaps or req.gap_matrix.top_opportunities or req.gap_matrix.cells):
        wide_parts.append(_gap_section(req, st))
    if req.include_network_summary and req.graph and req.graph.nodes:
        wide_parts.append(_map_section(req, st))
    if wide_parts:
        story += [NextPageTemplate("landscape"), PageBreak()]
        for i, part in enumerate(wide_parts):
            if i:
                story.append(PageBreak())
            story += part
        story += [NextPageTemplate("portrait"), PageBreak()]

    if req.include_network_summary and req.graph and req.graph.nodes:
        text = _map_text(req, st)
        if text:
            story += [Paragraph("Literature Map: Interpretation", st["h1"])] + text

    if req.include_hf_models and req.hf_resources:
        story.append(Paragraph("Matched Hugging Face Models & Datasets", st["h1"]))
        for hf in req.hf_resources[:12]:
            story.append(Paragraph(
                f"<b>{_clean(hf.id)}</b> ({_clean(hf.resource_type)}, {_clean(hf.pipeline_tag)}, "
                f"{hf.downloads:,} downloads): {_rich(hf.description)}", st["bullet"], bulletText="•"))

    if req.selection_report and req.selection_report.included:
        story += _lines_section("How the Studies Were Chosen", selection_report_lines(req.selection_report, req.papers), st)
    if req.review_protocol:
        story += _lines_section("Review Method", review_protocol_lines(req.review_protocol), st)

    if req.bibliography:
        story.append(Paragraph("References", st["h1"]))
        for bib in req.bibliography:
            story.append(Paragraph(_rich(bib.apa_formatted), st["ref"]))

    doc.build(story)
    return buf.getvalue()
