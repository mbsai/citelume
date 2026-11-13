import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader

from models import (
    ElicitExtraction,
    ExportRequest,
    GapOpportunity,
    GraphEdge,
    GraphNode,
    KnowledgeGapMatrix,
    LiteratureGraph,
    PaperRecord,
    StructuredKnowledgeGap,
    SynthesisSection,
)
from services.fulltext import repair_ligatures
from services.pdf_report import generate_pdf_report


def _papers():
    return [
        PaperRecord(
            id=f"p{i}", title=f"Trial number {i} of vitamin D and fractures", authors=["Ann Author", "B Other"],
            year=2020 + i, source="PubMed", citation_index=i, stance="supporting",
            elicit=ElicitExtraction(methodology="Randomised trial", dataset_or_system="Older adults",
                                    key_findings=f"Finding sentence for study {i}.",
                                    limitations=f"Limitation sentence for study {i}."),
        )
        for i in range(1, 5)
    ]


def _request(**overrides):
    papers = _papers()
    cell = lambda r, c, idx: GapOpportunity(id=f"{r}{c}", method_row=r, domain_col=c, paper_count=len(idx),
                                            paper_indices=idx)
    gm = KnowledgeGapMatrix(
        rows=["Daily dosing", "Monthly bolus"], cols=["Hip fracture", "Any fracture"],
        cells=[[cell("Daily dosing", "Hip fracture", [1, 2]), cell("Daily dosing", "Any fracture", [])],
               [cell("Monthly bolus", "Hip fracture", [3]), cell("Monthly bolus", "Any fracture", [4])]],
        structured_gaps=[StructuredKnowledgeGap(
            gap_id="G1", gap_title="Optimal dose remains unclear", gap_statement="Dose unclear",
            current_knowledge="Doses varied.", unresolved_issue="Which dose works best.",
            supporting_study_indices=[1, 3])],
    )
    nodes = [GraphNode(id=p.id, label=f"[{p.citation_index}]", full_title=p.title, node_type="paper",
                       citation_index=p.citation_index, citations=10 * p.citation_index,
                       cluster_id=p.citation_index % 2, cluster_label=f"Theme {p.citation_index % 2}")
             for p in papers]
    edges = [GraphEdge(id=f"e{i}", source=f"p{i}", target=f"p{i + 1}") for i in range(1, 4)]
    data = dict(
        format="pdf", query="Does vitamin D reduce fractures in older adults?",
        executive_answer="Vitamin D alone does not reduce fractures [1]. Combined with calcium it may [2].",
        sections=[SynthesisSection(heading="Thematic synthesis", content="Body text of the section [3].")],
        papers=papers, gap_matrix=gm, graph=LiteratureGraph(nodes=nodes, edges=edges),
    )
    data.update(overrides)
    return ExportRequest(**data)


def _text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


class PdfReportTests(unittest.TestCase):
    def test_all_result_views_are_in_the_pdf(self):
        pdf = generate_pdf_report(_request())
        self.assertTrue(pdf.startswith(b"%PDF"))
        text = _text(pdf)
        for expected in ["Summary of the Evidence", "Thematic synthesis", "Evidence Table",
                         "Trial number 3 of vitamin D", "Finding sentence for study 4", "Knowledge Gaps",
                         "Optimal dose remains unclear", "Evidence coverage grid", "Monthly bolus",
                         "Literature Map", "Theme 1"]:
            self.assertIn(expected, text)

    def test_wide_views_are_on_landscape_pages(self):
        reader = PdfReader(io.BytesIO(generate_pdf_report(_request())))
        sizes = [(float(p.mediabox.width) > float(p.mediabox.height), p.extract_text()) for p in reader.pages]
        table_page = next(wide for wide, txt in sizes if "Evidence Table" in txt)
        first_page = sizes[0][0]
        self.assertTrue(table_page)
        self.assertFalse(first_page)

    def test_unticked_views_are_left_out(self):
        text = _text(generate_pdf_report(_request(include_evidence_table=False, include_gap_matrix=False,
                                                  include_network_summary=False)))
        self.assertNotIn("Evidence Table", text)
        self.assertNotIn("Knowledge Gaps", text)
        self.assertNotIn("Literature Map", text)
        self.assertIn("Summary of the Evidence", text)

    def test_minimal_request_and_markup_characters(self):
        pdf = generate_pdf_report(ExportRequest(format="pdf", query="a < b & c > d?",
                                                executive_answer="Text with <tags> & **bold** words."))
        self.assertIn("a < b & c > d?", _text(pdf))

    def test_lost_ligatures_are_repaired(self):
        self.assertEqual(repair_ligatures("The e\x00ects were signi\x00cant, e\x00cacy \x00rst"),
                         "The effects were significant, efficacy first")
        self.assertEqual(repair_ligatures("eﬃcacy and ﬁeld"), "efficacy and field")


if __name__ == "__main__":
    unittest.main()
