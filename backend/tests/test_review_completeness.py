"""A systematic-style or scoping review that synthesises fewer than all its eligible studies is labelled not complete
(24 Nov 2026): on the page, at the top of every export and in the PRISMA results text."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ExportRequest, ListedRecord, SelectionReport
from services.export_service import generate_markdown_report, generate_txt_report, incompleteness_note
from services.pdf_report import generate_pdf_report
from tests.test_pdf_report import _text


def report(method, left):
    return SelectionReport(selection_method=method, included=60, eligible=60 + left,
                           eligible_over_limit=[ListedRecord(title=f"Eligible study {i}") for i in range(left)])


class TestIncompleteness(unittest.TestCase):
    def test_note_only_for_reviews_that_left_eligible_studies_out(self):
        self.assertEqual(incompleteness_note(None), "")
        self.assertEqual(incompleteness_note(report("systematic_prisma", 0)), "")
        self.assertEqual(incompleteness_note(report("narrative", 40)), "")  # a narrative review has no such rule
        note = incompleteness_note(report("systematic_prisma", 88))
        self.assertTrue(note.startswith("Not complete: 60 of 148 eligible studies were synthesised."))
        self.assertIn("A systematic review includes every eligible study", note)
        self.assertIn("A scoping review", incompleteness_note(report("scoping", 5)))

    def test_every_export_carries_it(self):
        req = ExportRequest(format="pdf", query="Q?", executive_answer="Answer.",
                            selection_report=report("systematic_prisma", 88))
        for text in (generate_markdown_report(req), generate_txt_report(req), _text(generate_pdf_report(req))):
            self.assertIn("Not complete: 60 of 148 eligible studies were synthesised", text)
        plain = ExportRequest(format="md", query="Q?", executive_answer="Answer.",
                              selection_report=report("systematic_prisma", 0))
        self.assertNotIn("Not complete", generate_markdown_report(plain))


if __name__ == "__main__":
    unittest.main()
