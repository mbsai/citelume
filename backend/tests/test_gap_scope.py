"""Gap scope by mode (24 Nov 2026): a quick answer reports only open questions its studies' authors state and
disagreements among its studies; every gap names its reason after Robinson et al. 2011; a gap is checked against
every record the selection left out, not the first few."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import PaperRecord
from services.gap_matrix_engine import (ROBINSON_BY_CATEGORY, build_universal_knowledge_gap_matrix,
                                        clear_gap_verification_cache)
from tests.test_gap_derivation import CAD_Q, _p, cad_papers


def with_disagreement():
    papers = cad_papers()
    papers[2].stance = "contradicting"
    papers[0].stance = papers[1].stance = "supporting"
    return papers


class TestQuickAnswerGaps(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_gap_verification_cache()

    async def test_quick_answer_reports_only_author_questions_and_disagreements(self):
        quick = await build_universal_knowledge_gap_matrix(CAD_Q, with_disagreement(), [], quick=True)
        full = await build_universal_knowledge_gap_matrix(CAD_Q, with_disagreement(), [])
        self.assertEqual(quick.scope, "included_studies")
        self.assertEqual(full.scope, "review")
        profiles = {g.methodological_profile for g in quick.structured_gaps + quick.resolved_or_invalidated_gaps}
        self.assertTrue(profiles)
        self.assertLessEqual(profiles, {"author_stated_open_question", "comparative_standardization"})
        self.assertTrue(any(g.primary_gap_type == "Inconsistent findings" for g in quick.structured_gaps))
        self.assertEqual((quick.rows, quick.cells, quick.top_opportunities, quick.subquestion_coverage,
                          quick.research_opportunities), ([], [], [], [], []))
        self.assertIn("does not report gaps in the literature", quick.corpus_adequacy_explanation)
        # The deep review keeps the design-profile gaps (no cohort or trial in a set of surveys)
        self.assertTrue(any(g.methodological_profile.startswith("design_profile") for g in full.structured_gaps))
        self.assertTrue(full.rows)

    async def test_every_gap_has_a_robinson_reason(self):
        m = await build_universal_knowledge_gap_matrix(CAD_Q, with_disagreement(), [])
        for g in m.structured_gaps + m.resolved_or_invalidated_gaps:
            self.assertIn(g.robinson_reason, {"insufficient_or_imprecise", "biased", "inconsistent",
                                              "not_the_right_information"}, g.gap_title)
        conflict = next(g for g in m.structured_gaps if g.primary_gap_type == "Inconsistent findings")
        self.assertEqual(conflict.robinson_reason, "inconsistent")
        self.assertEqual(ROBINSON_BY_CATEGORY["Population or contextual limitations"], "not_the_right_information")


class TestVerificationReadsWholePool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_gap_verification_cache()

    async def test_matching_record_after_the_first_ten_is_found(self):
        filler = [PaperRecord(id=f"f{i}", title=f"Unrelated record {i}", abstract="Nothing relevant.", source="PubMed")
                  for i in range(40)]
        base = await build_universal_knowledge_gap_matrix(CAD_Q, cad_papers(), [], quick=True)
        lpa = next(g for g in base.structured_gaps if "lipoprotein" in g.gap_title.lower())
        words = " ".join(lpa.verification_queries[0].split())
        late = PaperRecord(id="late", title=f"Study of {words}", abstract=f"We measured {words} in young adults.",
                           source="Europe PMC")
        clear_gap_verification_cache()
        m = await build_universal_knowledge_gap_matrix(CAD_Q, cad_papers(), [], quick=True,
                                                       all_candidate_pool=filler + [late])
        ids = {r.verification_source_id for r in m.supplementary_verification_records}
        self.assertIn("SUPP-late", ids)


if __name__ == "__main__":
    unittest.main()


class TestExportsShowScopeAndReason(unittest.IsolatedAsyncioTestCase):
    async def test_quick_heading_and_reason_in_text_and_pdf(self):
        from models import ExportRequest
        from services.export_service import generate_markdown_report, generate_txt_report
        from services.pdf_report import generate_pdf_report
        from tests.test_pdf_report import _text
        clear_gap_verification_cache()
        gm = await build_universal_knowledge_gap_matrix(CAD_Q, with_disagreement(), [], quick=True)
        req = ExportRequest(format="pdf", query=CAD_Q, executive_answer="Answer.", papers=with_disagreement(),
                            gap_matrix=gm)
        for text in (generate_markdown_report(req), generate_txt_report(req), _text(generate_pdf_report(req))):
            self.assertIn("WHERE THE STUDIES DISAGREE", text.upper())
            self.assertIn("Inconsistent results", text)


if __name__ == "__main__":
    unittest.main()
