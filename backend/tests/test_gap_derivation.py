import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ElicitExtraction, PaperRecord, TaskHIGapAssignmentItem, TaskHIGapOutput
from services.gap_derivation import design_family, find_author_open_questions, open_question_title
from services.gap_matrix_engine import build_universal_knowledge_gap_matrix, clear_gap_verification_cache

CAD_Q = "What is the state of coronary artery disease in India"
SSC_Q = "Mechanisms of lung fibrosis in systemic sclerosis"


def _p(i, title, abstract, design="", quant="", findings=""):
    return PaperRecord(
        id=f"p{i}", citation_index=i, title=title, abstract=abstract, source="PubMed", year=2022,
        elicit=ElicitExtraction(study_design=design, quantitative_result=quant, key_findings=findings,
                                provenance_type="abstract_derived"),
    )


def cad_papers():
    return [
        _p(1, "Prevalence of coronary artery disease risk factors in rural Andhra Pradesh",
           "A cross-sectional community-based survey of 2,400 adults. Hypercholesterolaemia prevalence ranged from 4% to 9%. "
           "Further prospective studies are needed to estimate the incidence of coronary events in rural India.",
           quant="prevalence 4-9%", findings="Hypercholesterolaemia prevalence 4-9%"),
        _p(2, "Premature coronary artery disease among young Indian adults: a hospital-based study",
           "In this case-control study we enrolled 997 young patients at a tertiary care hospital. Smoking was associated with premature CAD (OR 14.25). "
           "The role of lipoprotein(a) in young Indians remains unclear.",
           quant="OR 14.25", findings="Smoking associated with premature CAD"),
        _p(3, "Cross-sectional survey of hypertension and coronary artery disease in Kerala",
           "A cross-sectional survey of 5,167 adults found coronary artery disease prevalence of 7.4%.",
           quant="prevalence 7.4%", findings="CAD prevalence 7.4%"),
        _p(4, "Angiographic profile of coronary artery disease at a single centre in North India",
           "Single-centre retrospective review of 1,200 angiograms showed triple-vessel disease in 38%.",
           quant="38% triple-vessel disease", findings="Triple vessel disease in 38%"),
    ]


def ssc_papers():
    return [
        _p(1, "IL-11 drives fibroblast activation in systemic sclerosis lung",
           "In vitro, IL-11 was significantly upregulated in SSc lung fibroblasts and knockout mice were protected from fibrosis. "
           "How IL-11 signalling interacts with TGF-beta in human SSc-ILD remains poorly understood.",
           findings="IL-11 upregulated in SSc lung fibroblasts"),
        _p(2, "Macrophage polarisation in a bleomycin mouse model of scleroderma lung disease",
           "Using a murine bleomycin model, M2 macrophages promoted collagen deposition in vivo.",
           findings="M2 macrophages promoted collagen deposition"),
        _p(3, "Single-cell profiling of SSc fibroblast cell lines",
           "Cell lines from SSc donors showed myofibroblast programs in assays of contraction.",
           findings="Myofibroblast programs in SSc cell lines"),
    ]


class TestEvidenceDerivedGaps(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_gap_verification_cache()

    def test_design_families(self):
        fams = [design_family(p) for p in cad_papers()]
        self.assertEqual(fams[:3], ["cross_sectional", "case_control", "cross_sectional"])
        self.assertEqual([design_family(p) for p in ssc_papers()], ["lab", "lab", "lab"])

    def test_open_questions_and_titles(self):
        qs = find_author_open_questions(cad_papers(), CAD_Q)
        self.assertTrue(any("lipoprotein" in q.sentence for q in qs))
        self.assertEqual(open_question_title("The role of lipoprotein(a) in young Indians remains unclear."),
                         "Role of lipoprotein(a) in young Indians remains unclear")
        self.assertTrue(open_question_title(
            "Further prospective studies are needed to estimate the incidence of coronary events in rural India."
        ).startswith("Needed: estimate the incidence of coronary events"))

    async def test_two_topics_get_different_specific_gaps(self):
        cad = await build_universal_knowledge_gap_matrix(CAD_Q, cad_papers(), [])
        ssc = await build_universal_knowledge_gap_matrix(SSC_Q, ssc_papers(), [])
        cad_text = " ".join(f"{g.gap_title} {g.unresolved_issue} {g.why_it_matters}" for g in cad.structured_gaps)
        ssc_text = " ".join(f"{g.gap_title} {g.unresolved_issue} {g.why_it_matters}" for g in ssc.structured_gaps)
        # No fixed template wording leaks into an epidemiology question
        for banned in ("wet-lab", "in vivo", "scaffold", "Unresolved Coverage", "Adjacent regimes", "physiological toxicity"):
            self.assertNotIn(banned, cad_text)
        self.assertIn("lipoprotein", cad_text)
        self.assertIn("cross-sectional", cad_text.lower())
        self.assertIn("IL-11", ssc_text)
        self.assertIn("cell or animal", ssc_text)
        self.assertFalse({g.gap_title for g in cad.structured_gaps} & {g.gap_title for g in ssc.structured_gaps})

    async def test_same_topic_pool_papers_do_not_mark_gaps_partially_addressed(self):
        pool = cad_papers() + [_p(9, "Coronary artery disease in India: an overview", "Coronary artery disease is common in India.")]
        m = await build_universal_knowledge_gap_matrix(CAD_Q, cad_papers(), [], all_candidate_pool=pool)
        self.assertTrue(m.structured_gaps)
        self.assertFalse(any(g.verification_status == "Partially Addressed" for g in m.structured_gaps))

    async def test_model_grid_cells_use_question_specific_axes(self):
        grid = TaskHIGapOutput(
            rows=["Risk factors", "Prevalence", "Treatment access"],
            cols=["Rural", "Urban", "Women"],
            assignments=[
                TaskHIGapAssignmentItem(row_idx=0, col_idx=0, paper_indices=[1]),
                TaskHIGapAssignmentItem(row_idx=1, col_idx=1, paper_indices=[3, 4]),
                TaskHIGapAssignmentItem(row_idx=2, col_idx=2, paper_indices=[],
                                        why_gap_exists="No study reports access to revascularisation among Indian women."),
            ],
        )
        m = await build_universal_knowledge_gap_matrix(CAD_Q, cad_papers(), [], llm_grid=grid)
        self.assertEqual(m.rows, ["Risk factors", "Prevalence", "Treatment access"])
        self.assertEqual(m.cells[1][1].paper_indices, [3, 4])
        texts = [g.unresolved_issue for g in m.structured_gaps]
        self.assertTrue(any("revascularisation among Indian women" in t for t in texts) or len(m.structured_gaps) == 6)


if __name__ == "__main__":
    unittest.main()
