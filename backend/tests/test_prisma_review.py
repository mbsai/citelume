"""Systematic-style (PRISMA 2020) reviews: eligibility-based inclusion, risk of bias, report order and exports."""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader

from frameworks.reporting import build_review_protocol_report
from frameworks.risk_of_bias import overall_judgement, tool_for
from models import (
    ElicitExtraction,
    ExportRequest,
    RiskOfBiasDomain,
    StudyRiskOfBias,
    SynthesisSection,
    TaskCStudyExtractionItem,
)
from services.export_service import generate_markdown_report
from services.graph_builder import deduplicate_and_rank_papers
from services.paper_selection import REASON_PARTIAL, REASON_REVIEW
from services.pdf_report import generate_pdf_report
from services.prisma_report import compose_prisma_report
from services.query_utils import compile_deterministic_decomposition_plan as plan_for
from tests.test_paper_selection import paper

QUERY = "Does vitamin D supplementation reduce fractures in older adults?"


def trial(i, **kw):
    return paper(i, f"Vitamin D supplementation and fractures in older adults: trial {i}",
                 "Older adults were randomly assigned to vitamin D or placebo; fractures were the outcome.", **kw)


def review(i):
    return paper(i, f"Vitamin D supplementation and fractures in older adults: a systematic review and meta-analysis {i}",
                 "We pooled trials of vitamin D and fractures in older adults.")


def partial(i):
    return paper(i, f"Vitamin D supplementation and muscle strength in older adults {i}",
                 "Vitamin D supplementation was given and grip strength measured in older adults.")


def select(papers, n=60, method="systematic_prisma"):
    plan = plan_for(QUERY, "deep_research")
    reports = []
    stats = {}
    chosen = deduplicate_and_rank_papers(papers, QUERY, n, plan.question_understanding, stats, reports, method=method)
    return chosen, reports[0], plan, stats


class TestPrismaSelection(unittest.TestCase):
    def test_every_eligible_primary_study_is_included_and_reviews_are_listed(self):
        papers = [trial(i) for i in range(1, 13)] + [review(i) for i in range(20, 24)] + [partial(i) for i in range(30, 36)]
        chosen, rep, _, _ = select(papers)
        self.assertEqual(rep.selection_method, "systematic_prisma")
        self.assertEqual(len(chosen), 12)  # all eligible trials, no reviews, no partial matches
        self.assertTrue(all("trial" in p.title for p in chosen))
        self.assertEqual(rep.excluded.get(REASON_REVIEW), 4)
        self.assertEqual(rep.excluded.get(REASON_PARTIAL), 6)
        self.assertEqual(rep.exclusion_stage[REASON_REVIEW], "eligibility")
        self.assertEqual([r.label for r in rep.excluded_reviews], ["R1", "R2", "R3", "R4"])
        self.assertEqual(sorted(p.citation_index for p in chosen), list(range(1, 13)))

    def test_eligible_studies_beyond_the_limit_are_listed(self):
        chosen, rep, _, _ = select([trial(i) for i in range(1, 16)], n=10)
        self.assertEqual(len(chosen), 10)
        self.assertEqual(rep.eligible, 15)
        self.assertEqual(len(rep.eligible_over_limit), 5)
        self.assertTrue(any("15 studies met the eligibility criteria" in n for n in rep.notes))

    def test_criteria_are_relaxed_and_reported_when_too_few_studies(self):
        # 2 full matches are too few, so the 4 partial matches are admitted; that makes 6, so the review stays out
        chosen, rep, _, _ = select([trial(1), trial(2)] + [partial(i) for i in range(30, 34)] + [review(40)])
        self.assertEqual(len(chosen), 6)
        self.assertTrue(any(n.startswith("Eligibility criteria relaxed") for n in rep.notes))
        self.assertEqual(len(rep.excluded_reviews), 1)
        # With only 2 primary studies in all, reviews are admitted as well
        chosen, rep, _, _ = select([trial(1), trial(2), review(40)])
        self.assertEqual(len(chosen), 3)
        self.assertFalse(rep.excluded_reviews)

    def test_narrative_selection_is_unchanged(self):
        papers = [trial(i) for i in range(1, 13)] + [review(i) for i in range(20, 24)]
        chosen, rep, _, _ = select(papers, method="narrative")
        self.assertEqual(rep.selection_method, "narrative")
        self.assertTrue(any("meta-analysis" in p.title for p in chosen))


class TestRiskOfBias(unittest.TestCase):
    def test_overall_rule(self):
        self.assertEqual(overall_judgement(["Low"] * 5), "Low")
        self.assertEqual(overall_judgement(["Low", "High", "Low", "Low", "Low"]), "High")
        self.assertEqual(overall_judgement(["Low", "Some concerns", "Low", "Low", "Low"]), "Some concerns")
        self.assertEqual(overall_judgement(["Low", "No information", "Low", "Low", "Low"]), "Some concerns")
        self.assertEqual(overall_judgement(["No information"] * 4 + ["Low"]), "No information")
        self.assertEqual(overall_judgement(["low risk", "serious", "unclear"]), "High")

    def test_tool_matches_design(self):
        self.assertEqual(tool_for("rct").tool_id, "rob2")
        self.assertEqual(tool_for("controlled_observational").tool_id, "robins")
        self.assertEqual(tool_for("something_else").tool_id, "generic")

    def test_model_reply_shapes_are_accepted(self):
        for value in (["Low", "High"], "Low, High", {"domains": ["Low", "High"]}, [{"judgement": "Low"}, {"judgement": "High"}]):
            item = TaskCStudyExtractionItem(citation_index=1, risk_of_bias=value)
            self.assertEqual(item.risk_of_bias, ["Low", "High"])


def _prisma_result(n_studies=8):
    papers = [trial(i) for i in range(1, n_studies + 1)] + [review(i) for i in range(20, 23)]
    chosen, rep, plan, stats = select(papers)
    for p in chosen:
        js = ["Low", "Some concerns", "Low", "No information", "Low"]
        p.elicit = ElicitExtraction(key_findings="Fewer fractures.", study_design="RCT", risk_of_bias=StudyRiskOfBias(
            tool=tool_for("rct").label, domains=[RiskOfBiasDomain(domain=d, judgement=j)
                                                 for d, j in zip(tool_for("rct").domains, js)],
            overall=overall_judgement(js), reason="Allocation concealment not reported."))
    chosen[0].relevance = "off_topic"  # judged off topic during extraction
    protocol = build_review_protocol_report(
        framework_id="systematic_prisma", organization_id="thematic", primary_domain="biomedical_clinical", plan=plan,
        included_papers=chosen, identified_by_source={"PubMed": 12}, identified_total=12,
        unique_records=stats.get("unique_records", 11), enabled_sources=["PubMed"], year_from=None, year_to=None,
        selection_report=rep)
    sections = [
        SynthesisSection(heading="1. Introduction", content="Earlier reviews disagreed [R1]."),
        SynthesisSection(heading="B.1 Daily dosing", content="Daily dosing reduced fractures [2]."),
        SynthesisSection(heading="B.2 Bolus dosing", content="Bolus dosing did not [3]."),
        SynthesisSection(heading="C. Heterogeneity and Conflicting Evidence", content="Populations differed [2, 3]."),
        SynthesisSection(heading="D. Risk of Bias and Certainty of Evidence", content="Moderate certainty [2]."),
        SynthesisSection(heading="4. Discussion", content="Consistent with [R1] for daily dosing [2]."),
    ]
    abstract, out = compose_prisma_report(QUERY, "Vitamin D with calcium reduces fractures [2].", sections, chosen, rep,
                                          protocol, model="test-model")
    return abstract, out, chosen, protocol, rep


class TestPrismaReport(unittest.TestCase):
    def test_flow_counts_add_up(self):
        _, _, chosen, protocol, rep = _prisma_result()
        f = protocol.flow
        self.assertEqual(f.eligible, 8)
        self.assertEqual(f.included, 8)
        self.assertEqual(f.excluded_at_extraction, 1)
        self.assertEqual(f.included_in_synthesis, 7)
        self.assertEqual(f.excluded_at_eligibility.get(REASON_REVIEW), 3)
        self.assertEqual(f.assessed_for_eligibility - sum(f.excluded_at_eligibility.values()), f.eligible)
        self.assertEqual(len(protocol.previous_reviews), 3)

    def test_sections_follow_prisma_order(self):
        abstract, out, *_ = _prisma_result()
        headings = [s.heading for s in out]
        self.assertEqual(headings[:6], ["1. Introduction", "2. Methods", "3.1 Study Selection",
                                        "3.2 Study Characteristics", "3.3 Risk of Bias in Studies",
                                        "3.4 Results of Individual Studies"])
        self.assertEqual(headings[6:9], ["3.5.1 Daily dosing", "3.5.2 Bolus dosing",
                                         "3.5.3 Heterogeneity and Conflicting Results"])
        self.assertEqual(headings[9:], ["3.6 Reporting Biases", "3.7 Certainty of Evidence", "4. Discussion",
                                        "5. Other Information"])
        self.assertEqual(out[2].figure, "prisma_flow")
        self.assertEqual(out[4].figure, "risk_of_bias")
        methods = out[1].content
        for item in ("item 5", "item 8", "item 11", "item 13", "RoB 2", "test-model", "SWiM"):
            self.assertIn(item, methods)
        for label in ("**Objective.**", "**Methods.**", "**Results.**"):
            self.assertIn(label, abstract)
        self.assertIn("[R1]", out[-1].content)

    def test_exports_show_flow_and_risk_of_bias(self):
        abstract, out, chosen, protocol, rep = _prisma_result()
        req = ExportRequest(format="pdf", query=QUERY, mode="deep_research", executive_answer=abstract, sections=out,
                            papers=chosen, review_protocol=protocol, selection_report=rep)
        md = generate_markdown_report(req)
        self.assertIn("## Abstract", md)
        self.assertIn("PRISMA 2020 flow of records", md)
        self.assertIn("| Study | D1 | D2 | D3 | D4 | D5 | Overall | Main concern |", md)
        text = "\n".join(pg.extract_text() for pg in PdfReader(io.BytesIO(generate_pdf_report(req))).pages)
        for expected in ("IDENTIFICATION", "ELIGIBILITY", "Records assessed for eligibility", "RoB 2",
                         "Allocation concealment not reported", "3.7 Certainty of Evidence"):
            self.assertIn(expected, text)


class TestAiScreening(unittest.TestCase):
    def _run(self, fake, n_trials=12, limit=5):
        import asyncio
        from unittest import mock
        from models import ApiKeysConfig
        import services.ai_screening as scr
        chosen, rep, _, _ = select([trial(i) for i in range(1, n_trials + 1)], n=400)
        with mock.patch.object(scr, "execute_validated_llm_task", fake):
            out = asyncio.run(scr.screen_and_limit(chosen, rep, QUERY, ApiKeysConfig(), limit))
        return out, rep, scr

    def test_excluded_records_are_counted_and_the_limit_filled(self):
        async def fake(**kw):
            # Excludes every even-numbered record in each batch
            lines = [ln for ln in kw["user_prompt"].split("\n") if ln[:1].isdigit()]
            from services.ai_screening import ScreeningBatchOutput, ScreeningDecision
            from models import TaskExecutionRecord
            out = ScreeningBatchOutput(decisions=[ScreeningDecision(n=i, include="no" if i % 2 == 0 else "yes",
                                                                    reason="different outcome")
                                                  for i in range(1, len(lines) + 1)])
            return out, TaskExecutionRecord(task_id="Task_S", task_name="t", model_used="fake")
        out, rep, scr = self._run(fake)
        self.assertEqual(len(out), 5)
        self.assertEqual([p.citation_index for p in out], [1, 2, 3, 4, 5])
        self.assertEqual(rep.excluded[scr.REASON_AI_SCREEN], 6)
        self.assertEqual(rep.exclusion_stage[scr.REASON_AI_SCREEN], "eligibility")
        self.assertEqual(rep.eligible, 6)
        self.assertEqual(len(rep.eligible_over_limit), 1)
        self.assertTrue(any("screened 12" in n for n in rep.notes))

    def test_without_a_model_the_word_based_decision_stands(self):
        async def fake(**kw):
            from models import TaskExecutionRecord
            return None, TaskExecutionRecord(task_id="Task_S", task_name="t", model_used="deterministic_offline_engine",
                                             degraded_to_deterministic=True)
        out, rep, scr = self._run(fake)
        self.assertEqual(len(out), 5)
        self.assertNotIn(scr.REASON_AI_SCREEN, rep.excluded)
        self.assertTrue(any("not available" in n for n in rep.notes))


if __name__ == "__main__":
    unittest.main()
