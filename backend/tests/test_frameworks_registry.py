import sys
import unittest
from pathlib import Path

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from frameworks.appraisal_routes import detect_biomedical_design, route_biomedical
from frameworks.reporting import build_review_protocol_report
from frameworks.review_types import ORGANIZATIONS, REVIEW_FRAMEWORKS
from frameworks.risk_of_bias import TOOLS as ROB_TOOLS
from frameworks.rules import GATES, RULES, method_rule_texts, render_rules_for_prompt, rule_texts, select_rules
from frameworks.sources import SOURCES
from models import PaperRecord
from services.domain_profiles import DOMAIN_PROFILES, METHODOLOGICAL_PROFILES
from services.query_utils import compile_deterministic_decomposition_plan, understand_research_question
from services.verification_gates import extract_nontrivial_numbers
from services.review_composition import compose_organized_body, finalize_llm_sections, grade_informed_indicators
from models import SynthesisSection

SELECTABLE_DOMAINS = [
    "biomedical_clinical", "computer_science_ai", "interdisciplinary_biomedical_ai", "chemistry", "physics",
    "mathematics", "engineering_materials", "social_sciences", "humanities_interdisciplinary",
]


def _paper(idx, year, design, stance="supporting", abstract=""):
    p = PaperRecord(id=f"p{idx}", title=f"Study {idx}", year=year, source="PubMed", citation_index=idx,
                    authors=["Chen L", "Patel R"], abstract=abstract, stance=stance)
    p.elicit.study_design = design
    p.elicit.key_findings = f"Finding {idx}"
    return p


class TestRuleRegistryIntegrity(unittest.TestCase):
    def test_every_rule_cites_known_sources(self):
        for r in RULES:
            self.assertTrue(r.source_ids, f"{r.rule_id} has no source")
            for s in r.source_ids:
                self.assertIn(s, SOURCES, f"{r.rule_id} cites unknown source {s}")

    def test_every_source_is_used(self):
        used = {s for r in RULES for s in r.source_ids}
        used |= {s for f in REVIEW_FRAMEWORKS.values() for s in f.source_ids}
        used |= {s for o in ORGANIZATIONS.values() for s in o.source_ids}
        used |= {s for t in ROB_TOOLS.values() for s in t.source_ids}
        self.assertEqual(set(SOURCES) - used, set())

    def test_risk_of_bias_tools_cite_known_sources(self):
        for t in ROB_TOOLS.values():
            self.assertTrue(t.domains, t.tool_id)
            self.assertTrue(set(t.source_ids) <= set(SOURCES), t.tool_id)

    def test_enforcement_points_to_real_gates(self):
        for r in RULES:
            self.assertTrue(r.enforcement == "prompt" or r.enforcement in GATES, r.rule_id)

    def test_every_selectable_domain_has_appraisal_and_synthesis_rules(self):
        for dom in SELECTABLE_DOMAINS:
            self.assertTrue(rule_texts(dom, "appraisal"), f"{dom} has no appraisal rules")
            self.assertTrue(rule_texts(dom, "synthesis"), f"{dom} has no synthesis rules")
            self.assertEqual(DOMAIN_PROFILES[dom].synthesis_interpretation_rules, rule_texts(dom, "synthesis"))

    def test_rule_wording_has_no_figures_that_would_trip_the_number_gate(self):
        for r in RULES:
            self.assertEqual(extract_nontrivial_numbers(f"{r.text} ({r.citation_label()})"), [], r.rule_id)

    def test_every_method_profile_has_registry_rules(self):
        for method_id, prof in METHODOLOGICAL_PROFILES.items():
            self.assertTrue(prof.appraisal_criteria, f"{method_id} has no appraisal rules")
            self.assertTrue(prof.synthesis_qualification_rule, f"{method_id} has no synthesis rule")
            self.assertEqual(prof.appraisal_criteria, method_rule_texts(method_id, "appraisal"))

    def test_no_rule_text_is_written_inline_in_profiles(self):
        text = (BACKEND_DIR / "services/domain_profiles.py").read_text()
        for field in ("methodological_appraisal_rules=[", "synthesis_interpretation_rules=[", "appraisal_criteria=["):
            self.assertNotIn(field, text)
        self.assertNotRegex(text, r'synthesis_qualification_rule="\w')  # a literal sentence, not the registry view

    def test_prompt_rule_block_respects_token_budget(self):
        block = render_rules_for_prompt(select_rules("biomedical_clinical", ["rct", "animal"], "systematic_prisma"), 120)
        self.assertLessEqual(len(block) / 3.5, 125)
        self.assertTrue(block.startswith("1. "))

    def test_audited_wording_errors_are_gone(self):
        for rel in ("services/claim_synthesis.py", "services/evidence_table_engine.py"):
            text = (BACKEND_DIR / rel).read_text()
            self.assertNotIn("synthesized qualitatively", text)
            self.assertNotIn("domain:Unclear", text)
            self.assertNotIn("PROBAST+AI / QUIPS (Clinical Prediction", text)
            self.assertNotIn("ARRIVE 2.0 / SYRCLE (Preclinical", text)


class TestDesignRouting(unittest.TestCase):
    CASES = {
        "In this double-blind randomized controlled trial of 200 adults": "rct",
        "We trained a random forest classifier to predict sepsis": "prediction_model",
        "Sensitivity and specificity of CT against the reference standard": "diagnostic_accuracy",
        "A population-based cohort of 10,000 adults exposed to PM2.5": "observational_exposure",
        "Tumour size was an independent predictor of survival": "prognostic_factor",
        "Treated mice showed reduced tumour growth in a xenograft model": "animal",
        "A non-randomized single-arm study of the new therapy": "nrsi_intervention",
    }

    def test_designs_route_to_matching_framework(self):
        for abstract, design in self.CASES.items():
            self.assertEqual(detect_biomedical_design(abstract), design, abstract)

    def test_routes_never_issue_tool_ratings(self):
        for abstract in self.CASES:
            route = route_biomedical(abstract)
            self.assertIn("not a formal rating", route.framework_label)
            for concern in route.concerns:
                self.assertNotIn("Unclear", concern)
                self.assertNotRegex(concern, r"\b(low|high) risk of bias\b")


class TestDomainOverride(unittest.TestCase):
    def test_user_domain_wins_over_detection(self):
        u = understand_research_question("Transformer models for protein structure prediction", "physics")
        self.assertEqual(u.primary_domain, "physics")
        self.assertTrue(u.domain_selected_by_user)

    def test_auto_detection_unchanged_without_override(self):
        u = understand_research_question("Transformer models for protein structure prediction")
        self.assertFalse(u.domain_selected_by_user)
        self.assertNotEqual(u.primary_domain, "physics")


class TestReviewTypes(unittest.TestCase):
    def setUp(self):
        self.papers = [
            _paper(1, 2018, "Randomized Trial"), _paper(2, 2019, "Cohort", "mixed"),
            _paper(3, 2021, "Randomized Trial", "contradicting"), _paper(4, 2023, "Cohort"),
            _paper(5, 2024, "Animal Study"),
        ]

    def test_each_organization_produces_its_heading(self):
        for org_id, org in ORGANIZATIONS.items():
            if org_id == "thematic":
                continue  # thematic is the composer's own default body
            section = compose_organized_body(org_id, self.papers, [], [])
            self.assertEqual(section.heading, org.body_heading)
            self.assertTrue(section.content)

    def test_chronological_periods_are_in_order(self):
        content = compose_organized_body("chronological", self.papers, [], []).content
        self.assertLess(content.index("2018"), content.index("2024"))

    def test_methodological_groups_by_design(self):
        content = compose_organized_body("methodological", self.papers, [], []).content
        for design in ("Randomized Trial", "Cohort", "Animal Study"):
            self.assertIn(design, content)

    def test_prisma_flow_counts_add_up(self):
        plan = compile_deterministic_decomposition_plan("statins and cardiovascular outcomes", "deep_research")
        report = build_review_protocol_report(
            "systematic_prisma", "thematic", "biomedical_clinical", plan, self.papers,
            {"PubMed": 30, "Semantic Scholar": 20}, identified_total=50, unique_records=31,
            enabled_sources=["PubMed", "Semantic Scholar"], year_from=None, year_to=None,
        )
        f = report.flow
        self.assertEqual(f.identified_total - f.duplicates_removed, f.screened)
        self.assertEqual(f.screened - f.excluded_at_screening, f.included)
        self.assertIn("not a registered systematic review", report.disclosure)
        self.assertTrue(report.search_strategies["PubMed"])
        self.assertIn("RV-prisma-flow", [r.rule_id for r in report.rules_applied])
        # Method profiles detected by the planner select their rules too (here: observational / review designs)
        methods = plan.question_understanding.active_methodological_profiles
        self.assertTrue(methods)
        applied = {r.rule_id for r in report.rules_applied}
        for m in methods:
            for r in RULES:
                if r.study_designs == (m,) and r.stage in ("appraisal", "synthesis"):
                    self.assertIn(r.rule_id, applied)

    def test_narrative_has_no_flow_and_llm_sections_get_disclosure(self):
        plan = compile_deterministic_decomposition_plan("statins and cardiovascular outcomes")
        report = build_review_protocol_report(
            "narrative", "thematic", "biomedical_clinical", plan, self.papers, {}, 5, 5, ["PubMed"], None, None)
        self.assertIsNone(report.flow)
        secs = [SynthesisSection(heading=h, content="x", cited_indices=[1]) for h in ("B. x", "C. x", "D. x", "E. x")]
        out = finalize_llm_sections("systematic_prisma", secs, self.papers, [])
        self.assertIn("GRADE-informed indicators", out[2].content)
        self.assertIn("not a registered systematic review", out[-1].content)
        self.assertIn("not a GRADE certainty rating", grade_informed_indicators(self.papers, []))



class TestLatexCleaning(unittest.TestCase):
    def test_latex_markup_is_removed_from_titles_and_abstracts(self):
        p = PaperRecord(id="x", title=r"Muon \ensuremath{\mathrm{g}-2} anomaly", source="arXiv",
                        abstract=r"We measure $a_\mu$ with \textbf{high} precision.")
        self.assertEqual(p.title, "Muon g-2 anomaly")
        self.assertNotIn("\\", p.abstract)
        self.assertNotIn("$", p.abstract)


if __name__ == "__main__":
    unittest.main()
