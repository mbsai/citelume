"""Search planning: concepts, breadth, aspects and the searches compiled from them."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.query_planning import aspects_from_model, concepts_from_model, detect_breadth
from services.query_utils import compile_deterministic_decomposition_plan as plan_for
from services.query_utils import understand_research_question


class TestBreadth(unittest.TestCase):
    def test_topics_are_broad_and_questions_focused(self):
        for q in ("gut microbiome", "cancer immunotherapy", "machine learning in drug discovery",
                  "What is the state of Coronary artery disease in India"):
            self.assertEqual(plan_for(q).question_understanding.breadth, "broad", q)
        for q in ("How does sleep affect memory in older adults?", "Does vitamin D prevent COVID-19?",
                  "SGLT2 inhibitors vs GLP-1 agonists for heart failure"):
            self.assertEqual(plan_for(q).question_understanding.breadth, "focused", q)

    def test_model_cannot_make_a_relational_question_broad(self):
        q = "How does sleep affect memory in older adults?"
        u = plan_for(q, model_plan={"breadth": "broad"}).question_understanding
        self.assertEqual(u.breadth, "focused")


class TestPlanShape(unittest.TestCase):
    def test_core_reviews_then_aspects(self):
        p = plan_for("gut microbiome")
        self.assertEqual([f.kind for f in p.search_facets], ["core", "reviews"] + ["aspect"] * 4)
        self.assertEqual(len(plan_for("gut microbiome", "deep_research").search_facets), 8)
        core, reviews, first = p.search_facets[0], p.search_facets[1], p.search_facets[2]
        self.assertNotIn(" AND ", core.queries.pubmed.replace("gut[Title/Abstract] AND microbiome", ""))
        self.assertIn("[Publication Type]", reviews.queries.pubmed)
        # Every aspect search still requires the topic
        for f in p.search_facets:
            self.assertIn('"gut microbiome"[Title/Abstract]', f.queries.pubmed)
        self.assertIn(" AND (", first.queries.pubmed)

    def test_subquestions_are_readable(self):
        u = plan_for("gut microbiome").question_understanding
        self.assertEqual(u.subquestions[0].question_text, "gut microbiome")
        self.assertTrue(all("gut microbiome" in sq.question_text for sq in u.subquestions[1:]))
        self.assertFalse(any("benchmarks" in sq.question_text for sq in u.subquestions))

    def test_aspects_fit_the_question(self):
        labels = lambda q: [f.facet_name for f in plan_for(q).search_facets[2:]]
        self.assertIn("Trials and controlled studies", labels("Does vitamin D prevent COVID-19?"))
        self.assertIn("Epidemiology and risk factors", labels("What is the state of Coronary artery disease in India"))
        self.assertIn("Evaluation and benchmarks", labels("machine learning in drug discovery"))
        self.assertIn("Open problems", labels("Riemann hypothesis recent progress"))

    def test_methods_and_compounds_kept_as_concepts(self):
        names = lambda q: [c.name for c in plan_for(q).question_understanding.concepts.search_concepts]
        self.assertEqual(names("machine learning in drug discovery"), ["drug discovery", "machine learning"])
        self.assertEqual(names("climate change and health")[0], "climate change")


class TestModelPlanValidation(unittest.TestCase):
    def setUp(self):
        self.u = understand_research_question("gentamycin heteroresistance")

    def test_synonyms_and_spelling_fix_accepted(self):
        got = concepts_from_model([{"name": "gentamicin", "synonyms": ["gentamycin", "gentamicin"]},
                                   {"name": "heteroresistance", "synonyms": ["hetero-resistance"]}], self.u)
        self.assertEqual([c.name for c in got], ["gentamicin", "heteroresistance"])
        self.assertEqual(got[0].synonyms, ["gentamycin"])

    def test_swapped_topic_rejected(self):
        self.assertEqual(concepts_from_model([{"name": "vancomycin tolerance", "synonyms": []}], self.u), [])

    def test_boolean_and_tags_stripped_and_junk_dropped(self):
        got = aspects_from_model([
            {"label": "Detection methods", "terms": ['"population analysis"[tiab]', "Etest AND PAP", "microdilution"]},
            {"label": "Detection methods", "terms": ["dup", "dup2"]},
            {"label": "Too thin", "terms": ["one"]},
            "not a dict",
        ], 4)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].terms, ["population analysis", "Etest PAP", "microdilution"])

    def test_model_plan_is_compiled_by_rules(self):
        p = plan_for("gentamycin heteroresistance", model_plan={
            "concepts": concepts_from_model([{"name": "gentamicin", "synonyms": ["gentamycin"]},
                                             {"name": "heteroresistance"}], self.u),
            "aspects": aspects_from_model([{"label": "Mechanisms", "terms": ["efflux", "gene amplification"]},
                                           {"label": "Clinical impact", "terms": ["treatment failure", "outcome"]}], 4),
        })
        pm = p.search_facets[2].queries.pubmed
        self.assertIn("gentamicin[Title/Abstract] OR gentamycin[Title/Abstract]", pm)
        self.assertIn('"gene amplification"[Title/Abstract]', pm)
        self.assertEqual(pm.count("("), pm.count(")"))
        self.assertEqual(p.question_understanding.planned_by, "model")


if __name__ == "__main__":
    unittest.main()
