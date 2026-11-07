"""How people actually phrase searches: every style must parse into sensible concepts without crashing."""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException

from models import SearchRequest
from services.identifier_query import identifier_in_query
from services.query_utils import compile_deterministic_decomposition_plan, understand_research_question


def concepts(q):
    return understand_research_question(q).concepts.concept_phrases


class TestQueryStyles(unittest.TestCase):
    EXPECT = {
        "sepsis": ["sepsis"],
        "gut microbiome": ["gut microbiome"],
        "COPD": ["COPD"],
        "diabetes, exercise, HbA1c": ["diabetes", "exercise", "hba1c"],
        "What causes Alzheimer's disease?": ["alzheimer disease"],
        "How does metformin work?": ["metformin"],
        "Does vitamin D prevent COVID-19?": ["vitamin D", "covid-19"],
        "Tell me about CAR-T cell therapy toxicity": ["CAR-T cell therapy toxicity"],
        "Can you please find papers on long COVID in children?": ["long COVID"],
        "I'm writing my thesis and need everything on microplastics in drinking water": ["microplastics", "drinking water"],
        "SGLT2 inhibitors vs GLP-1 agonists for heart failure": ["SGLT2 inhibitors", "GLP-1 agonists", "heart failure"],
        "Advances in perovskite solar cells since 2020": ["perovskite solar cells"],
        "Riemann hypothesis recent progress": ["riemann hypothesis"],
        '"deep brain stimulation" Parkinson': ["deep brain stimulation", "parkinson"],
        "IMMUNOTHERAPY FOR MELANOMA": ["immunotherapy", "melanoma"],
        "🧬 gene therapy for sickle cell 💉": ["gene therapy", "sickle cell"],
    }

    def test_concepts_for_each_style(self):
        for q, want in self.EXPECT.items():
            with self.subTest(q=q):
                self.assertEqual(concepts(q)[:len(want)], want)

    def test_subject_before_measures(self):
        self.assertEqual(concepts("Prevalence of anaemia among pregnant women in Nigeria")[0], "anaemia")

    def test_negated_terms_are_not_required(self):
        q = compile_deterministic_decomposition_plan("(asthma OR wheeze) AND (air pollution) NOT children")
        pm = q.search_facets[0].queries.pubmed
        self.assertNotIn("children[Title/Abstract] OR", pm)
        self.assertIn("(asthma[Title/Abstract] OR wheeze[Title/Abstract]) AND", pm)

    def test_pasted_abstract_section_labels_ignored(self):
        q = ("Background: Atrial fibrillation (AF) is the most common arrhythmia. Methods: We conducted a cohort "
             "study. Results: Ablation lowered stroke risk.")
        self.assertTrue(concepts(q)[0].startswith("atrial fibrillation"))

    def test_non_latin_letters_kept_whole(self):
        self.assertIn("depresión", concepts("efectos del ejercicio en la depresión"))

    def test_identifiers(self):
        self.assertEqual(identifier_in_query("10.1038/s41569-022-00760-3"), ("doi", "10.1038/s41569-022-00760-3"))
        self.assertEqual(identifier_in_query("https://pubmed.ncbi.nlm.nih.gov/36064969/"), ("pmid", "36064969"))
        self.assertEqual(identifier_in_query("arXiv:1706.03762"), ("arxiv", "1706.03762"))
        self.assertIsNone(identifier_in_query("type 2 diabetes 2020"))

    def test_queries_without_a_topic_are_refused_before_searching(self):
        import main
        for q in ("x", "what is it about", "What is the state of the art?", "???"):
            with self.subTest(q=q):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(main._prepare_query(SearchRequest(query=q)))
                self.assertEqual(ctx.exception.status_code, 400)
        r, note = asyncio.run(main._prepare_query(SearchRequest(query="gut microbiome")))
        self.assertEqual((r.query, note), ("gut microbiome", ""))

    def test_no_style_crashes_or_yields_an_empty_search(self):
        for q in ["x y", "a, b, c", "((((", "NOT NOT NOT cancer", "COVID-19 vs. influenza in 2021-2022",
                  "2+2", "β-amyloid and τ", "10,000 steps/day", "Ca2+ signalling", "type 2 diabetes"]:
            with self.subTest(q=q):
                plan = compile_deterministic_decomposition_plan(q)
                self.assertTrue(plan.search_facets)


class TestSpellingVariants(unittest.TestCase):
    def test_variant_spellings_match_and_are_searched(self):
        from services.question_parsing import concept_mentions
        self.assertTrue(concept_mentions("gentamycin heteroresistance", "gentamicin heteroresistance in E. coli"))
        self.assertTrue(concept_mentions("anaemia", "Anemia in pregnancy"))
        self.assertTrue(concept_mentions("tumour", "tumor growth"))
        pm = compile_deterministic_decomposition_plan("gentamycin heteroresistance").search_facets[0].queries.pubmed
        self.assertIn("gentamicin[Title/Abstract]", pm)


if __name__ == "__main__":
    unittest.main()
