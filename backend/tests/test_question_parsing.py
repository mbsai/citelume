import sys
import unittest
from pathlib import Path

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import FacetApiQueries, PaperRecord, RequiredConstraint, TaskCStudyExtractionItem
from services.graph_builder import _topic_relevance, deduplicate_and_rank_papers
from services.query_utils import (
    CONVERSATIONAL_STOP_WORDS,
    apply_required_constraints,
    compile_deterministic_decomposition_plan,
    understand_research_question,
)
from services.question_parsing import concept_phrases, detect_constraints
from services.verification_gates import verify_claims_against_sources


class TestQuestionParsing(unittest.TestCase):
    def test_concepts_stay_whole_and_framing_words_drop(self):
        q = "What is the state of Coronary artery disease in India"
        self.assertEqual(concept_phrases(q, CONVERSATIONAL_STOP_WORDS), ["coronary artery disease"])
        self.assertNotIn("state", understand_research_question(q).concepts.main_entities)

    def test_place_and_population_constraints(self):
        found = {(c.kind, c.name) for c in detect_constraints("Asthma in Chinese children exposed to air pollution")}
        self.assertEqual(found, {("place", "China"), ("population", "children")})
        names = [c.name for c in detect_constraints("Malaria burden in sub-Saharan Africa")]
        self.assertEqual(names, ["Sub-Saharan Africa"])  # not also "Africa"

    def test_constraints_are_required_in_every_source_query(self):
        plan = compile_deterministic_decomposition_plan("What is the state of Coronary artery disease in India")
        for facet in plan.search_facets:
            q = facet.queries
            self.assertIn("india[Title/Abstract]", q.pubmed)
            self.assertNotIn('OR "india"', q.pubmed)  # never an optional term
            self.assertIn("India", q.semantic_scholar)
            self.assertIn("all:india", q.arxiv)
            self.assertTrue(all("India" in b for b in q.biorxiv_medrxiv))
            self.assertIn('"coronary artery disease"', q.pubmed)

    def test_llm_queries_get_missing_constraints_added(self):
        india = RequiredConstraint(kind="place", name="India", spellings=["india", "indian"])
        q = apply_required_constraints(
            FacetApiQueries(pubmed='"coronary artery disease"[Title/Abstract]', semantic_scholar="CAD prevalence",
                            arxiv="all:coronary", biorxiv_medrxiv=["CAD"]), [india])
        self.assertIn("india[Title/Abstract]", q.pubmed)
        self.assertIn("India", q.semantic_scholar)
        self.assertIn("all:india", q.arxiv)
        self.assertEqual(q.biorxiv_medrxiv, ["CAD India"])


class TestSampleSizeFinder(unittest.TestCase):
    def test_counts_found_and_years_ages_ignored(self):
        from services.extraction_patterns import find_sample_size
        self.assertEqual(find_sample_size("We recruited 997 young patients"), "997 young patients")
        self.assertEqual(find_sample_size("227 articles were published"), "227 articles")
        self.assertEqual(find_sample_size("A total of 1,024 adults aged 20 to 79 years"), "1,024 adults")
        for not_a_sample in ("between 2013 and 2017 patients were", "aged 35 years", "prevalence was 12.5%"):
            self.assertIsNone(find_sample_size(not_a_sample))


class TestRelevanceScreening(unittest.TestCase):
    def _paper(self, i, title, abstract):
        return PaperRecord(id=f"p{i}", title=title, abstract=abstract, source="PubMed", citation_index=i)

    def test_topic_relevance_levels(self):
        c = "coronary artery disease"
        self.assertEqual(_topic_relevance(c, "Coronary artery disease in India", ""), "addresses")
        self.assertEqual(_topic_relevance(c, "HIV outcomes", "Comorbidities included coronary artery disease."), "partly")
        self.assertEqual(_topic_relevance(c, "Fatty liver in India", "Prevalence of NAFLD."), "off_topic")

    def test_ranking_drops_studies_outside_the_constraints(self):
        u = understand_research_question("What is the state of Coronary artery disease in India")
        india = [self._paper(i, f"Coronary artery disease in India study {i}", "Indian adults with CAD.") for i in range(1, 7)]
        others = [self._paper(10, "Coronary artery disease risk factors in Iran", "Iranian survey of CAD."),
                  self._paper(11, "Fatty liver disease in India", "NAFLD prevalence in Indian adults.")]
        stats = {}
        ranked = deduplicate_and_rank_papers(india + others, "coronary artery disease india", 20, u, stats)
        titles = [p.title for p in ranked]
        self.assertFalse(any("Iran" in t or "Fatty liver" in t for t in titles))
        self.assertEqual(stats["constraint_excluded"], 2)

    def test_claims_may_cite_non_contiguous_reference_numbers(self):
        papers = [self._paper(i, f"T{i}", "a") for i in (1, 2, 16)]  # 3-15 screened out as off-topic

        class Claim:
            claim_id, supporting_indices, opposing_indices, proposition, claim_category = "c1", [16], [], "x", "association"

        self.assertEqual([e for e in verify_claims_against_sources([Claim()], papers) if "not among" in e], [])

        class BadClaim(Claim):
            supporting_indices = [5]

        self.assertTrue(any("not among" in e for e in verify_claims_against_sources([BadClaim()], papers)))

    def test_model_screening_fields_exist(self):
        item = TaskCStudyExtractionItem(citation_index=1, addresses_question="no", relevance_reason="Iran, not India")
        self.assertEqual(item.addresses_question, "no")


if __name__ == "__main__":
    unittest.main()
