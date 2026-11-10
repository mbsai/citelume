import sys
import unittest
from pathlib import Path

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.llm_service import classify_model_scale
from services.query_utils import compile_deterministic_decomposition_plan, understand_research_question
from services.verification_gates import extract_citation_numbers, normalize_citations

SSC_QUESTION = (
    "I want to do a academic grade study on the following topic  Systemic sclerosis drives fibrosis in multiple organs, "
    "I want to what are the genes that drive in lung. I would like to know how similar are the patterns from the lung "
    "in other organs.I plan to involve animal models (mouse), organoids and human clinical samples as well."
)


class TestCitationParsing(unittest.TestCase):
    def test_lists_ranges_and_labels(self):
        text = "Fibrosis [1, 2, 4] and [Study 7]; see [9-11] and [Refs. 13, 15–16], not [2019]."
        self.assertEqual(extract_citation_numbers(text), [1, 2, 4, 7, 9, 10, 11, 13, 15, 16])

    def test_normalised_to_clickable_form(self):
        self.assertEqual(normalize_citations("as shown [Study 3] and [5–7]"), "as shown [3] and [5, 6, 7]")


class TestConversationalQuestion(unittest.TestCase):
    def test_request_wording_is_not_searched(self):
        c = understand_research_question(SSC_QUESTION).concepts
        self.assertEqual(c.concept_phrases[:2], ["systemic sclerosis", "fibrosis"])
        for phrase in ("lung", "genes"):
            self.assertIn(phrase, c.concept_phrases)
        joined = " ".join(c.concept_phrases + c.main_entities)
        for noise in ("academic", "grade", "following", "drives", "patterns"):
            self.assertNotIn(noise, joined)

    def test_model_systems_target_a_search_but_are_not_required(self):
        u = understand_research_question(SSC_QUESTION)
        self.assertEqual({m.name for m in u.concepts.model_systems},
                         {"mouse models", "animal models", "organoids", "human clinical samples"})
        self.assertEqual(u.concepts.required_constraints, [])
        plan = compile_deterministic_decomposition_plan(SSC_QUESTION, "deep_research")
        core = plan.search_facets[0].queries.pubmed
        systems = next(f for f in plan.search_facets if f.facet_name == "Model systems").queries.pubmed
        self.assertNotIn("mouse", core)
        self.assertIn('"mouse"[Title/Abstract]', systems)
        # AI-only terms are not used for biomedical questions
        self.assertFalse(any("diffusion" in f.queries.pubmed for f in plan.search_facets))


class TestModelScale(unittest.TestCase):
    def test_tiers(self):
        cases = {
            ("ollama", "qwen3.5:9b"): "small",
            ("ollama", "llama3.2:latest"): "small",
            ("openai", "gpt-4o-mini"): "small",
            ("openai", "gpt-4o"): "large",
            ("gemini", "gemini-2.5-pro"): "large",
            ("gemini", "gemini-2.5-flash"): "medium",
            ("openrouter", "meta-llama/llama-3.3-70b-instruct"): "large",
            ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"): "large",
            ("openrouter", "nvidia/nemotron-3.5-lightning:free"): "small",
        }
        for (provider, model), tier in cases.items():
            self.assertEqual(classify_model_scale(provider, model)[0], tier, model)


if __name__ == "__main__":
    unittest.main()
