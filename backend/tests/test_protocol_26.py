"""Selection protocol 2.6 and the review fixes from a demo user's report (26 Nov 2026: "Assesment of the atrazine
biotransformation by aspergillus hortai" searched for "assessment", never for the organism, and widened
"atrazine biotransformation" to any biotransformation; the synthesis repeated that nothing was found)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import SearchConcept
from services import mesh_terms
from services.broadening import broaden_suggestions, species_genus
from services.query_planning import build_search_plan
from services.query_utils import understand_research_question

ATRAZINE = "Assesment of the atrazine biotransformation by aspergillus hortai"


def plan(q):
    u = understand_research_question(q)
    facets = build_search_plan(u, "deep_research", "")
    return u, facets


class TestSearchPlan(unittest.TestCase):
    def test_research_action_is_not_a_concept(self):
        u, _ = plan(ATRAZINE)
        names = [c.name for c in u.concepts.search_concepts]
        self.assertEqual(names, ["atrazine biotransformation", "aspergillus hortai"])
        u, _ = plan("Evaluation of probiotics for the prevention of necrotising enterocolitis in preterm infants")
        self.assertNotIn("evaluation", " ".join(c.name for c in u.concepts.search_concepts))

    def test_every_concept_is_searched(self):
        concepts = [SearchConcept(name="probiotics"), SearchConcept(name="necrotising enterocolitis"),
                    SearchConcept(name="infants")]
        u = understand_research_question("probiotics necrotising enterocolitis infants")
        facets = build_search_plan(u, "deep_research", "", concepts=concepts)
        core = [f for f in facets if f.kind == "core"]
        self.assertEqual(len(core), 2)
        self.assertIn("infants", core[1].queries.pubmed)
        self.assertIn("probiotics", core[1].queries.pubmed)
        self.assertNotIn("enterocolitis", core[1].queries.pubmed)

    def test_process_words_have_their_other_names(self):
        from services.query_planning import process_alternatives
        self.assertEqual(process_alternatives("atrazine biotransformation"),
                         ["atrazine biodegradation", "atrazine degradation"])
        self.assertEqual(process_alternatives("macular degeneration"), [])
        self.assertEqual(process_alternatives("degradation"), [])

    def test_environmental_question_gets_its_own_aspects(self):
        u, _ = plan(ATRAZINE)
        labels = [a.label for a in u.aspects]
        self.assertIn("Pathways and metabolites", labels)
        self.assertNotIn("Diagnosis and treatment", labels)


class TestMesh(unittest.TestCase):
    def test_head_noun_heading_keeps_its_modifier(self):
        ts = [{"from": "atrazine", "to": '"atrazine"[Supplementary Concept] OR "atrazine"[All Fields]'},
              {"from": "biotransformation", "to": '"biotransformation"[MeSH Terms]'}]
        groups = mesh_terms.groups_from_translation(ts, "atrazine biotransformation")
        self.assertEqual(mesh_terms.mesh_alternative(groups),
                         '(("biotransformation"[MeSH Terms] OR "biotransformation"[Title/Abstract]) AND '
                         '("atrazine"[Title/Abstract]))')
        # A leading word that is not a substance stays optional (a development review lost records when it did not)
        ts = [{"from": "refractory", "to": '"refractory"[All Fields]'},
              {"from": "glaucoma", "to": '"glaucoma"[MeSH Terms] OR "glaucoma"[All Fields]'}]
        self.assertEqual(mesh_terms.groups_from_translation(ts, "refractory glaucoma"),
                         ['("glaucoma"[MeSH Terms] OR "glaucoma"[Title/Abstract])'])


class TestDesignLabel(unittest.TestCase):
    def test_attention_is_not_a_transformer(self):
        import inspect
        import re
        from services import synthesis_engine as se
        src = inspect.getsource(se)
        pattern = re.search(r'\("Transformer / Foundation Model", r"([^"]+)"\)', src).group(1)
        self.assertIsNone(re.search(pattern, "their emission has attracted increasing attention", re.I))
        self.assertIsNotNone(re.search(pattern, "a pretrained transformer model", re.I))


class TestBroadening(unittest.TestCase):
    def test_species_to_genus(self):
        self.assertEqual(species_genus("aspergillus hortai"), "aspergillus")
        self.assertEqual(species_genus("Candida albicans"), "candida")
        for not_species in ("diabetes mellitus", "multiple sclerosis", "study area", "vitamin d", "aspergillus sp"):
            self.assertIsNone(species_genus(not_species), not_species)

    def test_offered_only_when_little_was_found(self):
        protocol = SimpleNamespace(concepts=[SimpleNamespace(name="atrazine biotransformation", role="topic"),
                                             SimpleNamespace(name="aspergillus hortai", role="topic")])
        records = [SimpleNamespace(title="Atrazine degradation by Aspergillus niger", abstract="")]
        out = broaden_suggestions(protocol, eligible=0, records=records)
        self.assertEqual([(b.concept, b.broader, b.mentions) for b in out], [("aspergillus hortai", "aspergillus", 0)])
        self.assertEqual(broaden_suggestions(protocol, eligible=12, records=records), [])


class TestSynthesisRule(unittest.TestCase):
    def test_absence_said_once(self):
        from services import synthesis_engine as se
        self.assertIn("say so in one sentence", se.NEAREST_EVIDENCE_FIRST)
        self.assertIn("Do not restate", se.NEAREST_EVIDENCE_LATER)


if __name__ == "__main__":
    unittest.main()
