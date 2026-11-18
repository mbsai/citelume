"""Session 5c: more sensitive searches. Question text is normalised (dashes, word/word), purpose words and class
labels are not required concepts, bracketed abbreviations are synonyms, and PubMed queries carry the concepts' MeSH
headings as an explicit alternative (looked up once, stored with the plan)."""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import SearchConcept, SearchFacet, FacetApiQueries
from services import mesh_terms
from services.europepmc_client import translate_pubmed_query
from services.query_planning import _pubmed_concept_block, build_search_plan
from services.query_utils import normalize_question_text, understand_research_question


def concepts(q):
    u = understand_research_question(q)
    build_search_plan(u, "deep_research", "")
    return [(c.name, c.synonyms) for c in u.concepts.search_concepts]


class TestQuestionParsing(unittest.TestCase):
    def test_normalised_text(self):
        self.assertEqual(normalize_question_text("Non‐pharmacological care for prostatitis/chronic pelvic pain"),
                         "Non-pharmacological care for prostatitis or chronic pelvic pain")
        self.assertEqual(normalize_question_text("dose in mg/kg and/or weight"), "dose in mg/kg or weight")

    def test_review_titles(self):
        self.assertEqual(concepts("Non-pharmacological interventions for treating chronic prostatitis/chronic pelvic "
                                  "pain syndrome"), [("chronic prostatitis", ["chronic pelvic pain syndrome"])])
        self.assertEqual(concepts("Non-invasive positive pressure ventilation for prevention of complications after "
                                  "pulmonary resection")[:2],
                         [("non-invasive positive pressure ventilation", []), ("complications", [])])
        self.assertEqual(concepts("Blue-light filtering intraocular lenses (IOLs) for protecting macular health"),
                         [("blue-light filtering intraocular lenses", ["iols"]), ("macular health", [])])
        # A specific procedure is kept; only class labels such as "... interventions" are dropped
        self.assertEqual(concepts("Cyclodestructive procedures for refractory glaucoma")[0][0], "cyclodestructive procedures")


class TestMesh(unittest.TestCase):
    def test_groups_from_translation(self):
        ts = [{"from": "antidepressants", "to": '"antidepressive agents"[Pharmacological Action] OR '
                                                '"antidepressive agents"[MeSH Terms] OR "antidepressant"[All Fields]'}]
        self.assertEqual(mesh_terms.groups_from_translation(ts, "antidepressants"),
                         ['("antidepressive agents"[Pharmacological Action] OR "antidepressive agents"[MeSH Terms] '
                          'OR "antidepressants"[Title/Abstract])'])
        split = [{"from": "adjuvant", "to": '"adjuvants, pharmaceutic"[MeSH Terms]'},
                 {"from": "gonadotropin-releasing hormone", "to": '"gonadotropin-releasing hormone"[MeSH Terms]'},
                 {"from": "humans", "to": '"humans"[MeSH Terms]'}]
        groups = mesh_terms.groups_from_translation(split, "adjuvant gonadotropin-releasing hormone analogues")
        self.assertEqual(groups, ['("gonadotropin-releasing hormone"[MeSH Terms] OR '
                                  '"gonadotropin-releasing hormone"[Title/Abstract])'])

    def test_queries_rewritten_and_failure_harmless(self):
        c = SearchConcept(name="antidepressants")
        other = SearchConcept(name="insomnia")
        q = f"{_pubmed_concept_block(c)} AND {_pubmed_concept_block(other)}"
        facets = [SearchFacet(facet_code="S1", facet_name="Core", queries=FacetApiQueries(pubmed=q))]

        async def fake_lookup(term, client, api_key=""):
            return ['("antidepressive agents"[Pharmacological Action])'] if term == "antidepressants" else []
        with mock.patch.object(mesh_terms, "lookup", fake_lookup):
            n = asyncio.run(mesh_terms.add_mesh_headings([c, other], facets))
        self.assertEqual(n, 1)
        self.assertEqual(c.mesh, ['("antidepressive agents"[Pharmacological Action])'])
        self.assertIn('"antidepressive agents"[Pharmacological Action]', facets[0].queries.pubmed)
        self.assertTrue(facets[0].queries.pubmed.endswith("AND (insomnia[Title/Abstract])"))
        # Europe PMC (the PubMed mirror) reads the heading too
        self.assertIn('MESH:"antidepressive agents"', translate_pubmed_query(facets[0].queries.pubmed))

        async def broken(*_a, **_k):
            raise RuntimeError("NCBI down")
        c2 = SearchConcept(name="melatonin")
        f2 = [SearchFacet(facet_code="S1", facet_name="Core", queries=FacetApiQueries(pubmed=_pubmed_concept_block(c2)))]
        with mock.patch.object(mesh_terms, "lookup", broken):
            self.assertEqual(asyncio.run(mesh_terms.add_mesh_headings([c2], f2)), 0)
        self.assertEqual(f2[0].queries.pubmed, "(melatonin[Title/Abstract])")


if __name__ == "__main__":
    unittest.main()
