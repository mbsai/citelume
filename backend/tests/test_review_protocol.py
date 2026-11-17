"""The review protocol (Session 3b): it drives the searches and selection, its default changes nothing, edits
change what they should, and the same protocol and records always give the same studies."""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from models import ApiKeysConfig, ProtocolConcept, ProtocolConstraint, SearchCount, SearchRequest
from services import corpus_store, search_counts
from services.llm_service import set_test_llm_responder
from services.paper_selection import REASON_DESIGN, REASON_REVIEW_BY_PROTOCOL, REASON_YEARS
from services.query_utils import compile_deterministic_decomposition_plan
from services.replay import check
from services.review_protocol import plan_from_protocol, propose_protocol, recommend_review_type
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import CONFIG, fake_sources

GOLDEN = json.loads(golden.GOLDEN.read_text())


def protocol_for(mode="standard", framework="narrative"):
    return asyncio.run(propose_protocol(golden.QUERY, mode=mode, review_framework=framework))


def run_with(protocol, mode="standard", framework="narrative"):
    golden.query_utils._DECOMPOSITION_CACHE.clear()
    golden.synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
    golden.clear_gap_verification_cache()
    req = SearchRequest(query=golden.QUERY, mode=mode, review_framework=framework, protocol=protocol,
                        config=ApiKeysConfig(**CONFIG))
    with fake_sources():
        return asyncio.run(main._run_research_search(req))


class TestDefaultProtocol(unittest.TestCase):
    def test_default_protocol_compiles_the_same_searches(self):
        for q in [golden.QUERY, "How does sleep affect memory in older adults?", "gut microbiome",
                  "coronary artery disease in India", "machine learning in drug discovery"]:
            for mode in ("standard", "deep_research"):
                p = asyncio.run(propose_protocol(q, mode=mode))
                a = compile_deterministic_decomposition_plan(q, mode)
                b = plan_from_protocol(p, mode)
                self.assertEqual([f.queries.model_dump() for f in a.search_facets],
                                 [f.queries.model_dump() for f in b.search_facets], (q, mode))

    def test_running_with_the_default_protocol_changes_nothing(self):
        for case, (mode, framework) in {"standard": ("standard", "narrative"),
                                        "prisma": ("deep_research", "systematic_prisma")}.items():
            result = run_with(protocol_for(mode, framework), mode, framework)
            self.assertEqual(golden.snapshot(result), GOLDEN[case], case)
            self.assertEqual(result.decomposition_plan.question_understanding.planned_by, "default review protocol")

    def test_every_corpus_records_its_protocol(self):
        result = golden.run_search()
        corpus = corpus_store.load(result.corpus_id)
        self.assertIsNotNone(corpus.protocol)
        self.assertEqual([c.name for c in corpus.protocol.concepts], ["vitamin D supplementation", "fractures"])
        self.assertEqual([c.name for c in corpus.protocol.populations_and_places], ["older adults"])

    def test_review_type_follows_the_purpose(self):
        self.assertEqual(recommend_review_type("effect")[0], "systematic_prisma")
        self.assertEqual(recommend_review_type("map")[0], "scoping")
        self.assertIsNone(recommend_review_type("something else"))
        quick = asyncio.run(propose_protocol(golden.QUERY, mode="standard", purpose="effect"))
        self.assertEqual(quick.review_type, "narrative")  # a quick answer is always narrative
        deep = asyncio.run(propose_protocol(golden.QUERY, mode="deep_research", purpose="effect"))
        self.assertEqual(deep.review_type, "systematic_prisma")
        self.assertIn("Recommended because", deep.review_type_reason)


class TestEditedProtocol(unittest.TestCase):
    def ids(self, result):
        return [p.id for p in result.papers]

    def test_a_synonym_changes_the_search_strings(self):
        p = protocol_for()
        p.concepts[0].synonyms.append("cholecalciferol")
        p.concepts[0].set_by = "user"
        plan = plan_from_protocol(p, "standard")
        self.assertIn("cholecalciferol", plan.search_facets[0].queries.pubmed)
        self.assertEqual(plan.question_understanding.planned_by, "review protocol (user)")

    def test_changing_the_population_changes_the_selection(self):
        p = protocol_for()
        p.populations_and_places = [ProtocolConstraint(name="children", spellings=["children", "child"], set_by="user")]
        result = run_with(p)
        self.assertTrue(result.papers)
        self.assertTrue(all("children" in r.title for r in result.papers), [r.title for r in result.papers])

    def test_eligibility_rules(self):
        p = protocol_for()
        p.eligibility.study_designs, p.eligibility.changed_by_user = ["rct"], ["study_designs"]
        r = run_with(p)
        self.assertTrue(r.papers and all(x.study_type == "rct" for x in r.papers))
        self.assertIn(REASON_DESIGN, r.selection_report.excluded)

        p = protocol_for()
        p.eligibility.year_from = golden.YEAR - 2
        r = run_with(p)
        self.assertTrue(all(x.year >= golden.YEAR - 2 for x in r.papers))
        self.assertIn(REASON_YEARS, r.selection_report.excluded)

        p = protocol_for()
        p.eligibility.reviews = "exclude"
        r = run_with(p)
        self.assertFalse(any(x.study_type in ("review", "evidence_synthesis") for x in r.papers))
        self.assertIn(REASON_REVIEW_BY_PROTOCOL, r.selection_report.excluded)

        p = protocol_for("deep_research", "systematic_prisma")
        p.eligibility.reviews = "include"
        r = run_with(p, "deep_research", "systematic_prisma")
        self.assertTrue(any(x.study_type == "evidence_synthesis" for x in r.papers))

    def test_comparisons_are_recorded_but_not_searched_or_required(self):
        p = protocol_for()
        p.concepts.append(ProtocolConcept(role="comparison", name="placebo", set_by="user"))
        plan = plan_from_protocol(p, "standard")
        self.assertNotIn("placebo", plan.search_facets[0].queries.pubmed)
        self.assertEqual(self.ids(run_with(p)), GOLDEN["standard"]["included_ids"])

    def test_same_protocol_same_studies_and_replay(self):
        p = protocol_for()
        p.eligibility.study_designs = ["rct", "controlled_observational"]
        a, b = run_with(p), run_with(p)
        self.assertEqual(self.ids(a), self.ids(b))
        self.assertEqual(a.corpus_hash, b.corpus_hash)
        ok, lines = check(corpus_store.load(a.corpus_id))
        self.assertTrue(ok, lines)
        # A different protocol is a different selection, even on the same records
        self.assertNotEqual(a.corpus_hash, run_with(protocol_for()).corpus_hash)

    def test_with_a_protocol_no_model_plans_the_searches(self):
        def responder(system_prompt, *_):
            if system_prompt.startswith("You plan literature searches"):
                raise AssertionError("the model was asked to plan the searches")
            return None
        set_test_llm_responder(responder)
        try:
            p = protocol_for()
            golden.query_utils._DECOMPOSITION_CACHE.clear()
            req = SearchRequest(query=golden.QUERY, protocol=p,
                                config=ApiKeysConfig(**{**CONFIG, "llm_api_key": "test-key"}))
            with fake_sources():
                result = asyncio.run(main._run_research_search(req))
            self.assertEqual(self.ids(result), GOLDEN["standard"]["included_ids"])
        finally:
            set_test_llm_responder(None)


class TestSuggestionsAndEndpoints(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        self.client = TestClient(main.app)

    def test_model_suggestions_are_offered_not_applied(self):
        def responder(system_prompt, *_):
            if system_prompt.startswith("You plan literature searches"):
                return json.dumps({"concepts": [{"name": "vitamin D", "synonyms": ["cholecalciferol", "vitamin D3"]},
                                                {"name": "fracture", "synonyms": ["bone fracture"]}],
                                   "breadth": "focused",
                                   "aspects": [{"label": "Dose", "terms": ["dose", "IU"], "question": "q"},
                                               {"label": "Falls", "terms": ["falls", "fall risk"], "question": "q"}]})
            return None
        set_test_llm_responder(responder)
        try:
            p = asyncio.run(propose_protocol(golden.QUERY, suggest=True,
                                             config=ApiKeysConfig(llm_provider="openrouter", llm_api_key="k")))
        finally:
            set_test_llm_responder(None)
        values = {(s.kind, s.value) for s in p.suggestions}
        self.assertIn(("synonym", "cholecalciferol"), values)
        self.assertIn(("aspect", "Dose"), values)
        self.assertNotIn(("concept", "fractures"), values)  # same concept as the protocol's "fractures"
        self.assertEqual(p.concepts[0].synonyms, [])  # nothing was applied

    def test_propose_and_preview_endpoints(self):
        r = self.client.post("/api/protocol/propose", json={"query": golden.QUERY, "mode": "deep_research",
                                                            "purpose": "effect"})
        self.assertEqual(r.status_code, 200, r.text)
        protocol = r.json()
        self.assertEqual(protocol["review_type"], "systematic_prisma")

        async def fake_count(*_args, **_kwargs):
            return SearchCount(source="PubMed", query="q", count=1234)
        with mock.patch.object(search_counts, "count_pubmed", fake_count), \
                mock.patch.object(search_counts, "count_semantic_scholar", fake_count), \
                mock.patch.object(search_counts, "count_preprints", fake_count), \
                mock.patch.object(search_counts, "count_arxiv", fake_count):
            r = self.client.post("/api/protocol/preview", json={"protocol": protocol, "mode": "deep_research"})
        self.assertEqual(r.status_code, 200, r.text)
        preview = r.json()
        self.assertEqual(len(preview["facets"]), 8)
        self.assertTrue(all(c["count"] == 1234 for c in preview["core_counts"]))
        self.assertIn("retrieves the first 100", preview["core_counts"][0]["note"])

        protocol["concepts"] = []
        r = self.client.post("/api/protocol/preview", json={"protocol": protocol})
        self.assertEqual(r.status_code, 400)

    def test_corpus_endpoint_accepts_and_stores_a_protocol(self):
        p = protocol_for().model_dump()
        p["eligibility"]["study_designs"] = ["rct"]
        with fake_sources():
            r = self.client.post("/api/corpus", json={"query": golden.QUERY, "protocol": p, "config": CONFIG})
        self.assertEqual(r.status_code, 200, r.text)
        corpus = corpus_store.load(r.json()["corpus_id"])
        self.assertEqual(corpus.protocol.eligibility.study_designs, ["rct"])
        self.assertTrue(all(x["study_type"] == "rct" for x in r.json()["included"]))


if __name__ == "__main__":
    unittest.main()
