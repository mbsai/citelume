"""The corpus and per-module endpoints, on the golden test's fixed fake database results (no network)."""
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from services import corpus_store, demo_mode
from services.llm_service import set_test_llm_responder
from tests import test_golden_search as golden

GOLDEN = json.loads(golden.GOLDEN.read_text())


def fake_sources() -> ExitStack:
    stack = ExitStack()
    for name, source in (("search_pubmed", "PubMed"), ("search_semantic_scholar", "Semantic Scholar"),
                         ("search_biorxiv", "bioRxiv"), ("search_arxiv", "arXiv")):
        stack.enter_context(mock.patch.object(main, name, golden._by_source(source)))
    stack.enter_context(mock.patch.object(main, "search_openalex", golden._empty))
    stack.enter_context(mock.patch.object(main, "search_clinicaltrials", golden._empty))
    stack.enter_context(mock.patch.object(main, "enrich_citation_counts", golden._none))
    stack.enter_context(mock.patch.object(main, "expand_paper_neighbors", golden._empty))
    stack.enter_context(mock.patch.object(main.citation_chasing, "_post_batch", golden._empty))
    stack.enter_context(mock.patch.object(main, "add_mesh_headings", golden._none))
    stack.enter_context(mock.patch.object(main, "mark_mesh_concepts", golden._none))
    return stack


CONFIG = {"llm_provider": "openrouter", "llm_api_key": "", "enable_hf": False, "enable_fulltext": False}


class TestCorpusEndpoints(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        golden.synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
        golden.clear_gap_verification_cache()
        set_test_llm_responder(None)
        self.sources = fake_sources()
        self.sources.__enter__()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.sources.__exit__(None, None, None)

    def create(self, mode="standard", framework="narrative"):
        r = self.client.post("/api/corpus", json={"query": golden.QUERY, "mode": mode,
                                                  "review_framework": framework, "config": CONFIG})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_create_and_read_a_corpus(self):
        summary = self.create()
        self.assertFalse(summary["analysed"])
        self.assertEqual([p["id"] for p in summary["included"]], GOLDEN["standard"]["included_ids"])
        self.assertEqual(summary["records_found"], GOLDEN["standard"]["records_found"])
        corpus = self.client.get(f"/api/corpus/{summary['corpus_id']}").json()
        self.assertEqual(corpus["included_ids"], GOLDEN["standard"]["included_ids"])
        self.assertEqual(corpus["content_hash"], summary["corpus_hash"])
        self.assertEqual(len(corpus["records"]), summary["unique_records"])

    def test_the_map_needs_no_analysis(self):
        summary = self.create()
        r = self.client.post(f"/api/corpus/{summary['corpus_id']}/map", json={"config": CONFIG})
        self.assertEqual(r.status_code, 200, r.text)
        papers = [n for n in r.json()["graph"]["nodes"] if n["node_type"] == "paper"]
        self.assertEqual(len(papers), len(summary["included"]))
        self.assertIsNone(corpus_store.load(summary["corpus_id"]).analysis)  # nothing was analysed

    def test_modules_share_one_stored_analysis_that_matches_a_full_search(self):
        for case, (mode, framework) in {"standard": ("standard", "narrative"),
                                        "prisma": ("deep_research", "systematic_prisma")}.items():
            summary = self.create(mode, framework)
            cid = summary["corpus_id"]
            table = self.client.post(f"/api/corpus/{cid}/evidence-table", json={"config": CONFIG}).json()
            self.assertEqual([p["id"] for p in table["papers"]], GOLDEN[case]["included_ids"])
            self.assertEqual(len(table["evidence_matrix"]["rows"]), GOLDEN[case]["evidence_rows"])
            # The other modules read the stored analysis instead of running the models again
            with mock.patch.object(main, "_analyse", side_effect=AssertionError("analysis ran twice")):
                synthesis = self.client.post(f"/api/corpus/{cid}/synthesis", json={"config": CONFIG}).json()
                gaps = self.client.post(f"/api/corpus/{cid}/gaps", json={"config": CONFIG}).json()
                graph = self.client.post(f"/api/corpus/{cid}/map", json={"config": CONFIG}).json()
            self.assertEqual([s["heading"] for s in synthesis["sections"]], GOLDEN[case]["section_headings"])
            self.assertEqual([g["gap_id"] for g in gaps["gap_matrix"]["structured_gaps"]], GOLDEN[case]["gap_ids"])
            self.assertEqual(sum(1 for n in graph["graph"]["nodes"] if n["node_type"] == "paper"),
                             GOLDEN[case]["graph_paper_nodes"])
            # The full analysis of a stored corpus gives what one /api/search call gives
            full = self.client.post(f"/api/corpus/{cid}/analyse", json={"config": CONFIG}).json()
            snap = golden.snapshot(main.ResearchResponse(**full))
            self.assertEqual(snap, GOLDEN[case], case)

    def test_search_still_works_and_returns_its_corpus(self):
        r = self.client.post("/api/search", json={"query": golden.QUERY, "mode": "standard", "config": CONFIG})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual([p["id"] for p in body["papers"]], GOLDEN["standard"]["included_ids"])
        corpus = corpus_store.load(body["corpus_id"])
        self.assertIsNotNone(corpus.analysis)
        self.assertEqual(corpus.content_hash, body["corpus_hash"])

    def test_unknown_or_malformed_ids(self):
        self.assertEqual(self.client.get("/api/corpus/" + "0" * 32).status_code, 404)
        self.assertEqual(self.client.get("/api/corpus/not-an-id").status_code, 404)
        self.assertEqual(self.client.post("/api/corpus/" + "0" * 32 + "/gaps", json={}).status_code, 404)

    def test_a_corpus_whose_selection_cannot_be_reproduced_is_refused(self):
        summary = self.create()
        corpus = corpus_store.load(summary["corpus_id"])
        corpus.included_ids = list(reversed(corpus.included_ids))
        corpus_store.save(corpus)
        r = self.client.post(f"/api/corpus/{summary['corpus_id']}/synthesis", json={"config": CONFIG})
        self.assertEqual(r.status_code, 409)

    def test_demo_allows_one_shared_key_analysis_per_corpus(self):
        summary = self.create()
        cid = summary["corpus_id"]
        with mock.patch.object(demo_mode, "demo_enabled", return_value=True), \
                mock.patch.object(demo_mode, "apply_demo_config", side_effect=lambda c: (c, True)), \
                mock.patch.object(demo_mode, "take_quota", return_value=None) as charged:
            first = self.client.post(f"/api/corpus/{cid}/synthesis", json={"config": CONFIG})
            self.assertEqual(first.status_code, 200, first.text)
            charged.assert_not_called()  # the first analysis is covered by building the corpus
            again = self.client.post(f"/api/corpus/{cid}/synthesis", json={"config": CONFIG, "rerun": True})
            self.assertEqual(again.status_code, 429)


if __name__ == "__main__":
    unittest.main()
