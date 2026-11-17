"""Study limits by review type, scoping selection, and choosing another number from a stored corpus."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from services import corpus_store, demo_mode
from services.paper_selection import REASON_REVIEW
from services.replay import check
from services.study_limits import ceiling, resolve_study_limit, selection_method
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import CONFIG, fake_sources
from tests.test_review_protocol import protocol_for, run_with


class TestLimits(unittest.TestCase):
    def test_defaults_by_review_type(self):
        self.assertEqual(resolve_study_limit("standard", "narrative"), 20)
        self.assertEqual(resolve_study_limit("standard", "systematic_prisma"), 20)  # a quick answer is always 20
        self.assertEqual(resolve_study_limit("deep_research", "narrative"), 60)
        self.assertEqual(resolve_study_limit("deep_research", "systematic_prisma"), 150)
        self.assertEqual(resolve_study_limit("deep_research", "scoping"), 200)
        self.assertEqual(resolve_study_limit("deep_research", "narrative", 500), 150)  # clamped to the ceiling
        self.assertEqual(resolve_study_limit("deep_research", "narrative", 2), 5)
        with mock.patch.object(demo_mode, "demo_enabled", return_value=True):
            self.assertEqual(ceiling("deep_research", "scoping"), 60)
            self.assertEqual(resolve_study_limit("deep_research", "systematic_prisma"), 60)
        self.assertEqual(selection_method("deep_research", "scoping"), "scoping")
        self.assertEqual(selection_method("standard", "scoping"), "narrative")

    def test_protocol_limit_is_applied_and_replays(self):
        p = protocol_for("deep_research")
        p.study_limit = 10
        r = run_with(p, "deep_research")
        self.assertEqual(len(r.papers), 10)
        corpus = corpus_store.load(r.corpus_id)
        self.assertEqual(corpus.study_limit, 10)
        self.assertTrue(check(corpus)[0])

    def test_scoping_includes_every_eligible_study_and_reviews(self):
        p = protocol_for("deep_research", "scoping")
        r = run_with(p, "deep_research", "scoping")
        self.assertEqual(r.selection_report.selection_method, "scoping")
        self.assertTrue(any(x.study_type == "evidence_synthesis" for x in r.papers))
        self.assertNotIn(REASON_REVIEW, r.selection_report.excluded)
        self.assertEqual(len(r.papers), r.selection_report.eligible)  # all eligible, under the ceiling
        self.assertTrue(r.review_protocol.flow and r.review_protocol.flow.assessed_for_eligibility)
        self.assertFalse(any(x.elicit.risk_of_bias for x in r.papers))  # no risk of bias in a scoping review
        self.assertTrue(check(corpus_store.load(r.corpus_id))[0])


class TestReselect(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        self.client = TestClient(main.app)

    def create(self, mode, framework):
        with fake_sources():
            r = self.client.post("/api/corpus", json={"query": golden.QUERY, "mode": mode,
                                                      "review_framework": framework, "config": CONFIG})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_choose_fewer_studies_without_searching_again(self):
        s = self.create("deep_research", "narrative")
        self.assertEqual(s["study_limit"], 60)
        self.assertGreater(s["max_study_limit"], 60)
        with mock.patch.object(main, "_build_corpus", side_effect=AssertionError("searched again")):
            r = self.client.post(f"/api/corpus/{s['corpus_id']}/reselect", json={"study_limit": 10})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(len(d["included"]), 10)
        self.assertNotEqual(d["corpus_id"], s["corpus_id"])
        derived = corpus_store.load(d["corpus_id"])
        self.assertEqual(derived.root_id, s["corpus_id"])
        self.assertEqual(derived.records, corpus_store.load(s["corpus_id"]).records)
        self.assertTrue(check(derived)[0])
        self.assertEqual(len(corpus_store.load(s["corpus_id"]).included_ids), len(s["included"]))  # unchanged

    def test_systematic_review_can_only_go_down(self):
        s = self.create("deep_research", "systematic_prisma")
        n = len(s["included"])
        r = self.client.post(f"/api/corpus/{s['corpus_id']}/reselect", json={"study_limit": 8})
        self.assertEqual([p["id"] for p in r.json()["included"]], [p["id"] for p in s["included"]][:8])
        self.assertTrue(check(corpus_store.load(r.json()["corpus_id"]))[0])
        too_many = self.client.post(f"/api/corpus/{s['corpus_id']}/reselect", json={"study_limit": s["study_limit"] + 1})
        self.assertEqual(too_many.status_code, 400)
        self.assertGreaterEqual(s["study_limit"], n)

    def test_demo_counts_analyses_on_the_original_search(self):
        s = self.create("standard", "narrative")
        d = self.client.post(f"/api/corpus/{s['corpus_id']}/reselect", json={"study_limit": 8}).json()
        e = self.client.post(f"/api/corpus/{s['corpus_id']}/reselect", json={"study_limit": 9}).json()
        with mock.patch.object(demo_mode, "demo_enabled", return_value=True), \
                mock.patch.object(demo_mode, "apply_demo_config", side_effect=lambda c: (c, True)), \
                mock.patch.object(demo_mode, "take_quota", return_value=None), fake_sources():
            first = self.client.post(f"/api/corpus/{d['corpus_id']}/analyse", json={"config": CONFIG})
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post(f"/api/corpus/{e['corpus_id']}/analyse", json={"config": CONFIG})
            self.assertEqual(second.status_code, 429)


if __name__ == "__main__":
    unittest.main()
