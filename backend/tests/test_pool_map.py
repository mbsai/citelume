"""Session 4b: the map builds from a stored corpus alone, citation links only by default, with a view of the whole
candidate pool grouped by words; large groups with no included study are flagged."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from models import PaperRecord
from services.pool_map import build_pool_map
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import CONFIG, fake_sources


def records(topic, words, n, start=0):
    return [PaperRecord(id=f"{topic}{i}", title=f"{words} study {i}", year=2020,
                        abstract=f"{words} {words} outcomes in a cohort of patients, report {i}")
            for i in range(start, start + n)]


class TestPoolMap(unittest.TestCase):
    def test_clusters_and_flags(self):
        pool = (records("v", "vitamin D supplementation hip fracture", 12)
                + records("s", "statin therapy myocardial infarction cholesterol", 12)
                + records("e", "exercise training depressive symptoms", 3))
        included = [p.model_copy(update={"citation_index": i + 1}) for i, p in enumerate(pool[:4])]
        m = build_pool_map(pool, included, excluded={"s0": "Off topic"}, ranks={"v5": 25})
        self.assertEqual((m.records, m.included), (27, 4))
        self.assertEqual(sum(c.size for c in m.clusters), 27)
        vit = next(c for c in m.clusters if any(x.id == "v0" for x in c.members))
        stat = next(c for c in m.clusters if any(x.id == "s0" for x in c.members))
        self.assertEqual(vit.included, 4)
        self.assertTrue(vit.members[0].included, "included studies are listed first")
        self.assertTrue(stat.flagged, "a large group with nothing included is flagged")
        self.assertFalse(vit.flagged)
        self.assertIn("no included study", m.note)
        self.assertEqual(next(x for x in stat.members if x.id == "s0").excluded_reason, "Off topic")
        self.assertEqual(next(x for x in vit.members if x.id == "v5").excluded_reason,
                         "Eligible; ranked 25, below the study limit")
        # Same input, same clusters
        again = build_pool_map(pool, included, excluded={"s0": "Off topic"}, ranks={"v5": 25})
        self.assertEqual([c.model_dump() for c in again.clusters], [c.model_dump() for c in m.clusters])

    def test_small_pool_is_one_group(self):
        m = build_pool_map(records("v", "vitamin D", 4), [])
        self.assertEqual(len(m.clusters), 1)
        self.assertFalse(m.clusters[0].flagged)


class TestMapEndpoint(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        golden.synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
        golden.clear_gap_verification_cache()
        self.sources = fake_sources()
        self.sources.__enter__()
        self.addCleanup(self.sources.__exit__, None, None, None)
        self.client = TestClient(main.app)

    def test_map_from_stored_corpus_then_after_analysis(self):
        cid = self.client.post("/api/corpus", json={"query": golden.QUERY, "config": CONFIG}).json()["corpus_id"]
        m = self.client.post(f"/api/corpus/{cid}/map", json={"config": CONFIG}).json()
        g = m["graph"]
        self.assertEqual((m["view"], g["view"]), ("citations", "citations"))
        self.assertTrue({e["network_layer"] for e in g["edges"]} <= {"bibliographic"})
        self.assertGreater(g["pool"]["records"], g["pool"]["included"])
        self.assertEqual(sum(c["size"] for c in g["pool"]["clusters"]), g["pool"]["records"])
        wide = self.client.post(f"/api/corpus/{cid}/map", json={"config": CONFIG, "view": "all"}).json()["graph"]
        self.assertEqual(wide["view"], "all")
        self.assertGreaterEqual(len(wide["edges"]), len(g["edges"]))
        # After the analysis the stored map is the citation view, with the pool
        res = self.client.post(f"/api/corpus/{cid}/analyse", json={"config": CONFIG}).json()
        self.assertEqual(res["graph"]["view"], "citations")
        self.assertIsNotNone(res["graph"]["pool"])
        after = self.client.post(f"/api/corpus/{cid}/map", json={"config": CONFIG, "view": "all"}).json()
        self.assertEqual(after["graph"]["view"], "all")


if __name__ == "__main__":
    unittest.main()
