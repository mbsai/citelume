"""Session 5: the committed benchmark is well formed, its split is the pre-registered seeded draw, test reviews are
held out, and included studies are matched to corpus records as eval/PROTOCOL.md defines."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from eval import scoring  # noqa: E402
from models import PaperRecord  # noqa: E402

SEED = 20261118


class TestBenchmarkFiles(unittest.TestCase):
    def test_reviews_and_split(self):
        reviews = scoring.load_reviews()
        split = scoring.load_split()
        self.assertGreaterEqual(len(reviews), 30)
        self.assertEqual(scoring.check_split(reviews, split, SEED), [])
        self.assertEqual(len(split["development"]), len(reviews) // 3)
        for r in reviews.values():
            self.assertTrue(r["question"] and r["year_to"] and r["review_version"]["doi"])
            self.assertGreaterEqual(r["included_resolved"], 2)
            self.assertTrue(all(s["pmid"].isdigit() for s in r["included"]))
            self.assertTrue(all((s["year"] or 0) <= r["year_to"] for s in r["included"]))
        excluded = (scoring.BENCHMARK / "EXCLUDED.md").read_text()
        self.assertTrue(all(rid not in excluded for rid in reviews))

    def test_test_set_is_held_out(self):
        with self.assertRaises(PermissionError):
            scoring.review_split("test")
        self.assertEqual(len(scoring.review_split("development")), len(scoring.load_split()["development"]))


class TestMatching(unittest.TestCase):
    def test_match_rules(self):
        included = [
            {"pmid": "111", "doi": "10.1/a", "title": "Trial A of vitamin D in older adults"},
            {"pmid": "222", "doi": "10.1/B", "title": "Trial B of vitamin D in older adults"},
            {"pmid": "333", "doi": "", "title": "Trial C: vitamin D and falls in care homes"},
            {"pmid": "444", "doi": "", "title": "Trial D, never found by any search at all"},
            {"pmid": "555", "doi": "", "title": "Trial E title shared with a record that has a DOI"},
        ]
        records = [
            PaperRecord(id="r1", title="Different title", pmid="111"),
            {"id": "r2", "title": "x", "doi": "https://doi.org/10.1/b"},
            PaperRecord(id="r3", title="trial C - Vitamin D and falls in care homes!"),
            PaperRecord(id="r5", title="Trial E title shared with a record that has a DOI", doi="10.9/other"),
        ]
        m = scoring.match(included, records)
        self.assertEqual([(f["pmid"], f["record_id"], f["how"]) for f in m.found],
                         [("111", "r1", "pmid"), ("222", "r2", "doi"), ("333", "r3", "title")])
        self.assertEqual([s["pmid"] for s in m.missing], ["444", "555"])  # titles match only records with no IDs
        self.assertAlmostEqual(m.recall, 3 / 5)
        self.assertEqual(scoring.match([], records).recall, 0.0)


if __name__ == "__main__":
    unittest.main()
