"""Session 5b: each included study is followed through the stages and its loss attributed to the right one."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from eval.loss_by_stage import follow, reached  # noqa: E402
from models import Corpus, CorpusCandidate, PaperRecord, ScreeningDecisionRecord, SelectionReport  # noqa: E402


def gold(pmid):
    return {"pmid": pmid, "doi": "", "title": f"Included trial number {pmid} of something"}


class TestFollow(unittest.TestCase):
    def test_stages_and_reasons(self):
        included = [gold(str(i)) for i in range(1, 9)]
        raw = [PaperRecord(id=f"r{i}", title="t", pmid=str(i), source="PubMed") for i in (2, 3, 4, 5, 6, 7)]
        raw.append(PaperRecord(id="c8", title="t", pmid="8", source="Citation searching"))
        # 2 was merged away (its record carries another ID); 3 failed eligibility
        records = [PaperRecord(id=f"r{i}", title="t", pmid=str(i)) for i in (3, 4, 5, 6, 7)]
        records.append(PaperRecord(id="c8", title="t", pmid="8"))
        corpus = Corpus(
            records=records,
            excluded={"r3": "Outside the place or population asked about"},
            candidates=[CorpusCandidate(record_id=r, rank=k) for k, r in
                        ((1, "r4"), (30, "r5"), (70, "r6"), (90, "r7"), (5, "c8"))],
            included_ids=["r4", "c8"],
            selection_report=SelectionReport(screening_decisions=[
                ScreeningDecisionRecord(record_id="r4", include=True),
                ScreeningDecisionRecord(record_id="c8", include=True),
                ScreeningDecisionRecord(record_id="r5", include=False, failed_criterion="Population"),
                ScreeningDecisionRecord(record_id="r6", include=True),
            ]),
        )
        rows = {r["pmid"]: r for r in follow(included, corpus, raw)}
        self.assertEqual((rows["1"]["stage"], rows["1"]["reason"]), ("missed", "not returned by any search"))
        self.assertEqual((rows["2"]["stage"], rows["2"]["reason"]), ("found", "lost when duplicates were merged"))
        self.assertEqual((rows["3"]["stage"], rows["3"]["reason"]),
                         ("merged", "Outside the place or population asked about"))
        self.assertEqual(rows["4"]["stage"], "included")
        self.assertEqual((rows["5"]["stage"], rows["5"]["reason"]), ("eligible", "screened out: Population"))
        self.assertEqual((rows["6"]["stage"], rows["6"]["reason"]), ("screened_in", "beyond the study limit"))
        self.assertEqual((rows["7"]["stage"], rows["7"]["reason"]),
                         ("eligible", "not screened: the study limit was filled first"))
        self.assertTrue(rows["8"]["found_by_chasing"] and not rows["8"]["found_by_search"])
        self.assertEqual(rows["8"]["stage"], "included")
        self.assertEqual(reached(list(rows.values())),
                         {"found": 7, "merged": 6, "eligible": 5, "screened_in": 3, "included": 2})


if __name__ == "__main__":
    unittest.main()
