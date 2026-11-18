"""Session 5d: eligibility reads PubMed's MeSH indexing as well as the words, studies matching only the main concepts
are screened rather than excluded by words, and records without an abstract are judged on their titles."""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import PaperRecord, SearchConcept
from services import mesh_terms
from services.paper_selection import REASON_NO_ABSTRACT, _concept_level
from tests.test_prisma_review import paper, select, trial


class FakeResponse:
    def __init__(self, ids):
        self.ids = ids

    def raise_for_status(self):
        pass

    def json(self):
        return {"esearchresult": {"idlist": self.ids}}


class FakeClient:
    """PubMed esearch: answers which of the asked PubMed IDs a MeSH term matches."""
    indexed = {"gnrh": {"11", "33"}}
    calls = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, data):
        FakeClient.calls.append(data["term"])
        asked = {x.split("[")[0] for x in data["term"].split("AND (", 1)[1].rstrip(")").split(" OR ")}
        hits = set().union(*(ids for key, ids in self.indexed.items() if key in data["term"]))
        return FakeResponse(sorted(asked & hits))


class TestMeshMatching(unittest.TestCase):
    def test_concept_level_reads_mesh(self):
        c = SearchConcept(name="gonadotropin-releasing hormone analogues")
        title, abstract = "Goserelin during chemotherapy for breast cancer", "Ovarian function was preserved."
        self.assertEqual(_concept_level(c, title, abstract), 0.0)
        self.assertEqual(_concept_level(c, title, abstract, [c.name]), 0.7)
        self.assertEqual(_concept_level(c, "GnRH analogues: gonadotropin-releasing hormone analogues", "", [c.name]), 1.0)

    def test_records_marked_by_pubmed_indexing(self):
        c = SearchConcept(name="gnrh analogues", mesh=['("gnrh"[MeSH Terms])'])
        plain = SearchConcept(name="ovarian failure")  # no MeSH: never asked
        recs = [PaperRecord(id="a", title="t", pmid="11"), PaperRecord(id="b", title="t", pmid="22"),
                PaperRecord(id="c", title="t"), PaperRecord(id="d", title="t", pmid="33")]
        FakeClient.calls = []
        with mock.patch.object(mesh_terms.httpx, "AsyncClient", FakeClient):
            n = asyncio.run(mesh_terms.mark_mesh_concepts(recs, [c, plain], batch=2))
        self.assertEqual(n, 2)
        self.assertEqual([r.mesh_concepts for r in recs], [["gnrh analogues"], [], [], ["gnrh analogues"]])
        self.assertEqual(len(FakeClient.calls), 2, "one request per concept per batch of IDs")


class TestTitleOnlyRecords(unittest.TestCase):
    def test_kept_when_the_title_matches(self):
        title_only = paper(90, "Vitamin D supplementation and fractures in older adults: a randomised trial", None)
        off = paper(91, "Calcium intake in adolescents", None)
        chosen, rep, _, _ = select([trial(i) for i in range(1, 8)] + [title_only, off])
        ids = [p.id for p in chosen]
        self.assertIn(title_only.id, ids)
        self.assertNotIn(off.id, ids)
        self.assertIsNone(rep.excluded.get(REASON_NO_ABSTRACT))
        kept = next(p for p in chosen if p.id == title_only.id)
        self.assertIn("assessed on title only", kept.relevance_note)


if __name__ == "__main__":
    unittest.main()


class TestUnscreenedPartialMatches(unittest.TestCase):
    def test_without_a_model_partial_matches_stay_excluded(self):
        from models import ApiKeysConfig, SelectionReport
        from services.ai_screening import screen_and_limit
        from services.paper_selection import REASON_PARTIAL
        full = [PaperRecord(id=f"f{i}", title="t", relevance="addresses") for i in range(3)]
        part = [PaperRecord(id=f"p{i}", title="t", relevance="partly") for i in range(2)]
        rep = SelectionReport()
        chosen = asyncio.run(screen_and_limit(full + part, rep, "q", ApiKeysConfig(llm_api_key=""), limit=10))
        self.assertEqual([p.id for p in chosen], ["f0", "f1", "f2"])
        self.assertEqual(rep.excluded.get(REASON_PARTIAL), 2)
        self.assertFalse(any(d.screened for d in rep.screening_decisions))
