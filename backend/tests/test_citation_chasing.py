"""Session 4b: citation chasing adds papers linked to several selected studies as records identified by other
methods; they pass the same selection as searched records, are counted separately in the PRISMA flow, and the
stored corpus still replays without the network."""
import asyncio
import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from models import PaperRecord
from services import citation_chasing as cc
from services import corpus_store
from services.replay import check
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import CONFIG, fake_sources

CHASED = "c" * 40
CHASED_TITLE = "Vitamin D supplementation and fracture prevention in older adults: a randomised controlled trial"
CHASED_ABSTRACT = ("In this randomised controlled trial of 2,000 older adults, daily vitamin D supplementation "
                   "reduced the incidence of hip and non-vertebral fractures compared with placebo over three years.")


def fake_batch(calls=None, details=None):
    """A Semantic Scholar batch endpoint where every seed cites the same paper (CHASED) and one paper of its own."""
    async def post(ids, fields, api_key):
        if calls is not None:
            calls.append((fields, list(ids)))
        if fields == cc.LINK_FIELDS:
            return [{"paperId": hashlib.sha1(sid.encode()).hexdigest(),
                     "references": [{"paperId": CHASED}, {"paperId": hashlib.sha1(b"own" + sid.encode()).hexdigest()}],
                     "citations": []} for sid in ids]
        return details if details is not None else [
            {"paperId": pid, "title": CHASED_TITLE if pid == CHASED else f"Unrelated paper {pid[:6]}",
             "abstract": CHASED_ABSTRACT if pid == CHASED else "", "year": 2019, "externalIds": {"DOI": f"10.9999/{pid[:8]}"},
             "publicationTypes": ["JournalArticle", "ClinicalTrial"]} for pid in ids]
    return post


def rec(i, **kw):
    return PaperRecord(id=f"p{i}", title=f"Study number {i} about vitamin D", doi=f"10.1/{i}", **kw)


class TestChasingRules(unittest.TestCase):
    def test_choose_candidates(self):
        links = {"a": (["x", "y"], ["z"]), "b": (["x"], ["z", "w"]), "c": (["x", "v"], [])}
        self.assertEqual(cc.choose_candidates(links, set(), min_links=2), [("x", 3), ("z", 2)])
        self.assertEqual(cc.choose_candidates(links, {"x"}, min_links=2), [("z", 2)])
        # A paper both citing and cited by one seed counts once for that seed
        self.assertEqual(cc.choose_candidates({"a": (["q"], ["q"])}, set(), min_links=2), [])

    def test_seed_identifiers(self):
        self.assertEqual(cc.seed_identifier(rec(1)), "DOI:10.1/1")
        self.assertEqual(cc.seed_identifier(PaperRecord(id="x", title="t", pmid="123")), "PMID:123")
        self.assertEqual(cc.seed_identifier(PaperRecord(id="x", title="t", arxiv_id="2101.00001v2")), "ARXIV:2101.00001")
        self.assertIsNone(cc.seed_identifier(PaperRecord(id="x", title="t")))

    def test_chase_adds_only_new_linked_papers(self):
        seeds = [rec(i) for i in range(3)]
        calls = []
        with mock.patch.object(cc, "_post_batch", fake_batch(calls)):
            res = asyncio.run(cc.chase_citations(seeds, seeds))
        self.assertEqual([p.s2_id for p in res.added], [CHASED])
        added = res.added[0]
        self.assertEqual((added.source, added.all_sources), (cc.SOURCE_LABEL, [cc.SOURCE_LABEL]))
        self.assertIn("linked to 3 selected studies", added.retrieved_by[0])
        r = res.report
        self.assertEqual((r.seeds, r.seeds_found, r.seeds_with_references, r.linked_candidates, r.added, r.requests),
                         (3, 3, 3, 1, 1, 2))
        self.assertEqual(res.seed_links["p0"]["references"][0], f"s2:{CHASED}")
        # Already in the pool by DOI: not added again
        in_pool = seeds + [PaperRecord(id="dup", title="x", doi=f"10.9999/{CHASED[:8]}")]
        with mock.patch.object(cc, "_post_batch", fake_batch()):
            res = asyncio.run(cc.chase_citations(seeds, in_pool))
        self.assertEqual((res.report.added, res.report.already_in_pool), (0, 1))

    def test_missing_abstracts_filled_from_pubmed(self):
        recs = [PaperRecord(id="a", title="t", pmid="11"), PaperRecord(id="b", title="t", pmid="12", abstract="x" * 80),
                PaperRecord(id="c", title="t")]

        async def details(pmids, api_key):
            self.assertEqual(pmids, ["11", "12"])  # b needs its publication types; c has no PubMed ID
            return {"11": {"abstract": "An abstract from PubMed.", "publication_types": ["Randomized Controlled Trial"]},
                    "12": {"abstract": "ignored", "publication_types": ["Review"]}}
        with mock.patch.object(cc, "fetch_pubmed_details", details):
            self.assertEqual(asyncio.run(cc.complete_from_pubmed(recs)), 1)
        self.assertEqual((recs[0].abstract, recs[0].publication_types), ("An abstract from PubMed.", ["Randomized Controlled Trial"]))
        self.assertEqual((recs[1].abstract, recs[1].publication_types), ("x" * 80, ["Review"]))

    def test_review_reference_lists_are_backward_only(self):
        async def post(ids, fields, api_key):
            if fields == cc.LINK_FIELDS:
                return [{"paperId": "f" * 40, "references": [{"paperId": "a" * 40}], "citations": [{"paperId": "b" * 40}]}]
            return [{"paperId": pid, "title": f"Trial {pid[:4]} of something specific", "abstract": "x"} for pid in ids]
        with mock.patch.object(cc, "_post_batch", post):
            res = asyncio.run(cc.chase_citations([rec(1)], [rec(1)], min_links=1, backward_only=True,
                                                 label=cc.REVIEW_REFS_LABEL))
        self.assertEqual([p.s2_id for p in res.added], ["a" * 40])
        self.assertEqual(res.added[0].source, cc.REVIEW_REFS_LABEL)
        self.assertIn("cited by 1 review", res.added[0].retrieved_by[0])

    def test_failure_is_reported_not_raised(self):
        async def broken(*_a, **_k):
            raise RuntimeError("rate limited")
        with mock.patch.object(cc, "_post_batch", broken):
            res = asyncio.run(cc.chase_citations([rec(1)], [rec(1)]))
        self.assertEqual(res.added, [])
        self.assertIn("stopped early", res.report.note)

    def test_reference_data_reaches_every_raw_record_of_a_seed(self):
        seed = rec(1)
        raw = [rec(1), PaperRecord(id="pm", title=seed.title, pmid="55"), rec(2)]
        n = cc.attach_reference_data(raw, [seed], {"p1": {"s2_id": "abc", "references": ["s2:r"], "cited_by": []}})
        self.assertEqual(n, 2)
        self.assertEqual((raw[0].s2_id, raw[0].references, raw[1].references, raw[2].references),
                         ("abc", ["s2:r"], ["s2:r"], []))


class TestChasingInThePipeline(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        golden.synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
        golden.clear_gap_verification_cache()
        self.sources = fake_sources()
        self.sources.__enter__()
        self.addCleanup(self.sources.__exit__, None, None, None)
        self.client = TestClient(main.app)

    def create(self, **extra):
        with mock.patch.object(cc, "_post_batch", fake_batch()):
            r = self.client.post("/api/corpus", json={"query": golden.QUERY, "config": CONFIG, **extra})
        self.assertEqual(r.status_code, 200, r.text)
        return corpus_store.load(r.json()["corpus_id"])

    def test_off_for_the_quick_answer_unless_asked(self):
        self.assertIsNone(self.create().citation_chasing)
        corpus = self.create(citation_chasing=True)
        self.assertEqual(corpus.citation_chasing.added, 1)

    def test_chased_record_is_selected_like_any_other_and_replays(self):
        corpus = self.create(mode="deep_research", review_framework="narrative")
        report = corpus.citation_chasing
        self.assertEqual(report.added, 1)
        chased = [p for p in corpus.records if p.source == cc.SOURCE_LABEL]
        self.assertEqual(len(chased), 1)
        self.assertIn(chased[0].id, corpus.included_ids, "an on-topic chased trial passes selection")
        self.assertEqual(report.included, 1)
        self.assertTrue(check(corpus)[0], "the corpus replays with the chased record and no network")
        # Seeds now carry reference data for the map
        self.assertTrue(any(p.references for p in corpus.records if p.id in corpus.included_ids))

    def test_prisma_flow_counts_other_methods_separately(self):
        with mock.patch.object(cc, "_post_batch", fake_batch()):
            r = self.client.post("/api/search", json={"query": golden.QUERY, "mode": "deep_research",
                                                      "review_framework": "systematic_prisma", "config": CONFIG})
        self.assertEqual(r.status_code, 200, r.text)
        flow = r.json()["review_protocol"]["flow"]
        corpus = corpus_store.load(r.json()["corpus_id"])
        other = flow["identified_other_methods"]
        self.assertEqual(other[cc.SOURCE_LABEL], 1)
        # Each excluded review cites a paper of its own in the fake endpoint: one per review, found by one review
        n_reviews = corpus.review_references.seeds_found
        self.assertGreater(n_reviews, 0)
        self.assertEqual(other[cc.REVIEW_REFS_LABEL], corpus.review_references.added)
        self.assertEqual(flow["identified_total"], corpus.records_found - sum(other.values()))
        self.assertNotIn(cc.SOURCE_LABEL, flow["identified_by_source"])


if __name__ == "__main__":
    unittest.main()
