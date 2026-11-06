import asyncio
import sys
import unittest
from pathlib import Path

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx

from services import europepmc_client, pubmed_client, s2_client, source_health
from services.europepmc_client import translate_pubmed_query
from services.source_health import RetryingTransport, begin_source_health, set_openalex_api_key, source_failure_warnings

EPMC_HIT = {
    "hitCount": 1,
    "resultList": {"result": [{
        "pmid": "26769341", "title": "Prevalence of coronary artery disease in Kerala.", "pubYear": "2016",
        "authorString": "Krishnan MN, Zachariah G, Venugopal K.", "doi": "10.1186/S12872-016-0189-3",
        "abstractText": "<h4>Background</h4>CAD prevalence in Kerala was 12.5% in men.", "citedByCount": 120,
        "journalInfo": {"journal": {"title": "BMC Cardiovascular Disorders"}},
    }]},
}


def _transport(handler):
    return lambda: RetryingTransport(httpx.MockTransport(handler))


class TestPubMedQueryTranslation(unittest.TestCase):
    def test_fields_and_filters(self):
        q = translate_pubmed_query('"coronary artery disease"[Title/Abstract] AND india[Title/Abstract] AND '
                                   '"Review"[Publication Type] AND (2015:2024[dp])')
        self.assertIn('(TITLE:"coronary artery disease" OR ABSTRACT:"coronary artery disease")', q)
        self.assertIn("(TITLE:india OR ABSTRACT:india)", q)
        self.assertIn('PUB_TYPE:"review"', q)
        self.assertIn("PUB_YEAR:[2015 TO 2024]", q)
        self.assertTrue(q.endswith("AND SRC:MED"))
        self.assertNotIn("[Title/Abstract]", q)

    def test_untagged_words_before_a_filter_are_not_swallowed(self):
        q = translate_pubmed_query('branched chain amino acids[Title/Abstract] AND cardiovascular AND review[pt]')
        self.assertIn('PUB_TYPE:"review"', q)
        self.assertNotIn('PUB_TYPE:"branched', q)
        self.assertNotIn("title:", q)  # field names must stay upper case


class TestPubMedFallback(unittest.TestCase):
    def setUp(self):
        source_health.BACKOFF_S = (0.0, 0.0)
        self._orig = (pubmed_client.retrying_transport, europepmc_client.retrying_transport)

    def tearDown(self):
        pubmed_client.retrying_transport, europepmc_client.retrying_transport = self._orig

    def test_ncbi_outage_uses_europe_pmc_mirror(self):
        pubmed_client.retrying_transport = _transport(lambda req: httpx.Response(500))
        europepmc_client.retrying_transport = _transport(lambda req: httpx.Response(200, json=EPMC_HIT))

        async def run():
            begin_source_health()
            papers = await pubmed_client.search_pubmed('"coronary artery disease"[Title/Abstract]', 5)
            return papers, source_failure_warnings()

        papers, warnings = asyncio.run(run())
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual((p.pmid, p.source, p.year), ("26769341", "PubMed", 2016))
        self.assertEqual(p.abstract, "Background CAD prevalence in Kerala was 12.5% in men.")
        self.assertIn("Europe PMC mirror", warnings[0])

    def test_both_down_is_reported_without_fallback(self):
        pubmed_client.retrying_transport = _transport(lambda req: httpx.Response(500))
        europepmc_client.retrying_transport = _transport(
            lambda req: httpx.Response(200, json={"errCode": 503, "errMsg": "temporarily unavailable"}))

        async def run():
            begin_source_health()
            papers = await pubmed_client.search_pubmed("fibrosis[Title/Abstract]", 5)
            return papers, source_failure_warnings()

        papers, warnings = asyncio.run(run())
        self.assertEqual(papers, [])
        self.assertIn("results come from the other sources only", warnings[0])


class TestOpenAlexKey(unittest.TestCase):
    def test_key_is_sent_as_api_key_parameter(self):
        seen = {}
        orig = s2_client.retrying_transport

        def handler(req):
            seen["api_key"] = req.url.params.get("api_key")
            return httpx.Response(200, json={"results": []})

        s2_client.retrying_transport = _transport(handler)
        try:
            async def run():
                set_openalex_api_key("test-openalex-key")
                await s2_client._fallback_openalex_search("systemic sclerosis", 3)
            asyncio.run(run())
        finally:
            s2_client.retrying_transport = orig
        self.assertEqual(seen["api_key"], "test-openalex-key")


if __name__ == "__main__":
    unittest.main()
