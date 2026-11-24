"""Logs never carry what users searched for (a gap found in the demo's logs on 24 Nov 2026), and a bioRxiv result
without a DOI no longer stops the bioRxiv search."""
import asyncio
import io
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from services import biorxiv_client


class TestRedaction(unittest.TestCase):
    def test_urls_lose_their_query_string(self):
        line = ("[PubMed Connector Warning] Server error '500' for url "
                "'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=neuroendocrine+prostate'")
        out = main.redact_query_strings(line)
        self.assertNotIn("neuroendocrine", out)
        self.assertIn("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?[query removed]", out)
        self.assertEqual(main.redact_query_strings("[search-stage] standard stage 2/6"), "[search-stage] standard stage 2/6")

    def test_stdout_is_redacted(self):
        buf = io.StringIO()
        stream = main._RedactingStream(buf)
        print("failed: https://export.arxiv.org/api/query?search_query=all:%22spatial+metabolomics%22", file=stream)
        self.assertNotIn("metabolomics", buf.getvalue())
        self.assertIsInstance(sys.stdout, main._RedactingStream)

    def test_access_log_keeps_path_not_parameters(self):
        record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                                   ("1.2.3.4:5", "GET", "/api/hf-models?search=vitamin+d", "1.1", 200), None)
        main._AccessLogFilter().filter(record)
        self.assertEqual(record.args[2], "/api/hf-models?[query removed]")
        self.assertNotIn("vitamin", record.getMessage())


class FakeResponse:
    status_code = 200

    def json(self):
        return {"resultList": {"result": [
            {"id": "PPR123", "title": "A preprint without a DOI", "abstractText": "Text.", "pubYear": "2024"},
            {"id": "PPR124", "doi": "10.1101/2024.01.01.000001", "title": "A preprint with a DOI",
             "abstractText": "Text.", "pubYear": "2024"}]}}


class FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, *a, **k):
        return FakeResponse()


class TestBiorxivWithoutDoi(unittest.TestCase):
    def test_no_doi_does_not_stop_the_search(self):
        with mock.patch.object(biorxiv_client.httpx, "AsyncClient", FakeClient):
            papers = asyncio.run(biorxiv_client.search_biorxiv("preprint", max_results=5))
        self.assertEqual(len(papers), 2)
        no_doi = next(p for p in papers if not p.doi)
        self.assertEqual(no_doi.url, "https://europepmc.org/article/PPR/PPR123")


if __name__ == "__main__":
    unittest.main()
