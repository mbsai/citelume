import asyncio
import sys
import unittest
from pathlib import Path

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx

from services import source_health
from services.source_health import (
    RetryingTransport,
    begin_source_health,
    report_source_failure,
    source_failure_warnings,
)


class TestSourceHealth(unittest.TestCase):
    def setUp(self):
        source_health.BACKOFF_S = (0.0, 0.0)  # no real waiting in tests

    def _client(self, statuses):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(statuses[min(len(calls) - 1, len(statuses) - 1)], headers={"retry-after": "0"})

        return httpx.AsyncClient(transport=RetryingTransport(httpx.MockTransport(handler))), calls

    def test_transient_errors_are_retried_until_success(self):
        async def run():
            client, calls = self._client([500, 429, 200])
            async with client:
                r = await client.get("https://example.org")
            return r.status_code, len(calls)
        self.assertEqual(asyncio.run(run()), (200, 3))

    def test_gives_up_after_max_retries_and_does_not_retry_client_errors(self):
        async def run(statuses):
            client, calls = self._client(statuses)
            async with client:
                r = await client.get("https://example.org")
            return r.status_code, len(calls)
        self.assertEqual(asyncio.run(run([503])), (503, 3))
        self.assertEqual(asyncio.run(run([404])), (404, 1))

    def test_failures_become_readable_warnings(self):
        async def run():
            begin_source_health()
            req = httpx.Request("GET", "https://eutils.ncbi.nlm.nih.gov")
            report_source_failure("PubMed", httpx.HTTPStatusError("x", request=req, response=httpx.Response(500, request=req)))
            report_source_failure("Semantic Scholar / OpenAlex",
                                  httpx.HTTPStatusError("x", request=req, response=httpx.Response(429, request=req)))
            return source_failure_warnings()
        warnings = asyncio.run(run())
        self.assertIn("Source unavailable: PubMed HTTP 500", warnings[0])
        self.assertIn("rate-limited", warnings[1])


if __name__ == "__main__":
    unittest.main()
