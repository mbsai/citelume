"""Evidence gap map (services/gap_map.py): query strings, the fixed gap rules, a full run against fake databases,
the demo's one-a-day allowance and the rule-based suggested axes. No network."""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import demo_mode, gap_map as G
from services.gap_map import GapMapCell, GapMapRequest, MapTerm


class TestQueries(unittest.TestCase):
    def test_clauses(self):
        t = MapTerm(name="tai chi", synonyms=["taiji", 'qi "gong"'])
        clause = asyncio.run(G.pubmed_clause(t, None, use_mesh=False))
        self.assertEqual(clause, '("tai chi"[Title/Abstract] OR "taiji"[Title/Abstract] OR "qi gong"[Title/Abstract])')
        self.assertEqual(G.openalex_clause(MapTerm(name="falls, accidental")), '("falls accidental")')
        self.assertEqual(G.with_design("A AND B", '"x"[pt]'), '(A AND B) AND "x"[pt]')
        self.assertEqual(G.combine("A", "", "B"), "A AND B")


def cell(all_=None, rct=None, sr=None, oa=None):
    c = GapMapCell(row=0, col=0, openalex_all=oa)
    c.counts = {"all": {"pubmed": all_}, "rct": {"pubmed": rct}, "sr": {"pubmed": sr}, "obs": {"pubmed": None}}
    return c


class TestFlags(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(G.cell_flags(cell(0, 0, 0, 0)), ["none"])
        self.assertEqual(G.cell_flags(cell(3, 1, 0, 2)), ["sparse"])
        self.assertEqual(G.cell_flags(cell(5, 0, 2, 24)), ["no_trials"])
        self.assertEqual(G.cell_flags(cell(300, 12, 0, 500)), ["unsynthesised"])
        self.assertEqual(G.cell_flags(cell(300, 12, 4, 500)), [])
        self.assertEqual(G.cell_flags(cell(None, None, None, None)), [])  # every query failed: no claim
        self.assertEqual(G.cell_flags(cell(0, 0, 0, 7)), ["no_trials"])  # OpenAlex finds some: not "none"
        self.assertEqual(G.cell_flags(cell(0, 0, 0, None)), ["none_pubmed"])  # OpenAlex unknown: not "any database"


class TestBuild(unittest.IsolatedAsyncioTestCase):
    async def test_full_run_with_fake_databases(self):
        async def fake_pubmed(client, query, key, pacer):
            if "fail" in query:
                return None
            if '"rare"' in query:
                return 0 if "randomized" in query else 2
            return 40 if "randomized" in query else (0 if "systematic" in query else 400)

        async def fake_openalex(client, expr, years, key, slots, state=None):
            return 1 if '"rare"' in expr else 900

        seen = []
        req = GapMapRequest(topic="t", population=MapTerm(name="older adults"),
                            rows=[MapTerm(name="common"), MapTerm(name="rare")], cols=[MapTerm(name="falls")],
                            use_mesh=False, year_from=2010)
        with mock.patch.object(G, "count_pubmed", fake_pubmed), mock.patch.object(G, "count_openalex", fake_openalex):
            m = await G.build_gap_map(req, progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(m.queries_run, 2 * 5 + 2 + 1)
        self.assertEqual(seen[-1], (m.queries_run, m.queries_run))
        self.assertEqual(m.queries_failed, 0)
        common, rare = m.cells
        self.assertIn('"older adults"[Title/Abstract]', common.pubmed_query)
        self.assertIn("2010:", common.pubmed_query)
        self.assertEqual(common.counts["all"]["pubmed"], 400)
        self.assertEqual(common.flags, ["unsynthesised"])
        self.assertEqual(rare.flags, ["sparse", "no_trials"])
        self.assertEqual([g.flag for g in m.gaps], ["sparse", "no_trials", "unsynthesised"])
        self.assertEqual(m.row_totals, [400, 2])

    async def test_axis_limit(self):
        req = GapMapRequest(rows=[MapTerm(name=f"r{i}") for i in range(6)], cols=[MapTerm(name="c")])
        with self.assertRaises(ValueError):
            await G.build_gap_map(req, max_axis=5)


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("x", request=None, response=None)


class TestRateLimits(unittest.IsolatedAsyncioTestCase):
    async def test_pubmed_retries_after_429(self):
        replies = [FakeResponse(429, {}), FakeResponse(200, {"esearchresult": {"count": "12"}})]

        class C:
            async def get(self, *a, **k):
                return replies.pop(0)

        with mock.patch.object(G.asyncio, "sleep", mock.AsyncMock()):
            self.assertEqual(await G.count_pubmed(C(), "q", "", G._Pacer(0)), 12)

    async def test_openalex_stops_when_the_keyless_budget_is_spent(self):
        calls = []

        class C:
            async def get(self, *a, **k):
                calls.append(1)
                return FakeResponse(429, {"error": "Insufficient budget"})

        state = G._OpenAlexState()
        slots = asyncio.Semaphore(4)
        self.assertIsNone(await G.count_openalex(C(), "x", "", "", slots, state))
        self.assertTrue(state.out_of_budget)
        self.assertIsNone(await G.count_openalex(C(), "y", "", "", slots, state))
        self.assertEqual(len(calls), 1)


class TestDemoAllowance(unittest.TestCase):
    def setUp(self):
        demo_mode.USAGE.reset()

    def tearDown(self):
        demo_mode.USAGE.reset()

    def test_one_a_day_whatever_the_key(self):
        self.assertEqual(demo_mode.bucket_for("gap_map", False), "gap_map")
        self.assertEqual(demo_mode.bucket_for("gap_map", True), "gap_map")
        with mock.patch.dict(os.environ, {"DEMO_GAP_MAP_PER_DAY": "1"}):
            demo_mode.take_quota("1.2.3.4", "gap_map")
            with self.assertRaises(demo_mode.DemoLimitError) as ctx:
                demo_mode.take_quota("1.2.3.4", "gap_map")
            self.assertIn("evidence gap map", str(ctx.exception))
            status = demo_mode.status_for("1.2.3.4")
            self.assertEqual((status["limits"]["gap_map"], status["remaining"]["gap_map"]), (1, 0))


class TestEndpoint(unittest.TestCase):
    def test_demo_charges_one_gap_map(self):
        from fastapi.testclient import TestClient
        import main

        async def fake_build(req, max_axis, progress=None):
            return G.GapMapResult(topic=req.topic, rows=req.rows, cols=req.cols)

        demo_mode.USAGE.reset()
        body = {"topic": "t", "rows": [{"name": "a"}], "cols": [{"name": "b"}]}
        with mock.patch.dict(os.environ, {"CITELUME_DEMO": "1", "DEMO_GAP_MAP_PER_DAY": "1"}), \
                mock.patch.object(G, "build_gap_map", fake_build):
            client = TestClient(main.app)
            self.assertEqual(client.post("/api/gap-map", json=body).status_code, 200)
            second = client.post("/api/gap-map", json=body)
            self.assertEqual(second.status_code, 429)
            too_big = dict(body, rows=[{"name": f"r{i}"} for i in range(6)])
            self.assertEqual(client.post("/api/gap-map", json=too_big).status_code, 400)
        demo_mode.USAGE.reset()


class TestSuggestion(unittest.IsolatedAsyncioTestCase):
    async def test_rules_read_the_question(self):
        s = await G.suggest_axes_from_rules("Does vitamin D supplementation reduce fractures in older adults?")
        self.assertEqual(s.source, "rules")
        self.assertTrue(any("vitamin d" in r.name.lower() for r in s.rows), s.rows)
        self.assertTrue(any("fracture" in c.name.lower() for c in s.cols), s.cols)
        self.assertIsNotNone(s.population)


if __name__ == "__main__":
    unittest.main()
