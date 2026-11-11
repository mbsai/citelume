"""Streaming calls: stall detection, backup race and keeping the complete part of a cut-off reply."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import llm_service
from services.llm_streaming import StreamLimits, salvage_partial_json, stream_ollama_chat, stream_openai_chat


def sse(text=None, finish=None, model="m-primary"):
    delta = {"content": text} if text is not None else {}
    return ("data: " + json.dumps({"model": model, "choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


def scripted(script, status=200, headers=None):
    """Transport whose reply body follows `script`: bytes are sent, numbers are pauses in seconds."""
    async def body():
        for step in script:
            if isinstance(step, (int, float)):
                await asyncio.sleep(step)
            else:
                yield step

    async def handler(request):
        return httpx.Response(status, headers=headers or {}, content=body())
    return httpx.MockTransport(handler)


async def run_stream(transport, limits, fn=stream_openai_chat, **kw):
    async with httpx.AsyncClient(transport=transport) as client:
        return await fn(client, "http://llm.test/chat", {}, {"model": "m-primary"}, limits, **kw)


FAST = StreamLimits(first_token_s=0.3, idle_s=0.3, deadline_s=5)


class TestStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_normal_reply_is_assembled(self):
        started = asyncio.Event()
        res = await run_stream(scripted([sse("Hello "), 0.05, sse("world"), sse(finish="stop"), b"data: [DONE]\n\n"]),
                               FAST, first_token=started)
        self.assertTrue(res.ok)
        self.assertEqual((res.text, res.finish_reason, res.model), ("Hello world", "stop", "m-primary"))
        self.assertTrue(started.is_set())

    async def test_slow_but_working_model_is_not_cut_off(self):
        # Total time is well past the idle limit, but tokens keep arriving
        script = []
        for i in range(8):
            script += [0.15, sse(f"t{i} ")]
        res = await run_stream(scripted(script + [sse(finish="stop")]), FAST)
        self.assertTrue(res.ok)
        self.assertEqual(len(res.text.split()), 8)
        self.assertGreater(res.elapsed_s, FAST.idle_s * 2)

    async def test_queued_request_is_abandoned_early(self):
        # Keep-alive comments but no answer: not mistaken for progress
        res = await run_stream(scripted([b": PROCESSING\n\n", 0.2, b": PROCESSING\n\n", 5, sse("late")]), FAST)
        self.assertFalse(res.ok)
        self.assertIn("no answer started", res.stalled)
        self.assertLess(res.elapsed_s, 1.0)

    async def test_reply_that_stops_midway_keeps_what_arrived(self):
        res = await run_stream(scripted([sse('{"items": [{"a": 1}, '), sse('{"a": 2}, {"a"'), 5, sse(" never")]), FAST)
        self.assertIn("no output for", res.stalled)
        self.assertEqual(json.loads(salvage_partial_json(res.text)), {"items": [{"a": 1}, {"a": 2}]})

    async def test_overall_deadline(self):
        script = []
        for i in range(50):
            script += [0.1, sse("x ")]
        res = await run_stream(scripted(script), StreamLimits(first_token_s=1, idle_s=1, deadline_s=0.5))
        self.assertIn("time budget", res.stalled)
        self.assertLess(res.elapsed_s, 1.0)

    async def test_http_error_and_stream_error_are_reported_not_raised(self):
        res = await run_stream(scripted([json.dumps({"error": {"message": "Rate limited"}}).encode()], status=429,
                                        headers={"retry-after": "3"}), FAST)
        self.assertEqual((res.status, res.error, res.retry_after), (429, "Rate limited", 3.0))
        res = await run_stream(scripted([b'data: {"error": {"message": "upstream died"}}\n\n']), FAST)
        self.assertEqual(res.error, "upstream died")

    async def test_ollama_stream(self):
        lines = [json.dumps({"message": {"content": "Hi "}, "done": False}).encode() + b"\n", 0.05,
                 json.dumps({"message": {"content": "there"}, "done": False}).encode() + b"\n",
                 json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop", "model": "qwen"}).encode() + b"\n"]
        res = await run_stream(scripted(lines), FAST, fn=stream_ollama_chat)
        self.assertEqual((res.text, res.finish_reason, res.model), ("Hi there", "stop", "qwen"))


class TestSalvage(unittest.TestCase):
    def test_complete_list_items_are_kept(self):
        cut = '{"study_extractions": [{"citation_index": 1, "key_findings": "A, [b] {c}"}, {"citation_index": 2}, {"citation_ind'
        self.assertEqual([x["citation_index"] for x in json.loads(salvage_partial_json(cut))["study_extractions"]], [1, 2])

    def test_complete_sentences_of_a_long_text_are_kept(self):
        body = "First finding is clear [1]. " * 12
        cut = '{"content": "' + body + "Second finding is incompl"
        self.assertTrue(json.loads(salvage_partial_json(cut))["content"].endswith("clear [1]."))

    def test_nothing_usable(self):
        self.assertIsNone(salvage_partial_json('{"study_extractions": [{"citation_index": 1, "key'))
        self.assertIsNone(salvage_partial_json("no json here"))
        self.assertEqual(salvage_partial_json('noise {"a": 1} trailing'), '{"a": 1}')


class TestBackupRace(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._hedge_after = llm_service.HEDGE_AFTER_S
        llm_service.HEDGE_AFTER_S = 0.1
        llm_service.begin_llm_request_health()

    def tearDown(self):
        llm_service.HEDGE_AFTER_S = self._hedge_after

    def _transport(self, primary_script, backup_script, calls):
        async def handler(request):
            model = json.loads(request.content)["model"]
            calls.append(model)
            script = primary_script if model == "m-primary" else backup_script

            async def body():
                for step in script:
                    if isinstance(step, (int, float)):
                        await asyncio.sleep(step)
                    else:
                        yield step
            return httpx.Response(200, content=body())
        return httpx.MockTransport(handler)

    async def _race(self, primary_script, backup_script, with_backup=True):
        calls = []
        limits = StreamLimits(first_token_s=1.0, idle_s=1.0, deadline_s=5)
        async with httpx.AsyncClient(transport=self._transport(primary_script, backup_script, calls)) as client:
            res = await llm_service._stream_with_backup(
                client, "http://llm.test/chat", {}, {"model": "m-primary"},
                {"model": "m-backup"} if with_backup else None, limits)
        return res, calls

    async def test_prompt_primary_never_starts_a_backup(self):
        res, calls = await self._race([sse("fast"), sse(finish="stop")], [sse("backup", model="m-backup")])
        self.assertEqual((res.text, calls), ("fast", ["m-primary"]))
        self.assertEqual(llm_service.llm_request_stats()["hedges"], 0)

    async def test_backup_wins_when_primary_has_not_started(self):
        res, calls = await self._race([5, sse("too late")], [sse("from backup", model="m-backup"), sse(finish="stop", model="m-backup")])
        self.assertEqual((res.text, res.model), ("from backup", "m-backup"))
        self.assertEqual(calls, ["m-primary", "m-backup"])
        self.assertLess(res.elapsed_s, 1.0)
        self.assertEqual(llm_service.llm_request_stats()["hedges"], 1)

    async def test_primary_still_wins_if_it_finishes_first_after_the_backup_starts(self):
        res, _ = await self._race([0.2, sse("primary"), sse(finish="stop")], [3, sse("backup", model="m-backup")])
        self.assertEqual(res.text, "primary")

    async def test_backup_races_are_limited_per_search(self):
        llm_service._REQUEST_LLM_HEALTH.get()["hedges"] = llm_service.MAX_HEDGES_PER_SEARCH
        res, calls = await self._race([0.3, sse("primary"), sse(finish="stop")], [sse("backup", model="m-backup")])
        self.assertEqual((res.text, calls), ("primary", ["m-primary"]))


class TestStreamInterpretation(unittest.TestCase):
    def setUp(self):
        llm_service.begin_llm_request_health()
        llm_service._LAST_LLM_PARTIAL.set(False)
        llm_service._LAST_LLM_TRUNCATED.set(False)

    def test_stalled_reply_with_complete_items_is_used(self):
        from services.llm_streaming import ChatStreamResult
        res = ChatStreamResult(text='{"study_extractions": [{"citation_index": 1}, {"citation_ind', stalled="no output for 25s")
        text = llm_service._text_from_stream(res, "openrouter", "m", expect_json=True)
        self.assertEqual(json.loads(text), {"study_extractions": [{"citation_index": 1}]})
        self.assertTrue(llm_service._LAST_LLM_PARTIAL.get())
        self.assertEqual(llm_service.llm_request_stats()["partials"], 1)

    def test_stall_with_nothing_usable_fails_with_a_reason(self):
        from services.llm_streaming import ChatStreamResult
        res = ChatStreamResult(text="", stalled="no answer started within 45s")
        self.assertIsNone(llm_service._text_from_stream(res, "openrouter", "m", expect_json=True))
        self.assertIn("no answer started", llm_service._LAST_LLM_ERROR.get())
        self.assertEqual(llm_service.llm_request_stats()["stalls"], 1)

    def test_prose_cut_at_the_token_limit_keeps_its_complete_sentences(self):
        from services.llm_streaming import ChatStreamResult
        res = ChatStreamResult(text='{"content": "' + "A finding is reported [1]. " * 12 + "An unfinished sent", finish_reason="length")
        text = llm_service._text_from_stream(res, "openrouter", "m", expect_json=True)
        self.assertTrue(json.loads(text)["content"].endswith("reported [1]."))
        self.assertFalse(llm_service._LAST_LLM_TRUNCATED.get())

    def test_token_limit_is_reported_so_the_job_can_be_split(self):
        from services.llm_streaming import ChatStreamResult
        res = ChatStreamResult(text='{"study_extractions": [{"citation_index": 1}, {"cit', finish_reason="length")
        self.assertIsNone(llm_service._text_from_stream(res, "openrouter", "m", expect_json=True))
        self.assertTrue(llm_service._LAST_LLM_TRUNCATED.get())


class TestRepairAndBudget(unittest.IsolatedAsyncioTestCase):
    """The step runner corrects a specific fault in place, and stops retrying when its time budget is spent."""

    def setUp(self):
        from models import ApiKeysConfig
        self.cfg = ApiKeysConfig(llm_provider="openrouter", llm_model="test/model", llm_api_key="k")
        self.calls = 0

    def tearDown(self):
        llm_service.set_test_llm_responder(None)

    async def _run(self, reply, **kw):
        from models import TaskJSectionOutput

        def responder(sys_p, usr_p, cfg, attempt):
            self.calls += 1
            return reply
        llm_service.set_test_llm_responder(responder)
        validate = lambda out: ["Contains the number '99.9'"] if "99.9" in out.content else []
        return await llm_service.execute_validated_llm_task(
            task_id="Task_J", task_name="section", system_prompt="s", user_prompt="u",
            response_model=TaskJSectionOutput, config=self.cfg, custom_validator=validate, max_retries=2, **kw)

    async def test_one_bad_figure_is_removed_without_a_second_call(self):
        from services.verification_gates import drop_unsupported_numbers
        repair = lambda out, errs: out.model_copy(update={"content": drop_unsupported_numbers(out.content, "AUROC 0.89", by_clause=False)})
        out, rec = await self._run(json.dumps({"content": "AUROC was 0.89 [1]. Accuracy was 99.9% [2]. Drift occurred [2]."}), repair=repair)
        self.assertEqual(out.content, "AUROC was 0.89 [1]. Drift occurred [2].")
        self.assertEqual((self.calls, rec.attempts_used, rec.degraded_to_deterministic), (1, 1, False))
        self.assertTrue(any(e.startswith(llm_service.REPAIRED_NOTE) for e in rec.validation_errors))

    async def test_without_repair_the_model_is_asked_again(self):
        out, rec = await self._run(json.dumps({"content": "Accuracy was 99.9% [2]."}))
        self.assertIsNone(out)
        self.assertEqual(self.calls, 3)

    async def test_spent_time_budget_stops_further_attempts(self):
        out, rec = await self._run(json.dumps({"content": "Accuracy was 99.9% [2]."}), deadline_s=0.0)
        self.assertIsNone(out)
        self.assertEqual(self.calls, 1)
        self.assertTrue(rec.degraded_to_deterministic)
        self.assertIn("time budget", rec.validation_errors[-1])

    def test_gap_grid_tolerates_blank_fields_for_cells_that_are_not_gaps(self):
        from models import TaskHIGapOutput
        out = TaskHIGapOutput.model_validate({"rows": ["a", "b", "c"], "cols": ["x", "y", "z"], "assignments": [
            {"row_idx": 0, "col_idx": 0, "paper_indices": [1, 2], "impact_level": "", "why_gap_exists": None},
            {"row_idx": 0, "col_idx": 1, "paper_indices": None, "impact_level": "Low"},
            {"row_idx": 1, "col_idx": 1, "paper_indices": [3], "impact_level": "high"}]})
        self.assertEqual([a.impact_level for a in out.assignments], ["Medium", "Medium", "High"])
        self.assertEqual(out.assignments[1].paper_indices, [])

    def test_citation_and_number_repairs(self):
        from services.verification_gates import drop_unsupported_numbers, strip_citations_not_in
        self.assertEqual(strip_citations_not_in("Works well [1, 99]. Also [99]. And [2-3].", {1, 2, 3}), "Works well [1]. Also. And [2-3].")
        self.assertEqual(drop_unsupported_numbers("OR 1.24; prevalence 55.5%; n = 40", "OR 1.24 in n = 40"), "OR 1.24; n = 40")


if __name__ == "__main__":
    unittest.main()
