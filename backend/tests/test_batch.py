"""Session 4: the headless runner runs questions end to end from the command line, writes the corpus, result and run
record for each, resumes after an interruption, and never uses or writes the demo's keys."""
import asyncio
import csv
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from models import ApiKeysConfig
from services import batch
from services.corpus_store import read_file
from services.llm_service import set_test_llm_responder
from services.replay import check
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import fake_sources

SECRET = "sk-test-secret-0123456789"


def offline_config(**extra):
    return ApiKeysConfig(llm_provider="openrouter", llm_api_key="", enable_hf=False, enable_fulltext=False, **extra)


class TestBatchInputs(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return Path(f.name)

    def test_questions_file(self):
        qs = batch.read_questions(self.write(
            '# comment\n{"id": "a", "query": "vitamin D and fractures"}\n\n"sleep and memory in older adults"\n'))
        self.assertEqual([q["id"] for q in qs], ["a", "q004"])
        for bad in ('{"query": ""}', '{"id": "../x", "query": "q"}', '{"id": "a", "query": "x"}\n{"id": "a", "query": "y"}',
                    '{"query": "x", "mode": "fast"}', "not json"):
            with self.assertRaises(batch.BatchError):
                batch.read_questions(self.write(bad))

    def test_keys_from_environment_and_demo_refused(self):
        with mock.patch.dict(os.environ, {"CITELUME_LLM_API_KEY": SECRET, "CITELUME_LLM_MODEL": "m/x"}, clear=False):
            cfg = batch.load_config(None)
            self.assertEqual((cfg.llm_api_key, cfg.llm_model), (SECRET, "m/x"))
            self.assertNotIn(SECRET, json.dumps(batch.public_settings(cfg)))
            with mock.patch.dict(os.environ, {"DEMO_OPENROUTER_API_KEY": SECRET}):
                with self.assertRaises(batch.BatchError):
                    batch.load_config(None)
            with mock.patch.dict(os.environ, {"CITELUME_DEMO": "1"}):
                with self.assertRaises(batch.BatchError):
                    batch.load_config(None)


class TestBatchRun(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        golden.synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
        golden.clear_gap_verification_cache()
        set_test_llm_responder(None)
        self.sources = fake_sources()
        self.sources.__enter__()
        self.addCleanup(self.sources.__exit__, None, None, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.questions = [{"id": "vitd", "query": golden.QUERY},
                          {"id": "vitd-prisma", "query": golden.QUERY, "mode": "deep_research",
                           "review_type": "systematic_prisma"}]
        self.defaults = {"mode": "standard", "review_type": "narrative"}

    def run_batch(self, **kw):
        return asyncio.run(batch.run_batch(self.questions, self.out, self.defaults, offline_config(), pause_s=0,
                                           log=lambda *_: None, **kw))

    def test_end_to_end_then_resume(self):
        self.assertEqual(self.run_batch(), (2, 0, 0))
        for qid in ("vitd", "vitd-prisma"):
            folder = self.out / qid
            run = json.loads((folder / "run.json").read_text())
            self.assertEqual(run["status"], "done")
            self.assertEqual(run["model"], "none (rule-based)")
            self.assertIn("corpus", run["timings_s"])
            self.assertIn("analysis", run["timings_s"])
            self.assertEqual(run["versions"]["runner"], batch.RUNNER_VERSION)
            result = json.loads((folder / "result.json").read_text())
            corpus = read_file(folder / "corpus.json.gz")
            self.assertEqual(result["corpus_id"], corpus.corpus_id)
            self.assertIsNotNone(corpus.analysis)
            self.assertEqual(len(corpus.included_ids), run["included"])
            self.assertTrue(check(corpus)[0], "the saved corpus replays")
        self.assertEqual(json.loads((self.out / "vitd-prisma" / "run.json").read_text())["review_type"],
                         "systematic_prisma")
        with (self.out / "summary.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["status"] for r in rows], ["done", "done"])
        # A second run skips what is done; --rerun runs it again
        self.assertEqual(self.run_batch(), (0, 0, 2))
        self.assertEqual(self.run_batch(rerun=True), (2, 0, 0))

    def test_failure_is_recorded_and_retried(self):
        with mock.patch.object(main, "_analyse", side_effect=RuntimeError("model down")):
            self.assertEqual(self.run_batch(), (0, 2, 0))
        run = json.loads((self.out / "vitd" / "run.json").read_text())
        self.assertEqual(run["status"], "failed")
        self.assertIn("model down", run["error"])
        self.assertTrue((self.out / "vitd" / "corpus.json.gz").exists(), "the corpus is kept when the analysis fails")
        self.assertEqual(self.run_batch(skip_failed=True), (0, 0, 2))
        self.assertEqual(self.run_batch(), (2, 0, 0))

    def test_unsearchable_question_fails_cleanly(self):
        self.questions = [{"id": "empty", "query": "state of the art"}]
        self.assertEqual(self.run_batch(), (0, 1, 0))
        self.assertIn("research topic", json.loads((self.out / "empty" / "run.json").read_text())["error"])

    def test_keys_never_written(self):
        cfg = offline_config(pubmed_api_key=SECRET)
        asyncio.run(batch.run_batch(self.questions[:1], self.out, self.defaults, cfg, pause_s=0, log=lambda *_: None))
        for path in (self.out / "vitd").iterdir():
            raw = path.read_bytes()
            if path.suffix == ".gz":
                raw = gzip.decompress(raw)
            self.assertNotIn(SECRET.encode(), raw, path.name)
        self.assertEqual(json.loads((self.out / "vitd" / "run.json").read_text())["settings"]["literature_keys"],
                         ["pubmed_api_key"])


if __name__ == "__main__":
    unittest.main()
