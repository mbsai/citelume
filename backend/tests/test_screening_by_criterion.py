"""Session 3e: the model answers each screening criterion; a fixed rule decides; answers are stored and replayed;
the user can overturn decisions with a reason."""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from services import corpus_store
from services.ai_screening import ScreeningDecision, criterion_reason, decide, screening_criteria
from services.llm_service import set_test_llm_responder
from services.replay import check
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import CONFIG, fake_sources
from tests.test_review_protocol import protocol_for

KEYED = {**CONFIG, "llm_api_key": "test-key"}


def screening_responder(system_prompt, user_prompt, _config, _attempt):
    """Trials meet every criterion; cohort studies fail the outcome criterion; anything else is unclear."""
    if not system_prompt.startswith("You are screening records"):
        return None  # every other step falls back to rule-based text
    criteria = [ln for ln in user_prompt.split("Records:")[0].split("Criteria:")[1].splitlines() if ln[:1].isdigit()]
    records = [ln for ln in user_prompt.split("Records:")[1].splitlines() if re.match(r"^\d+\. ", ln)]
    out = []
    for i, line in enumerate(records, 1):
        if "trial" in line.lower():
            answers = ["yes"] * len(criteria)
        elif "cohort" in line.lower():
            answers = ["no" if c.split(". ", 1)[1].startswith("Outcome") else "yes" for c in criteria]
        else:
            answers = ["unclear"] * len(criteria)
        out.append({"n": i, "answers": answers, "reason": "outcome not studied" if "no" in answers else "meets"})
    return json.dumps({"decisions": out})


class TestRules(unittest.TestCase):
    def test_criteria_come_from_the_protocol(self):
        p = protocol_for("deep_research", "systematic_prisma")
        labels = [label for label, _ in screening_criteria(p, "systematic_prisma")]
        self.assertEqual(labels, ["Intervention: vitamin D supplementation", "Outcome: fractures",
                                  "Population: older adults", "Design: primary study"])
        p.eligibility.study_designs = ["rct"]
        self.assertEqual(screening_criteria(p, "systematic_prisma")[-1][0], "Design: randomised controlled trial")
        self.assertNotIn("Design: primary study", [l for l, _ in screening_criteria(protocol_for(), "scoping")])

    def test_fixed_decision_rule(self):
        labels = ["A", "B", "C"]
        self.assertEqual(decide(["yes", "unclear", "unclear"], labels), (True, ""))  # unclear never excludes
        self.assertEqual(decide(["yes", "no", "no"], labels), (False, "B"))  # the first failure is named
        # When several fail, design is named before population, intervention and outcome
        named = ["Intervention: x", "Outcome: y", "Population: z", "Design: primary study"]
        self.assertEqual(decide(["no", "no", "yes", "no"], named), (False, "Design: primary study"))
        self.assertEqual(decide(["no", "no", "yes", "yes"], named), (False, "Intervention: x"))
        d = ScreeningDecision(n=1, answers=["Yes", "No.", "not sure", {"answer": "unclear"}])
        self.assertEqual(d.answers, ["yes", "no", "unclear", "unclear"])


class TestScreeningInTheCorpus(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        golden.synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
        set_test_llm_responder(screening_responder)
        self.client = TestClient(main.app)

    def tearDown(self):
        set_test_llm_responder(None)

    def build(self):
        p = protocol_for("deep_research", "systematic_prisma").model_dump()
        with fake_sources():
            r = self.client.post("/api/corpus", json={"query": golden.QUERY, "mode": "deep_research", "protocol": p,
                                                      "config": KEYED})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_each_exclusion_names_its_criterion_and_replays(self):
        s = self.build()
        rep = s["selection_report"]
        reason = criterion_reason("Outcome: fractures")
        cohorts = [d for d in rep["screening_decisions"] if "cohort" in d["title"]]
        self.assertTrue(cohorts)
        self.assertEqual(rep["excluded"][reason], len(cohorts))
        self.assertTrue(all(d["failed_criterion"] == "Outcome: fractures" and not d["include"] for d in cohorts))
        self.assertTrue(all(len(d["answers"]) == 4 for d in rep["screening_decisions"] if d["screened"]))
        self.assertFalse(any("cohort" in p["title"] for p in s["included"]))
        self.assertEqual(rep["screening_criteria"][1], "Outcome: fractures")
        self.assertTrue(rep["screening_prompt_version"] and rep["screening_date"] and rep["screening_model"])
        # Replay uses the stored answers: no model is called
        set_test_llm_responder(lambda *a: (_ for _ in ()).throw(AssertionError("model called during replay")))
        self.assertTrue(check(corpus_store.load(s["corpus_id"]))[0])

    def test_the_user_can_overturn_a_decision_with_a_reason(self):
        s = self.build()
        cohort = next(d for d in s["selection_report"]["screening_decisions"] if "cohort" in d["title"])
        url = f"/api/corpus/{s['corpus_id']}/screening-overrides"
        self.assertEqual(self.client.post(url, json={"overrides": [{"record_id": cohort["record_id"], "include": True,
                                                                     "reason": " "}]}).status_code, 400)
        self.assertEqual(self.client.post(url, json={"overrides": [{"record_id": "nope", "include": True,
                                                                     "reason": "x"}]}).status_code, 400)
        r = self.client.post(url, json={"overrides": [{"record_id": cohort["record_id"], "include": True,
                                                       "reason": "Reports fractures in a secondary table"}]})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        added = next(p for p in d["included"] if p["id"] == cohort["record_id"])
        self.assertIn("Reports fractures in a secondary table", added["selection_reason"])
        reason = criterion_reason("Outcome: fractures")
        self.assertEqual(d["selection_report"]["excluded"][reason], s["selection_report"]["excluded"][reason] - 1)
        derived = corpus_store.load(d["corpus_id"])
        self.assertTrue(any("changed 1 decision" in n for n in derived.protocol.notes))
        set_test_llm_responder(None)
        self.assertTrue(check(derived)[0])
        # The flow diagram counts the change
        with fake_sources():
            full = self.client.post(f"/api/corpus/{d['corpus_id']}/analyse", json={"config": CONFIG}).json()
        self.assertEqual(full["review_protocol"]["flow"]["excluded_at_eligibility"][reason],
                         s["selection_report"]["excluded"][reason] - 1)
        # And the user can exclude a study the model kept
        trial = next(p for p in s["included"] if "trial" in p["title"])
        r = self.client.post(url, json={"overrides": [{"record_id": trial["id"], "include": False,
                                                       "reason": "Duplicate report of another trial"}]}).json()
        self.assertNotIn(trial["id"], [p["id"] for p in r["included"]])
        self.assertEqual(r["selection_report"]["excluded"]["Excluded by the user after reviewing the model's screening"], 1)


if __name__ == "__main__":
    unittest.main()
