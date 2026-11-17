"""Session 3f: risk of bias from the full text only, every answer backed by a quote checked by code, RoB 2 domains
computed from signalling answers, and the reviewer's edits recorded."""
import asyncio
import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from frameworks import rob2
from models import ApiKeysConfig, PaperRecord
from services import corpus_store
from services.llm_service import set_test_llm_responder
from services.risk_of_bias_assessment import apply_edit, assess_risk_of_bias, quote_found
from tests import test_golden_search as golden
from tests.test_corpus_endpoints import CONFIG, fake_sources
from tests.test_imports import make_pdf
from tests.test_review_protocol import protocol_for
from tests.test_screening_by_criterion import screening_responder

REAL = "Older adults were randomly assigned to vitamin D supplementation or placebo and fractures were recorded"
INVENTED = "Allocation was concealed using sequentially numbered opaque sealed envelopes"
KEY = ApiKeysConfig(llm_provider="openrouter", llm_api_key="test-key")


def rob_responder(system_prompt, user_prompt, config, attempt):
    if system_prompt.startswith("You are assessing risk of bias in one randomised trial"):
        return json.dumps({"answers": [
            {"id": "1.1", "answer": "Y", "quote": REAL + "."},
            {"id": "1.2", "answer": "Y", "quote": INVENTED + "."},  # not in the text: must become NI
            {"id": "3.1", "answer": "Y", "quote": "  older adults were RANDOMLY assigned to vitamin D supplementation ... fractures were recorded"},
            {"id": "5.2", "answer": "N", "quote": "too short"},
        ], "main_concern": "Allocation concealment not reported"})
    if system_prompt.startswith("You are assessing risk of bias in one study"):
        return json.dumps({"domains": [{"judgement": "Low", "quote": REAL}, {"judgement": "High", "quote": INVENTED}],
                           "main_concern": "Selection unclear"})
    return screening_responder(system_prompt, user_prompt, config, attempt)


def paper(pid, types=(), study_type="rct", full_text=True):
    p = PaperRecord(id=pid, title=f"Trial {pid}", abstract="Abstract text.", publication_types=list(types),
                    study_type=study_type, citation_index=1)
    if full_text:
        p.full_text_source, p.full_text_sections = "test", {"methods": REAL + ". Outcomes were assessed yearly."}
    return p


class TestRob2Algorithm(unittest.TestCase):
    def test_domain_1(self):
        self.assertEqual(rob2.domain_1({"1.1": "Y", "1.2": "Y", "1.3": "N"}), "Low")
        self.assertEqual(rob2.domain_1({"1.1": "Y", "1.2": "Y", "1.3": "Y"}), "Some concerns")
        self.assertEqual(rob2.domain_1({"1.1": "N", "1.2": "Y", "1.3": "N"}), "Some concerns")
        self.assertEqual(rob2.domain_1({"1.2": "N"}), "High")
        self.assertEqual(rob2.domain_1({"1.2": "NI", "1.3": "PY"}), "High")
        self.assertEqual(rob2.domain_1({}), "Some concerns")

    def test_domain_2(self):
        blinded_itt = {"2.1": "N", "2.2": "N", "2.6": "Y"}
        self.assertEqual(rob2.domain_2(blinded_itt), "Low")
        self.assertEqual(rob2.domain_2({**blinded_itt, "2.6": "N", "2.7": "N"}), "Some concerns")
        self.assertEqual(rob2.domain_2({**blinded_itt, "2.6": "N", "2.7": "Y"}), "High")
        open_label = {"2.1": "Y", "2.2": "Y", "2.6": "Y"}
        self.assertEqual(rob2.domain_2({**open_label, "2.3": "N"}), "Low")
        self.assertEqual(rob2.domain_2({**open_label, "2.3": "NI"}), "Some concerns")
        self.assertEqual(rob2.domain_2({**open_label, "2.3": "Y", "2.4": "Y", "2.5": "N"}), "High")
        self.assertEqual(rob2.domain_2({**open_label, "2.3": "Y", "2.4": "Y", "2.5": "Y"}), "Some concerns")

    def test_domains_3_to_5_and_overall(self):
        self.assertEqual(rob2.domain_3({"3.1": "Y"}), "Low")
        self.assertEqual(rob2.domain_3({"3.1": "N", "3.2": "N", "3.3": "N"}), "Low")
        self.assertEqual(rob2.domain_3({"3.1": "N", "3.2": "N", "3.3": "Y", "3.4": "N"}), "Some concerns")
        self.assertEqual(rob2.domain_3({"3.1": "N", "3.2": "N", "3.3": "Y", "3.4": "Y"}), "High")
        self.assertEqual(rob2.domain_4({"4.1": "Y"}), "High")
        self.assertEqual(rob2.domain_4({"4.1": "N", "4.2": "N", "4.3": "N"}), "Low")
        self.assertEqual(rob2.domain_4({"4.1": "N", "4.2": "NI", "4.3": "N"}), "Some concerns")
        self.assertEqual(rob2.domain_4({"4.1": "N", "4.2": "N", "4.3": "Y", "4.4": "Y", "4.5": "N"}), "Some concerns")
        self.assertEqual(rob2.domain_5({"5.1": "Y", "5.2": "N", "5.3": "N"}), "Low")
        self.assertEqual(rob2.domain_5({"5.1": "NI", "5.2": "N", "5.3": "N"}), "Some concerns")
        self.assertEqual(rob2.domain_5({"5.2": "PY"}), "High")
        low = {"1.1": "Y", "1.2": "Y", "1.3": "N", "2.1": "N", "2.2": "N", "2.6": "Y", "3.1": "Y", "4.1": "N",
               "4.2": "N", "4.3": "N", "5.1": "Y", "5.2": "N", "5.3": "N"}
        self.assertEqual(rob2.judge(low)[1], "Low")
        self.assertEqual(rob2.judge({**low, "1.2": "N"})[1], "High")
        self.assertEqual(rob2.judge({**low, "5.1": "NI"})[1], "Some concerns")


class TestQuotes(unittest.TestCase):
    def test_quote_checking(self):
        src = "Older  adults were randomly\nassigned to vitamin D supplementa-\ntion. The eﬃcacy was 0·70."
        self.assertTrue(quote_found("older adults were randomly assigned to vitamin D", src))
        self.assertTrue(quote_found("randomly assigned to vitamin D supplementation", src))  # hyphen at line break
        self.assertTrue(quote_found("The efficacy was 0.70", src))
        self.assertTrue(quote_found("Older adults were randomly ... The efficacy was 0.70", src))
        self.assertFalse(quote_found("sealed opaque envelopes were used", src))
        self.assertFalse(quote_found("vitamin D", src))  # too short to show anything


class TestAssessment(unittest.TestCase):
    def setUp(self):
        set_test_llm_responder(rob_responder)

    def tearDown(self):
        set_test_llm_responder(None)

    def test_trials_answers_need_verified_quotes_and_domains_follow_the_algorithm(self):
        out = asyncio.run(assess_risk_of_bias([paper("t1", types=["Randomized Controlled Trial"], study_type="primary_other")], KEY))
        rob = out["t1"]
        self.assertTrue(rob.tool.startswith("RoB 2"))  # chosen from the publication type
        answers = {s.id: s for s in rob.signalling}
        self.assertEqual(answers["1.1"].answer, "Y")
        self.assertTrue(answers["1.1"].quote_verified)
        self.assertEqual(answers["1.2"].answer, "NI")  # its quote is not in the text
        self.assertFalse(answers["1.2"].quote_verified)
        self.assertEqual(answers["3.1"].answer, "Y")  # case, spacing and an ellipsis are tolerated
        self.assertEqual(answers["5.2"].answer, "NI")  # quote too short
        expected, overall = rob2.judge({s.id: s.answer for s in rob.signalling})
        self.assertEqual([d.judgement for d in rob.domains], [expected[k] for k in sorted(expected)])
        self.assertEqual(rob.overall, overall)
        self.assertEqual(rob.domains[0].judgement, "Some concerns")  # concealment not shown
        self.assertTrue(all(d.quote_verified for d in rob.domains if d.quote))
        self.assertTrue(rob.assessed_by and rob.prompt_version and rob.assessed_on)
        self.assertEqual(rob.status, "suggested")

    def test_other_designs_and_studies_without_full_text(self):
        out = asyncio.run(assess_risk_of_bias([
            paper("c1", study_type="controlled_observational"),
            paper("a1", full_text=False),
        ], KEY))
        c1 = out["c1"]
        self.assertEqual([d.judgement for d in c1.domains][:2], ["Low", "No information"])  # invented quote dropped
        self.assertEqual(out["a1"].overall, "Not assessed")
        self.assertIn("only the abstract", out["a1"].reason)
        none = asyncio.run(assess_risk_of_bias([paper("t2")], ApiKeysConfig(llm_provider="openrouter", llm_api_key="")))
        self.assertEqual(none["t2"].overall, "Not assessed")

    def test_edits_are_recorded_and_recompute_the_overall(self):
        rob = asyncio.run(assess_risk_of_bias([paper("t3", types=["Randomized Controlled Trial"])], KEY))["t3"]
        apply_edit(rob, "Randomisation process", "High", "Sequence generation by date of birth")
        self.assertEqual(rob.overall, "High")
        self.assertEqual(rob.status, "edited by the user")
        self.assertEqual((rob.edits[0].before, rob.edits[0].after), ("Some concerns", "High"))
        with self.assertRaises(ValueError):
            apply_edit(rob, "Not a domain", "Low", "x")


class TestInTheReview(unittest.TestCase):
    def setUp(self):
        golden.query_utils._DECOMPOSITION_CACHE.clear()
        golden.synthesis_engine._STUDY_EXTRACTION_CACHE.clear()
        set_test_llm_responder(rob_responder)
        self.client = TestClient(main.app)

    def tearDown(self):
        set_test_llm_responder(None)

    def test_systematic_review_assesses_full_texts_and_accepts_edits(self):
        pdf = self.client.post("/api/imports/pdf", json={
            "filename": "trial3.pdf",
            "content_base64": base64.b64encode(make_pdf(
                "Vitamin D supplementation and fractures in older adults: randomised trial 3", "10.1000/study3")).decode()}).json()
        keyed = {**CONFIG, "llm_api_key": "test-key"}
        with fake_sources():
            s = self.client.post("/api/corpus", json={"query": golden.QUERY, "mode": "deep_research", "config": keyed,
                                                      "protocol": protocol_for("deep_research", "systematic_prisma").model_dump(),
                                                      "user_pdfs": [{"pdf_id": pdf["pdf_id"]}]}).json()
            full = self.client.post(f"/api/corpus/{s['corpus_id']}/analyse", json={"config": keyed}).json()
        robs = {p["id"]: p["elicit"]["risk_of_bias"] for p in full["papers"] if p["elicit"].get("risk_of_bias")}
        trial3 = next(p["id"] for p in full["papers"] if p["doi"] == "10.1000/study3")
        self.assertNotEqual(robs[trial3]["overall"], "Not assessed")
        self.assertTrue(any(s["quote_verified"] for s in robs[trial3]["signalling"]))
        others = [r for pid, r in robs.items() if pid != trial3]
        self.assertTrue(others and all(r["overall"] == "Not assessed" for r in others))  # abstracts only
        rob_text = next(sec["content"] for sec in full["sections"] if sec["heading"].startswith("3.3"))
        self.assertIn("not assessed because only the abstract was available", rob_text)
        methods = next(sec["content"] for sec in full["sections"] if sec["heading"] == "2. Methods")
        self.assertIn("RoB 2 algorithm", methods)

        url = f"/api/corpus/{s['corpus_id']}/risk-of-bias"
        bad = self.client.post(url, json={"record_id": trial3, "domain": "Randomisation process", "judgement": "High",
                                          "reason": ""})
        self.assertEqual(bad.status_code, 400)
        r = self.client.post(url, json={"record_id": trial3, "domain": "Randomisation process", "judgement": "High",
                                        "reason": "Alternation by clinic day"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["overall"], "High")
        stored = corpus_store.load(s["corpus_id"]).analysis
        rob = next(p for p in stored.papers if p.id == trial3).elicit.risk_of_bias
        self.assertEqual(rob.edits[0].reason, "Alternation by clinic day")
        not_assessed = next(pid for pid, r in robs.items() if r["overall"] == "Not assessed")
        self.assertEqual(self.client.post(url, json={"record_id": not_assessed, "domain": "x", "judgement": "Low",
                                                     "reason": "y"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
