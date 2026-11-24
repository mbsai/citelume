"""Session 8: claims are split and stratified correctly for the blinded annotation sheet."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import annotation_sheet as A  # noqa: E402


class TestClaims(unittest.TestCase):
    def test_sentences_and_citations(self):
        self.assertEqual(A.sentences("One [1]. Two (n=20) [2]! Three? [3] Four. [4, 5] Five."),
                         ["One [1].", "Two (n=20) [2]!", "Three [3]?", "Four [4, 5].", "Five."])
        self.assertEqual(A.cited("Effects [1-3] and [5, 7]."), [1, 2, 3, 5, 7])
        self.assertEqual(A.claims_of("Answer [1]. No citation here.", [{"content": "Section claim [2]."}]),
                         ["Answer [1].", "Section claim [2]."])

    def test_strata_weights_and_blinding(self):
        tmp = Path(tempfile.mkdtemp())
        res = tmp / "runs" / "Q1"
        res.mkdir(parents=True)
        papers = [{"citation_index": i, "title": f"Trial {i}", "year": 2020, "abstract": f"Trial {i} found an effect."}
                  for i in (1, 2, 3)]
        ungated = {"model_written": True, "executive_answer": "Kept [1]. Removed by the gate, 42% [2].",
                   "sections": [{"heading": "S", "content": "Shared [3]."}]}
        (res / "result.json").write_text(json.dumps({
            "query": "Q?", "executive_answer": "Kept [1]. Rewritten by the gate [2].",
            "sections": [{"heading": "S", "content": "Shared [3]."}], "papers": papers,
            "synthesis_evidence": {"ungated_narrative": ungated}}))
        A.ANN = tmp / "ann"
        A.build(type("a", (), {"results": [tmp / "runs"], "n": 10, "second": 2})())
        key = json.loads((A.ANN / "key.json").read_text())
        self.assertEqual(key["strata_sizes"], {"both": 2, "ungated_only": 1, "gated_only": 1})
        with (A.ANN / "sheet.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 4)
        self.assertNotIn("stratum", rows[0])  # the sheet does not reveal the narrative
        removed = next(u for u, k in key["units"].items() if k["stratum"] == "ungated_only")
        self.assertEqual((key["units"][removed]["in_gated"], key["units"][removed]["in_ungated"]), (False, True))
        self.assertTrue(all(k["weight"] == 1.0 for k in key["units"].values()))


if __name__ == "__main__":
    unittest.main()
