"""The search and selection code measured in Session 6 is frozen (eval/PROTOCOL.md, amendment of 23 Nov 2026; git
tag eval-freeze-2.5). This test fails if any of it changes, so a change cannot slip in unnoticed. If a change is
intended, it must get a new selection protocol version and be reported as post-freeze, separately from the recall
results; then record the new hashes in eval/FREEZE.json with a note, in the same commit."""
import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

FREEZE = ROOT / "eval" / "FREEZE.json"
FROZEN_FILES = [
    "services/paper_selection.py", "services/query_planning.py", "services/question_parsing.py",
    "services/query_utils.py", "services/mesh_terms.py", "services/citation_chasing.py", "services/ai_screening.py",
    "services/corpus.py", "services/review_protocol.py", "services/study_limits.py", "services/pubmed_client.py",
    "services/s2_client.py", "services/europepmc_client.py", "services/arxiv_client.py",
    "services/biorxiv_client.py", "services/citation_enrichment.py",
]
# main.py changes for unrelated reasons; only the functions that build a corpus are frozen
FROZEN_FUNCTIONS = ["_build_corpus", "_harvest_facet_queries", "_prepare_query"]


def current_hashes() -> dict:
    import main
    out = {f: hashlib.sha256((ROOT / "backend" / f).read_bytes()).hexdigest() for f in FROZEN_FILES}
    for name in FROZEN_FUNCTIONS:
        out[f"main.{name}"] = hashlib.sha256(inspect.getsource(getattr(main, name)).encode()).hexdigest()
    return out


class TestEvaluationFreeze(unittest.TestCase):
    def test_frozen_code_unchanged(self):
        from services.paper_selection import PROTOCOL_VERSION
        freeze = json.loads(FREEZE.read_text())
        self.assertEqual(PROTOCOL_VERSION, freeze["protocol_version"],
                         "The selection protocol version changed after the evaluation freeze; record it in "
                         "eval/FREEZE.json as a post-freeze version.")
        changed = sorted(k for k, v in current_hashes().items() if freeze["hashes"].get(k) != v)
        self.assertEqual(changed, [], "Frozen search/selection code changed after the evaluation freeze "
                                      "(eval/PROTOCOL.md, 23 Nov 2026). See this test's docstring.")


if __name__ == "__main__":
    if "--write" in sys.argv:  # used once, at the freeze
        from services.paper_selection import PROTOCOL_VERSION
        FREEZE.write_text(json.dumps({
            "frozen": "2026-11-23", "commit": "47e40ef", "tag": "eval-freeze-2.5",
            "protocol_version": PROTOCOL_VERSION, "note": "Search and selection code measured in Session 6.",
            "hashes": current_hashes()}, indent=1) + "\n")
        print(f"wrote {FREEZE}")
    else:
        unittest.main()
