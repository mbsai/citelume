"""
Session 8 (E4, verification gates): re-run only the analysis of stored corpora, so each result holds both the
narrative as the model wrote it and the narrative after the quality gate (SynthesisEvidencePackage.ungated_narrative).

    CITELUME_LLM_API_KEY=... python -m eval.gates_runs --corpora runs/6_test_standard --out runs/8_gates \\
        --model nvidia/nemotron-3-ultra-550b-a55b:free

Selection is replayed from the stored records (no search; the frozen pipeline is untouched). One folder per
question with result.json; resumable (a question with a result is skipped). A run whose narrative was not written
by the model (quota, outage) is not saved, so the next run retries it.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.gates_runs")
    p.add_argument("--corpora", type=Path, required=True, help="batch-runner output folder with corpus.json.gz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    a = p.parse_args(argv)

    import main as app
    from models import ApiKeysConfig, Corpus

    cfg = ApiKeysConfig(llm_provider="openrouter", llm_model=a.model,
                        llm_api_key=os.environ.get("CITELUME_LLM_API_KEY", ""), enable_hf=False)
    a.out.mkdir(parents=True, exist_ok=True)
    for folder in sorted(x for x in a.corpora.iterdir() if (x / "corpus.json.gz").exists()):
        dest = a.out / folder.name
        if (dest / "result.json").exists():
            continue
        with gzip.open(folder / "corpus.json.gz", "rt", encoding="utf-8") as fh:
            corpus = Corpus.model_validate_json(fh.read())
        corpus.analysis = None
        try:
            result = asyncio.run(app._analyse(app._run_from_corpus(corpus, cfg)))
        except Exception as exc:  # the next run retries it
            print(f"{folder.name}: failed ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
            continue
        ungated = (result.synthesis_evidence.ungated_narrative if result.synthesis_evidence else {}) or {}
        if not ungated.get("model_written"):
            print(f"{folder.name}: the narrative was not written by the model (quota?); not saved", flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "result.json").write_text(result.model_dump_json(indent=0), encoding="utf-8")
        (dest / "meta.json").write_text(json.dumps({"model": a.model, "corpus_id": corpus.corpus_id,
                                                    "query": corpus.query}), encoding="utf-8")
        print(f"{folder.name}: saved ({len(result.warnings)} warnings)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
