"""
Headless runner: run many questions through the whole pipeline from the command line, with no browser.

    python -m services.batch run questions.jsonl --mode standard --out runs/
    python -m services.batch run questions.jsonl --mode deep_research --review-type systematic_prisma --out runs/

questions.jsonl holds one JSON object per line:

    {"id": "vitd-fractures", "query": "Does vitamin D supplementation prevent fractures in older adults?"}

Optional fields on a line override the command-line options for that question: "mode", "review_type"
(narrative, systematic_prisma or scoping), "year_from", "year_to", "max_per_source". Lines starting with # are
skipped. The id names the question's output folder (letters, digits, '-', '_' and '.'); without one, the line number
is used.

For each question, <out>/<id>/ receives:
    raw.json.gz      with --save-raw: every record the searches returned, before duplicates were merged
    corpus.json.gz   the stored corpus: every record, the plan, the selection and screening decisions (replayable
                     with `python -m services.replay`)
    result.json      the full analysis: synthesis, evidence table, gaps, map, risk of bias
    run.json         status, timings per stage, the model and settings used, versions and any error
and <out>/summary.csv has one row per question.

Resumable: a question whose run.json says "done" is skipped, so an interrupted batch continues where it stopped.
Failed questions are tried again unless --skip-failed is given; --rerun runs every question again. With a model, a
systematic-style run whose records were not screened by the model (more than 5% unscreened, for example when a free
quota runs out) is marked failed, so it is retried rather than counted.

Rate limits: questions run one at a time, with --pause seconds between them (default 10); the source clients retry
rate-limited calls and pace Semantic Scholar themselves.

Keys: your own, never the public demo's. Give them in a JSON file with the same fields as the page's settings
(--config keys.json, for example {"llm_provider": "openrouter", "llm_api_key": "...", "llm_model": "..."}), or in
the environment: CITELUME_LLM_PROVIDER, CITELUME_LLM_API_KEY, CITELUME_LLM_MODEL, CITELUME_LLM_BASE_URL and
the literature keys PUBMED_API_KEY, S2_API_KEY, OPENALEX_API_KEY. The runner refuses to start in demo mode
(CITELUME_DEMO) or with the demo's server key. Keys are never written to the output. Without a model key the
pipeline runs rule-based, and run.json says so.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RUNNER_VERSION = "citelume-batch/1.0"
MODES = ("standard", "deep_research")
REVIEW_TYPES = ("narrative", "systematic_prisma", "scoping")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_ENV_FIELDS = {
    "CITELUME_LLM_PROVIDER": "llm_provider",
    "CITELUME_LLM_API_KEY": "llm_api_key",
    "CITELUME_LLM_MODEL": "llm_model",
    "CITELUME_LLM_BASE_URL": "llm_base_url",
    "PUBMED_API_KEY": "pubmed_api_key",
    "S2_API_KEY": "s2_api_key",
    "OPENALEX_API_KEY": "openalex_api_key",
    "HF_API_KEY": "hf_api_key",
}
SUMMARY_COLUMNS = ("id", "status", "mode", "review_type", "records", "included", "model", "corpus_s", "analysis_s",
                   "total_s", "corpus_id", "content_hash", "error")


class BatchError(Exception):
    """A problem with the batch's inputs or settings, reported before any question runs."""


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------
def read_questions(path: Path) -> List[Dict[str, Any]]:
    questions, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchError(f"{path}:{n}: not valid JSON ({exc.msg})")
        if isinstance(q, str):
            q = {"query": q}
        if not isinstance(q, dict) or not str(q.get("query") or "").strip():
            raise BatchError(f"{path}:{n}: each line needs a \"query\"")
        qid = str(q.get("id") or f"q{n:03d}")
        if not _ID.match(qid):
            raise BatchError(f"{path}:{n}: id {qid!r} may use only letters, digits, '-', '_' and '.'")
        if qid in seen:
            raise BatchError(f"{path}:{n}: id {qid!r} is used twice")
        if q.get("mode") and q["mode"] not in MODES:
            raise BatchError(f"{path}:{n}: mode must be one of {', '.join(MODES)}")
        if q.get("review_type") and q["review_type"] not in REVIEW_TYPES:
            raise BatchError(f"{path}:{n}: review_type must be one of {', '.join(REVIEW_TYPES)}")
        seen.add(qid)
        questions.append({**q, "id": qid, "query": str(q["query"]).strip()})
    if not questions:
        raise BatchError(f"{path} holds no questions")
    return questions


def load_config(config_path: Optional[Path], model: str = ""):
    """The user's keys and model: a JSON file, else the environment. Refuses the public demo's settings."""
    from models import ApiKeysConfig
    from services import demo_mode

    if demo_mode.demo_enabled():
        raise BatchError("CITELUME_DEMO is set. The batch runner uses your own keys only; unset it to run.")
    if config_path:
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchError(f"Could not read {config_path}: {exc}")
    else:
        data = {field: os.environ[env].strip() for env, field in _ENV_FIELDS.items() if os.environ.get(env, "").strip()}
    if model:
        data["llm_model"] = model
    cfg = ApiKeysConfig(**data)
    demo_key = os.environ.get("DEMO_OPENROUTER_API_KEY", "").strip()
    if demo_key and cfg.llm_api_key.strip() == demo_key:
        raise BatchError("That is the public demo's server key. Use your own key for batch runs.")
    return cfg


def uses_model(cfg) -> bool:
    return bool(cfg.llm_api_key) or cfg.llm_provider in ("ollama", "custom")


def model_label(cfg) -> str:
    return f"{cfg.llm_provider}:{cfg.llm_model}" if uses_model(cfg) else "none (rule-based)"


def public_settings(cfg) -> Dict[str, Any]:
    """The settings worth recording, without any key."""
    return {
        "llm_provider": cfg.llm_provider if uses_model(cfg) else "",
        "llm_model": cfg.llm_model if uses_model(cfg) else "",
        "llm_fallback_models": list(cfg.llm_fallback_models),
        "llm_base_url": cfg.llm_base_url,
        "literature_keys": sorted(k for k in ("pubmed_api_key", "s2_api_key", "openalex_api_key", "hf_api_key")
                                  if getattr(cfg, k)),
        "sources": {s: getattr(cfg, f"enable_{s}") for s in ("pubmed", "arxiv", "biorxiv", "s2", "hf", "fulltext")},
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def versions() -> Dict[str, str]:
    from services.ai_screening import SCREENING_PROMPT_VERSION
    from services.paper_selection import PROTOCOL_VERSION
    from services.risk_of_bias_assessment import ROB_PROMPT_VERSION

    return {"runner": RUNNER_VERSION, "selection_protocol": PROTOCOL_VERSION, "screening_prompt": SCREENING_PROMPT_VERSION,
            "risk_of_bias_prompt": ROB_PROMPT_VERSION, "code_commit": _git_commit(), "python": platform.python_version()}


# ---------------------------------------------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_corpus(path: Path, corpus) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write(corpus.model_dump_json())
    tmp.replace(path)


# A run with a model must have had its records screened by it: when a free quota runs out mid-run, the pipeline
# carries on with the word-based decisions, which is right for a user but would make an evaluation run measure the
# wrong thing. Such a run is marked failed (the corpus is kept) so the next run of the batch retries it.
MAX_UNSCREENED_SHARE = 0.05


def screening_problem(corpus, cfg) -> str:
    """Why a run's model screening is incomplete ("" when it is complete or not part of the run)."""
    if not uses_model(cfg) or corpus.selection_method == "narrative" or corpus.selection_report is None:
        return ""
    decisions = corpus.selection_report.screening_decisions
    if not decisions:
        return ""
    unscreened = sum(1 for d in decisions if not d.screened)
    if unscreened == len(decisions):
        return f"the model screened none of the {len(decisions)} records (no model answer: quota or outage?)"
    if unscreened / len(decisions) > MAX_UNSCREENED_SHARE:
        return f"the model left {unscreened} of {len(decisions)} records unscreened (quota or outage?)"
    return ""


def model_endpoint(cfg) -> str:
    if not uses_model(cfg):
        return ""
    return "free" if cfg.llm_model.endswith(":free") else "paid"


def _write_raw(path: Path, records) -> None:
    """The records as the searches returned them, before duplicates were merged (for evaluation)."""
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write("[" + ",".join(r.model_dump_json() for r in records) + "]")
    tmp.replace(path)


def build_request(q: Dict[str, Any], defaults: Dict[str, Any], cfg):
    from models import SearchRequest

    fields = {"query": q["query"], "mode": q.get("mode") or defaults["mode"],
              "review_framework": q.get("review_type") or defaults["review_type"], "config": cfg}
    for key in ("year_from", "year_to", "max_per_source"):
        if q.get(key) is not None:
            fields[key] = q[key]
    if defaults.get("pubmed_depth"):
        fields["pubmed_depth"] = defaults["pubmed_depth"]
    return SearchRequest(**fields)


async def run_question(q: Dict[str, Any], folder: Path, defaults: Dict[str, Any], cfg) -> Dict[str, Any]:
    """Runs one question end to end and writes its files. Returns the run record (also written as run.json)."""
    import main
    from fastapi import HTTPException

    folder.mkdir(parents=True, exist_ok=True)
    req = build_request(q, defaults, cfg)
    record: Dict[str, Any] = {
        "id": q["id"], "query": q["query"], "status": "running", "mode": req.mode,
        "review_type": req.review_framework, "model": model_label(cfg), "model_endpoint": model_endpoint(cfg),
        "settings": public_settings(cfg),
        "versions": versions(), "started_at": _now(), "finished_at": "", "timings_s": {}, "error": "",
    }
    _write_json(folder / "run.json", record)
    t0 = time.monotonic()
    try:
        req, note = await main._prepare_query(req)
        if note:
            record["query_note"] = note
        t1 = time.monotonic()
        run = await main._build_corpus(req)
        t2 = time.monotonic()
        record["timings_s"]["corpus"] = round(t2 - t1, 1)
        _write_corpus(folder / "corpus.json.gz", run.corpus)
        if defaults.get("save_raw"):
            _write_raw(folder / "raw.json.gz", run.raw_records)
        record.update(corpus_id=run.corpus.corpus_id, content_hash=run.corpus.content_hash,
                      records=len(run.corpus.records), included=len(run.corpus.included_ids))
        _write_json(folder / "run.json", record)  # the corpus survives even if the analysis fails
        problem = screening_problem(run.corpus, cfg)
        if problem:
            raise RuntimeError(f"Screening incomplete: {problem}. The corpus is kept; the next run retries it.")

        result = await main._analyse(run)
        t3 = time.monotonic()
        record["timings_s"]["analysis"] = round(t3 - t2, 1)
        (folder / "result.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        _write_corpus(folder / "corpus.json.gz", run.corpus)  # now with the analysis stored in it
        record["model"] = run.corpus.analysis_model or record["model"]
        record["warnings"] = list(result.warnings)
        record["status"] = "done"
    except HTTPException as exc:
        record.update(status="failed", error=str(exc.detail))
    except Exception as exc:  # recorded and the batch moves on; the next run tries the question again
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:2000])
    record["timings_s"]["total"] = round(time.monotonic() - t0, 1)
    record["finished_at"] = _now()
    _write_json(folder / "run.json", record)
    return record


# ---------------------------------------------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------------------------------------------
def previous_status(folder: Path) -> str:
    try:
        return json.loads((folder / "run.json").read_text(encoding="utf-8")).get("status", "")
    except (OSError, json.JSONDecodeError):
        return ""


def write_summary(out: Path, questions: List[Dict[str, Any]]) -> Path:
    path = out / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLUMNS)
        for q in questions:
            try:
                r = json.loads((out / q["id"] / "run.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                r = {"id": q["id"], "status": "not run"}
            t = r.get("timings_s", {})
            w.writerow([r.get("id", q["id"]), r.get("status", ""), r.get("mode", ""), r.get("review_type", ""),
                        r.get("records", ""), r.get("included", ""), r.get("model", ""), t.get("corpus", ""),
                        t.get("analysis", ""), t.get("total", ""), r.get("corpus_id", ""), r.get("content_hash", ""),
                        r.get("error", "")])
    return path


async def run_batch(questions: List[Dict[str, Any]], out: Path, defaults: Dict[str, Any], cfg, *,
                    rerun: bool = False, skip_failed: bool = False, pause_s: float = 10.0,
                    log=print) -> Tuple[int, int, int]:
    """Runs the questions one at a time. Returns (done, failed, skipped)."""
    out.mkdir(parents=True, exist_ok=True)
    done = failed = skipped = 0
    ran_one = False
    for i, q in enumerate(questions, start=1):
        folder = out / q["id"]
        status = previous_status(folder)
        if not rerun and (status == "done" or (skip_failed and status == "failed")):
            log(f"[{i}/{len(questions)}] {q['id']}: {status} before, skipped")
            skipped += 1
            continue
        if ran_one and pause_s > 0:
            await asyncio.sleep(pause_s)
        log(f"[{i}/{len(questions)}] {q['id']}: running ({q.get('mode') or defaults['mode']})")
        record = await run_question(q, folder, defaults, cfg)
        ran_one = True
        if record["status"] == "done":
            done += 1
            log(f"[{i}/{len(questions)}] {q['id']}: done in {record['timings_s']['total']} s, "
                f"{record.get('included', 0)} of {record.get('records', 0)} records included")
        else:
            failed += 1
            log(f"[{i}/{len(questions)}] {q['id']}: FAILED - {record['error']}")
        write_summary(out, questions)
    write_summary(out, questions)
    return done, failed, skipped


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m services.batch", description="Run questions without the browser.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run every question in a JSONL file")
    r.add_argument("questions", type=Path)
    r.add_argument("--out", type=Path, required=True, help="output folder (one subfolder per question)")
    r.add_argument("--mode", choices=MODES, default="standard")
    r.add_argument("--review-type", choices=REVIEW_TYPES, default="narrative",
                   help="review type for deep_research runs (default narrative)")
    r.add_argument("--config", type=Path, help="JSON file with your keys and model (else the environment)")
    r.add_argument("--model", default="", help="model name, overriding the config")
    r.add_argument("--pause", type=float, default=10.0, help="seconds between questions (default 10)")
    r.add_argument("--rerun", action="store_true", help="run every question again, even those already done")
    r.add_argument("--skip-failed", action="store_true", help="do not retry questions that failed before")
    r.add_argument("--pubmed-depth", type=int, default=None,
                   help="records the PubMed core search fetches (20-2000; default: the review type's number)")
    r.add_argument("--save-raw", action="store_true",
                   help="also save raw.json.gz: every record found, before duplicates were merged (evaluation)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        questions = read_questions(args.questions)
        cfg = load_config(args.config, args.model)
    except BatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{len(questions)} questions, mode {args.mode}, model {model_label(cfg)}, output {args.out}")
    if not uses_model(cfg):
        print("No model key given: the pipeline runs rule-based (no screening model, no AI synthesis).")
    defaults = {"mode": args.mode, "review_type": args.review_type, "save_raw": args.save_raw,
                "pubmed_depth": args.pubmed_depth}
    done, failed, skipped = asyncio.run(run_batch(questions, args.out, defaults, cfg, rerun=args.rerun,
                                                  skip_failed=args.skip_failed, pause_s=args.pause))
    print(f"Finished: {done} done, {failed} failed, {skipped} skipped. Summary: {args.out / 'summary.csv'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
