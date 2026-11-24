"""
Builds the evaluation archive for deposit (Zenodo): everything needed to reproduce the paper's tables without
searching again.

    python -m eval.build_archive --out dist

Contents: the protocol, the benchmark and its split, the freeze record, the results, and for every evaluation run
the stored corpora (corpus.json.gz), raw search output (raw.json.gz, where saved), results and run records. Not
included: imported records or uploaded PDFs (none were used in the evaluation; the builder refuses any corpus that
holds them), API keys (never written by the batch runner; the builder checks), the drive's '._' metadata files.

Writes dist/citelume-evaluation-<date>.tar.gz, with MANIFEST.sha256 and README.md inside.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from datetime import date
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RUNS = ["5b_dev", "5c_dev", "5c_dev_full", "5d_dev", "6_test_prisma", "6_test_standard", "7_search_r1",
        "7_search_r2", "7_rescreen", "7_replan", "8_gates", "8_dev_standard"]
EVAL_FILES = ["PROTOCOL.md", "FREEZE.json", "benchmark", "results", "annotation/GUIDE.md", "annotation/key.json",
              "annotation/merge_key.json"]
KEY = re.compile(rb"sk-or-v1-[0-9a-f]{20,}|sk-ant-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_-]{30,}")

README = """# CiteLume evaluation archive

Stored corpora, runs and results behind the CiteLume evaluation (code: https://github.com/mbsai/citelume; the
search and selection code measured is tagged eval-freeze-2.5). Built {built} from commit {commit}.

- eval/PROTOCOL.md: the pre-registered protocol and its amendments.
- eval/benchmark/: the 33 CLEF TAR 2019 intervention reviews, their included studies and the development / test
  split.
- eval/results/: every results table and figure.
- runs/<run>/<review>/: corpus.json.gz (the records found, the plan, every selection and screening decision; replay
  with `python -m services.replay`), raw.json.gz (records before duplicate merging, where saved), result.json (the
  analysis) and run.json (settings, model, timings, versions).

Reproduce the recall table from the stored corpora (no search, no model):

    python -m eval.session6 --prisma runs/6_test_prisma --standard runs/6_test_standard --set test --name 6_test

The corpora hold the bibliographic records (titles and abstracts) as PubMed, Europe PMC, Semantic Scholar,
OpenAlex, arXiv and bioRxiv returned them; abstracts remain the copyright of their publishers and are included for
research reproducibility only. No imported database exports, uploaded PDFs or API keys are included.
"""


def _git_commit() -> str:
    head = (REPO / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref = REPO / ".git" / head.split()[1]
        if ref.exists():
            return ref.read_text().strip()[:7]
        packed = (REPO / ".git" / "packed-refs").read_text()
        m = re.search(rf"([0-9a-f]{{40}}) {re.escape(head.split()[1])}", packed)
        return m.group(1)[:7] if m else ""
    return head[:7]


def files_to_add() -> List[Path]:
    out = []
    for name in EVAL_FILES:
        p = ROOT / name
        out += [x for x in ([p] if p.is_file() else sorted(p.rglob("*"))) if x.is_file()]
    for run in RUNS:
        folder = REPO / "runs" / run
        if folder.exists():
            out += sorted(x for x in folder.rglob("*") if x.is_file())
    return [p for p in out if not p.name.startswith("._") and p.suffix not in (".log", ".tmp")]


def check(path: Path, data: bytes) -> None:
    raw = gzip.decompress(data) if path.suffix == ".gz" else data
    if KEY.search(raw):
        raise SystemExit(f"refusing: something that looks like an API key in {path}")
    if path.name == "corpus.json.gz":
        c = json.loads(raw)
        if c.get("imports") or c.get("user_pdfs"):
            raise SystemExit(f"refusing: {path} holds imported records or user PDFs")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.build_archive")
    p.add_argument("--out", type=Path, default=REPO / "dist")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    built = date.today().isoformat()
    dest = a.out / f"citelume-evaluation-{built}.tar.gz"
    manifest = []
    with tarfile.open(dest, "w:gz") as tar:
        for f in files_to_add():
            data = f.read_bytes()
            check(f, data)
            arc = f.relative_to(REPO).as_posix()
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {arc}")
            info = tarfile.TarInfo(arc)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
        for name, text in (("README.md", README.format(built=built, commit=_git_commit())),
                           ("MANIFEST.sha256", "\n".join(manifest) + "\n")):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    print(f"{dest}: {len(manifest)} files, {dest.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
