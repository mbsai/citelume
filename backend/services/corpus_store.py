"""
Where corpora are kept between requests.

Local installs write one gzipped JSON file per corpus under CORPUS_DIR (default backend/.cache/corpora) and delete
it after CORPUS_TTL_HOURS (default 168). The public demo keeps corpora in memory only: the demo deliberately writes
no question text to disk or logs, and a corpus contains the question. Set CORPUS_PERSIST=1 to write files there too.

Corpus IDs are random and act as the access key, so they are validated strictly before touching the file system.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from models import Corpus
from services import demo_mode

_ID = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_DIR = Path(__file__).resolve().parent.parent / ".cache" / "corpora"
MAX_IN_MEMORY = 40


def _ttl_seconds() -> float:
    default = "24" if demo_mode.demo_enabled() else "168"
    try:
        return float(os.environ.get("CORPUS_TTL_HOURS", default)) * 3600
    except ValueError:
        return float(default) * 3600


def _persist() -> bool:
    flag = os.environ.get("CORPUS_PERSIST")
    if flag is not None:
        return flag.strip() not in ("", "0", "false", "no")
    return not demo_mode.demo_enabled()


def _dir() -> Path:
    return Path(os.environ.get("CORPUS_DIR") or DEFAULT_DIR)


# corpus_id -> (stored at, corpus); most recently used last
_MEMORY: "OrderedDict[str, tuple]" = OrderedDict()


def valid_id(corpus_id: str) -> bool:
    return bool(_ID.match(corpus_id or ""))


def save(corpus: Corpus) -> None:
    if not valid_id(corpus.corpus_id):
        raise ValueError("Invalid corpus id")
    _MEMORY[corpus.corpus_id] = (time.time(), corpus)
    _MEMORY.move_to_end(corpus.corpus_id)
    while len(_MEMORY) > MAX_IN_MEMORY:
        _MEMORY.popitem(last=False)
    if _persist():
        try:
            folder = _dir()
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f"{corpus.corpus_id}.json.gz.tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                fh.write(corpus.model_dump_json())
            tmp.replace(folder / f"{corpus.corpus_id}.json.gz")
            purge_expired()
        except OSError as exc:
            # The search result is still returned; only later re-use of this corpus is lost
            print(f"[corpus-store] could not write corpus file: {type(exc).__name__}", flush=True)


def load(corpus_id: str) -> Optional[Corpus]:
    if not valid_id(corpus_id):
        return None
    ttl = _ttl_seconds()
    hit = _MEMORY.get(corpus_id)
    if hit is not None:
        if time.time() - hit[0] <= ttl:
            _MEMORY.move_to_end(corpus_id)
            return hit[1]
        _MEMORY.pop(corpus_id, None)
    if not _persist():
        return None
    path = _dir() / f"{corpus_id}.json.gz"
    try:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl:
            path.unlink(missing_ok=True)
            return None
        return read_file(path)
    except (OSError, ValueError):
        return None


def read_file(path: Path) -> Corpus:
    """Reads a corpus from a .json or .json.gz file (also used by the replay command)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return Corpus(**json.load(fh))


def delete(corpus_id: str) -> None:
    if not valid_id(corpus_id):
        return
    _MEMORY.pop(corpus_id, None)
    try:
        (_dir() / f"{corpus_id}.json.gz").unlink(missing_ok=True)
    except OSError:
        pass


def purge_expired() -> int:
    """Deletes corpus files older than the time limit. Returns how many were removed."""
    removed = 0
    ttl = _ttl_seconds()
    try:
        for path in _dir().glob("*.json.gz"):
            if time.time() - path.stat().st_mtime > ttl:
                path.unlink(missing_ok=True)
                removed += 1
    except OSError:
        pass
    return removed
