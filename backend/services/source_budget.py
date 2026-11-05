"""
Spending and rate limits for keyed literature APIs used with the server's own keys.

OpenAlex bills API keys in credits (a search costs 10; the free plan includes 10,000 credits a day). When a search
uses the server's OpenAlex key (DEMO_OPENALEX_API_KEY), calls are counted against a daily credit cap
(OPENALEX_DAILY_CREDIT_CAP, default 5,000 = half the free daily budget, leaving the rest for the key owner) and a
per-search call cap. Visitors' own OpenAlex keys are not counted. Counts reset at 00:00 UTC and are saved in
DEMO_STATE_DIR when it is set, so a restart does not reset them.

Semantic Scholar keys are rate limited (typically 1 request per second), so keyed calls are spaced out.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

OPENALEX_SEARCH_CREDITS = 10


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class _DailyCredits:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = ""
        self._used = 0

    def _path(self) -> Optional[Path]:
        d = os.environ.get("DEMO_STATE_DIR", "")
        return Path(d) / "openalex_credits.json" if d and Path(d).is_dir() else None

    def _sync(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._day == today:
            return
        self._day, self._used = today, 0
        p = self._path()
        if p and p.exists():
            try:
                data = json.loads(p.read_text())
                if data.get("day") == today:
                    self._used = int(data.get("used", 0))
            except Exception:
                pass

    def _save(self) -> None:
        p = self._path()
        if p:
            try:
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps({"day": self._day, "used": self._used}))
                tmp.replace(p)
            except Exception:
                pass

    def try_spend(self, credits: int, cap: int) -> bool:
        with self._lock:
            self._sync()
            if self._used + credits > cap:
                return False
            self._used += credits
            self._save()
            return True

    def used_today(self) -> int:
        with self._lock:
            self._sync()
            return self._used

    def reset(self) -> None:
        with self._lock:
            self._day, self._used = "", 0


OPENALEX_CREDITS = _DailyCredits()
_SEARCH_OPENALEX_CALLS: ContextVar[int] = ContextVar("search_openalex_calls", default=0)


def begin_search_budget() -> None:
    _SEARCH_OPENALEX_CALLS.set(0)


def openalex_call_allowed(key: str) -> Optional[str]:
    """None if this OpenAlex search may go ahead (and records its cost); otherwise the reason it may not."""
    server_key = os.environ.get("DEMO_OPENALEX_API_KEY", "").strip()
    if not key or key != server_key:
        return None  # keyless, or the visitor's own key: not ours to budget
    per_search = _env_int("OPENALEX_MAX_CALLS_PER_SEARCH", 12)
    if _SEARCH_OPENALEX_CALLS.get() >= per_search:
        return f"this search already used its {per_search} OpenAlex lookups"
    if not OPENALEX_CREDITS.try_spend(OPENALEX_SEARCH_CREDITS, _env_int("OPENALEX_DAILY_CREDIT_CAP", 5000)):
        return "the demo's daily OpenAlex allowance is used up (resets 00:00 UTC)"
    _SEARCH_OPENALEX_CALLS.set(_SEARCH_OPENALEX_CALLS.get() + 1)
    return None


# --- Semantic Scholar pacing -------------------------------------------------------------------------------
_S2_LOCK: Optional[asyncio.Lock] = None
_S2_LAST = 0.0


async def pace_semantic_scholar(keyed: bool) -> None:
    """Space keyed Semantic Scholar requests at least S2_MIN_INTERVAL_S apart (default 1.1 s)."""
    global _S2_LOCK, _S2_LAST
    if not keyed:
        return
    if _S2_LOCK is None:
        _S2_LOCK = asyncio.Lock()
    interval = float(os.environ.get("S2_MIN_INTERVAL_S", "1.1"))
    async with _S2_LOCK:
        wait = _S2_LAST + interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _S2_LAST = time.monotonic()
