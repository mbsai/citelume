"""
What each search matched and what was retrieved ("retrieved N of M"), for one search request. PRISMA-S asks for
the number of records each database search returned; PubMed reports how many records a query matches, so a
reader can see how deep the retrieval went. Kept per request (a context variable) and stored with the corpus.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, List, Optional

_LOG: ContextVar[Optional[List[Dict]]] = ContextVar("search_log", default=None)
_LAST_MATCHED: ContextVar[Optional[Dict[str, int]]] = ContextVar("last_matched", default=None)


def begin_search_log() -> None:
    _LOG.set([])
    _LAST_MATCHED.set({})


def note_matched(source: str, query: str, matched: int) -> None:
    """A source that reports its total (PubMed) notes it here; the dispatcher reads it back for the log entry."""
    d = _LAST_MATCHED.get()
    if d is not None:
        d[f"{source}\n{query}"] = matched


def matched_for(source: str, query: str) -> Optional[int]:
    d = _LAST_MATCHED.get()
    return d.get(f"{source}\n{query}") if d else None


def record(source: str, search: str, query: str, retrieved: int, matched: Optional[int]) -> None:
    log = _LOG.get()
    if log is not None:
        log.append({"source": source, "search": search, "query": query, "retrieved": retrieved, "matched": matched})


def entries() -> List[Dict]:
    return list(_LOG.get() or [])


def retrieval_lines(log: List[Dict]) -> List[str]:
    """One line per search whose database reported how many records it matched."""
    return [f"{e['source']}, {e['search']}: retrieved {e['retrieved']:,} of {e['matched']:,} matching records"
            for e in log if e.get("matched") is not None]
