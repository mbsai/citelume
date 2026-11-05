"""
Resilience and visibility for the literature sources (PubMed, Semantic Scholar/OpenAlex, arXiv, bioRxiv, ...).

* RetryingTransport retries transient failures - rate limits (429), server errors (5xx) and dropped connections -
  with short backoff, so a momentary NCBI or OpenAlex hiccup does not empty a source.
* report_source_failure records, per search, which sources still failed and why, so the results can say
  "PubMed was unavailable" instead of silently returning fewer papers.
"""
import asyncio
from contextvars import ContextVar
from typing import Dict, Optional

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 2
BACKOFF_S = (1.5, 4.0)
MAX_RETRY_AFTER_S = 8.0

# source name -> (last error, fallback used or ""), for the current search (None outside a search)
_SOURCE_FAILURES: ContextVar[Optional[Dict[str, tuple]]] = ContextVar("source_failures", default=None)
# Extra advice shown with a failure, e.g. how to lift a rate limit
_SOURCE_HINTS: Dict[str, str] = {
    "Semantic Scholar / OpenAlex": " OpenAlex now requires a free API key for regular use: add one under "
                                   "⚙ API Keys & Sources Setup (openalex.org/settings/api).",
}


class RetryingTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._inner.handle_async_request(request)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError):
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(BACKOFF_S[attempt])
                continue
            if response.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                return response
            try:
                wait = min(float(response.headers.get("retry-after") or BACKOFF_S[attempt]), MAX_RETRY_AFTER_S)
            except ValueError:
                wait = BACKOFF_S[attempt]
            await response.aclose()
            await asyncio.sleep(wait)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def retrying_transport() -> RetryingTransport:
    return RetryingTransport()


# OpenAlex API key for the current request (OpenAlex requires a free key for regular use)
_OPENALEX_KEY: ContextVar[str] = ContextVar("openalex_api_key", default="")


def set_openalex_api_key(key: str) -> None:
    _OPENALEX_KEY.set((key or "").strip())


def openalex_api_key() -> str:
    return _OPENALEX_KEY.get()


def begin_source_health() -> None:
    _SOURCE_FAILURES.set({})


def report_source_failure(source: str, exc: Exception, fallback: str = "") -> None:
    failures = _SOURCE_FAILURES.get()
    if failures is None:
        return
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        reason = "rate-limited (HTTP 429)" if code == 429 else f"HTTP {code}"
    elif isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        reason = "could not connect"
    elif isinstance(exc, httpx.TimeoutException):
        reason = "timed out"
    else:
        reason = str(exc) or type(exc).__name__
    failures[source] = (reason, fallback)


def source_failure_warnings() -> list:
    failures = _SOURCE_FAILURES.get() or {}
    out = []
    for src, (why, fallback) in failures.items():
        if fallback:
            out.append(f"Source unavailable: {src} {why} after retries; its records were retrieved through {fallback} instead.")
        else:
            hint = _SOURCE_HINTS.get(src, "") if "429" in why else ""
            out.append(f"Source unavailable: {src} {why} after retries; results come from the other sources only.{hint}")
    return out
