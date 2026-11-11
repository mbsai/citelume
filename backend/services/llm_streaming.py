"""
Streaming chat calls with stall detection.

A non-streamed call only reveals a problem when the whole reply fails to arrive. Streaming lets the caller see the
first token and every later one, so it can tell three situations apart within seconds:
  * the model is working (tokens keep arriving): keep waiting, however long the answer is;
  * the request is queued or dead (no first token): give up early, or race a backup model;
  * the reply stopped part-way (tokens stopped): keep what arrived and move on.

Two wire formats are supported: OpenAI-compatible server-sent events (OpenRouter and others) and Ollama's
newline-delimited JSON.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx


@dataclass
class StreamLimits:
    first_token_s: float      # give up if no answer text has started by then
    idle_s: float             # give up if answer text stops arriving for this long
    deadline_s: Optional[float] = None  # give up on the whole call after this long, whatever is happening


@dataclass
class ChatStreamResult:
    status: int = 200
    text: str = ""
    finish_reason: str = ""
    model: str = ""
    error: str = ""           # provider-reported error message
    stalled: str = ""         # why the stream was abandoned ("" if it ended normally)
    saw_reasoning: bool = False
    retry_after: Optional[float] = None
    first_token_after_s: Optional[float] = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error and not self.stalled and bool(self.text.strip())


def _retry_after(headers: httpx.Headers) -> Optional[float]:
    try:
        return float(headers.get("retry-after")) if headers.get("retry-after") else None
    except ValueError:
        return None


async def _read_error(resp: httpx.Response) -> str:
    try:
        body = (await resp.aread()).decode("utf-8", "replace")
    except Exception:
        return ""
    try:
        data = json.loads(body)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            msg = err.get("message") or str(err)
            raw = (err.get("metadata") or {}).get("raw")
            return f"{msg} — {str(raw)[:220]}" if raw else msg
        if err:
            return str(err)
    except Exception:
        pass
    return body[:220]


async def _lines_with_limits(lines: AsyncIterator[str], limits: StreamLimits, state: Dict[str, Any]) -> AsyncIterator[str]:
    """Yields lines, raising TimeoutError (with a reason in state['stalled']) when a limit is exceeded.
    The caller updates state['last_token'] and state['got_token'] as answer text arrives."""
    it = lines.__aiter__()
    while True:
        now = time.monotonic()
        waits = [(state["last_token"] + (limits.idle_s if state["got_token"] else limits.first_token_s) - now,
                  f"no output for {limits.idle_s:.0f}s" if state["got_token"]
                  else f"no answer started within {limits.first_token_s:.0f}s")]
        if limits.deadline_s is not None:
            waits.append((state["started"] + limits.deadline_s - now, f"exceeded the {limits.deadline_s:.0f}s time budget"))
        remaining, reason = min(waits)
        if remaining <= 0:
            state["stalled"] = reason
            raise asyncio.TimeoutError(reason)
        try:
            line = await asyncio.wait_for(it.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            state["stalled"] = reason
            raise
        yield line


async def stream_openai_chat(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    limits: StreamLimits,
    first_token: Optional[asyncio.Event] = None,
) -> ChatStreamResult:
    """One streamed OpenAI-compatible chat completion. Never raises for HTTP errors or stalls: see the result."""
    res = ChatStreamResult()
    state = {"started": time.monotonic(), "last_token": time.monotonic(), "got_token": False, "stalled": ""}
    parts = []
    try:
        async with client.stream("POST", url, headers=headers, json={**payload, "stream": True}) as resp:
            res.status = resp.status_code
            if resp.status_code != 200:
                res.retry_after = _retry_after(resp.headers)
                res.error = await _read_error(resp) or f"HTTP {resp.status_code}"
                return res
            async for line in _lines_with_limits(resp.aiter_lines(), limits, state):
                if not line.startswith("data:"):
                    continue  # keep-alive comments and blank separators
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    err = obj["error"]
                    res.error = (err.get("message") if isinstance(err, dict) else None) or str(err)
                    break
                res.model = obj.get("model") or res.model
                choice = (obj.get("choices") or [{}])[0] or {}
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, list):
                    piece = "".join(p.get("text", "") for p in piece if isinstance(p, dict))
                if piece:
                    parts.append(piece)
                    state["last_token"] = time.monotonic()
                    if not state["got_token"]:
                        state["got_token"] = True
                        res.first_token_after_s = round(state["last_token"] - state["started"], 1)
                        if first_token is not None:
                            first_token.set()
                elif delta.get("reasoning") or delta.get("reasoning_content"):
                    res.saw_reasoning = True
                    state["last_token"] = time.monotonic()  # hidden reasoning is still the model working
                if choice.get("finish_reason"):
                    res.finish_reason = choice["finish_reason"]
    except asyncio.TimeoutError:
        res.stalled = state["stalled"] or "timed out"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        res.stalled = f"{type(exc).__name__}: {exc}".rstrip(": ")
    res.text = "".join(parts)
    res.elapsed_s = round(time.monotonic() - state["started"], 1)
    return res


async def stream_ollama_chat(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    limits: StreamLimits,
) -> ChatStreamResult:
    """One streamed Ollama /api/chat call (newline-delimited JSON)."""
    res = ChatStreamResult()
    state = {"started": time.monotonic(), "last_token": time.monotonic(), "got_token": False, "stalled": ""}
    parts = []
    try:
        async with client.stream("POST", url, headers=headers, json={**payload, "stream": True}) as resp:
            res.status = resp.status_code
            if resp.status_code != 200:
                res.error = await _read_error(resp) or f"HTTP {resp.status_code}"
                return res
            async for line in _lines_with_limits(resp.aiter_lines(), limits, state):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("error"):
                    res.error = str(obj["error"])
                    break
                msg = obj.get("message") or {}
                piece = msg.get("content")
                if piece:
                    parts.append(piece)
                    state["last_token"] = time.monotonic()
                    if not state["got_token"]:
                        state["got_token"] = True
                        res.first_token_after_s = round(state["last_token"] - state["started"], 1)
                elif msg.get("thinking"):
                    res.saw_reasoning = True
                    state["last_token"] = time.monotonic()
                if obj.get("done"):
                    res.finish_reason = obj.get("done_reason") or "stop"
                    res.model = obj.get("model") or res.model
                    break
    except asyncio.TimeoutError:
        res.stalled = state["stalled"] or "timed out"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        res.stalled = f"{type(exc).__name__}: {exc}".rstrip(": ")
    res.text = "".join(parts)
    res.elapsed_s = round(time.monotonic() - state["started"], 1)
    return res


def salvage_partial_json(text: str) -> Optional[str]:
    """
    Makes valid JSON from a reply that stopped part-way, keeping what was complete:
      * a root object whose value is a list of objects ({"items": [{...}, {...}, {"half):  the complete items;
      * a root object ending inside a long string ({"content": "Sentence one. Sentence tw):  the complete sentences.
    Returns None when nothing complete can be kept.
    """
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    stack = []
    in_string = escape = False
    last_item_end = None        # index just after the last complete element of a list directly under the root
    string_start = None
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string, string_start = True, i
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            if stack == ["{", "["]:
                last_item_end = i + 1
            if not stack:
                return s[: i + 1]  # the JSON was complete after all
    if last_item_end is not None:
        return s[:last_item_end] + "]}"
    if in_string and stack == ["{"] and string_start is not None:
        body = s[string_start + 1:]
        if body.endswith("\\"):
            body = body[:-1]
        cut = max(body.rfind(". "), body.rfind(".\\n"), body.rfind("] "))
        if cut > 200:
            return s[: string_start + 1] + body[: cut + 1] + '"}'
    return None
