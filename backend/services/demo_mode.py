"""
Public demo mode (e.g. a Hugging Face Space), switched on with the environment variable CITELUME_DEMO=1.

- Visitors without their own LLM key run on the server's OpenRouter key, restricted to NVIDIA's free models,
  and are limited per IP address per UTC day (default: 3 Standard + 1 Deep Research run).
- Visitors who enter their own LLM key are not counted against those limits (only a generous abuse cap applies).
  Their key is used for that request only and is never stored or logged.
- The server's key never leaves the server.

Environment variables:
  CITELUME_DEMO=1                    turn demo mode on
  DEMO_OPENROUTER_API_KEY          server-side OpenRouter key (set as a Space secret)
  DEMO_MODEL                       default model (must be an NVIDIA ':free' model)
  DEMO_STANDARD_PER_DAY=3, DEMO_DEEP_PER_DAY=1, DEMO_OWN_KEY_PER_DAY=30
  DEMO_MAX_CONCURRENT=2            searches running at once (free CPU Spaces are small)
  DEMO_PROXY_HOPS=1                trusted reverse proxies in front of the app (for X-Forwarded-For)
  DEMO_OPENALEX_API_KEY, DEMO_S2_API_KEY, DEMO_PUBMED_API_KEY   optional literature-source keys
  DEMO_STATE_DIR                   where usage counts are saved (default /data if present, else memory only)
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from models import ApiKeysConfig

# NVIDIA's free models on OpenRouter that can write the review (the content-safety classifier cannot)
NVIDIA_FREE_MODELS: List[str] = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def demo_enabled() -> bool:
    return os.environ.get("CITELUME_DEMO", "").strip().lower() in ("1", "true", "yes", "on")


def demo_limits() -> Dict[str, int]:
    return {
        "standard": _env_int("DEMO_STANDARD_PER_DAY", 3),
        "deep_research": _env_int("DEMO_DEEP_PER_DAY", 1),
        "own_key": _env_int("DEMO_OWN_KEY_PER_DAY", 30),
    }


def allowed_models() -> List[str]:
    return [m for m in NVIDIA_FREE_MODELS if m.startswith("nvidia/") and m.endswith(":free")]


def default_model() -> str:
    m = os.environ.get("DEMO_MODEL", "").strip()
    return m if m in allowed_models() else allowed_models()[0]


def server_key_configured() -> bool:
    return bool(os.environ.get("DEMO_OPENROUTER_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Client identity
# ---------------------------------------------------------------------------

def client_ip(forwarded_for: Optional[str], peer: Optional[str]) -> str:
    """
    The visitor's IP. Each trusted proxy appends the address it received the request from to X-Forwarded-For,
    so the real client is the entry `DEMO_PROXY_HOPS` from the right; anything further left is client-supplied
    and could be forged.
    """
    hops = max(0, _env_int("DEMO_PROXY_HOPS", 1))
    parts = [p.strip() for p in (forwarded_for or "").split(",") if p.strip()]
    if hops and parts:
        candidate = parts[-hops] if len(parts) >= hops else parts[0]
        try:
            return _quota_key(ipaddress.ip_address(candidate))
        except ValueError:
            pass
    try:
        return _quota_key(ipaddress.ip_address(peer or ""))
    except ValueError:
        return peer or "unknown"


def _quota_key(ip: "ipaddress._BaseAddress") -> str:
    # IPv6 devices rotate their addresses within one network, so count the /64 network (one household or phone)
    if ip.version == 6 and not ip.ipv4_mapped:
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip.ipv4_mapped or ip) if ip.version == 6 else str(ip)


# ---------------------------------------------------------------------------
# Daily usage counts (per IP, per UTC day)
# ---------------------------------------------------------------------------

class _UsageStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = ""
        self._counts: Dict[str, Dict[str, int]] = {}
        state_dir = os.environ.get("DEMO_STATE_DIR") or ("/data" if Path("/data").is_dir() else "")
        self._path = Path(state_dir) / "citelume_demo_usage.json" if state_dir else None
        self._load()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if data.get("day") == self._today():
                self._day, self._counts = data["day"], data.get("counts", {})
        except Exception:
            pass

    def _save(self) -> None:
        if not self._path:
            return
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"day": self._day, "counts": self._counts}))
            tmp.replace(self._path)
        except Exception:
            pass

    def _roll(self) -> None:
        today = self._today()
        if self._day != today:
            self._day, self._counts = today, {}

    def used(self, ip: str) -> Dict[str, int]:
        with self._lock:
            self._roll()
            return dict(self._counts.get(ip, {}))

    def try_take(self, ip: str, bucket: str, limit: int) -> bool:
        with self._lock:
            self._roll()
            mine = self._counts.setdefault(ip, {})
            if mine.get(bucket, 0) >= limit:
                return False
            mine[bucket] = mine.get(bucket, 0) + 1
            self._save()
            return True

    def give_back(self, ip: str, bucket: str) -> None:
        with self._lock:
            self._roll()
            mine = self._counts.get(ip, {})
            if mine.get(bucket, 0) > 0:
                mine[bucket] -= 1
                self._save()

    def reset(self) -> None:
        with self._lock:
            self._day, self._counts = "", {}


USAGE = _UsageStore()
_SEARCH_SLOTS: Optional[asyncio.Semaphore] = None


def search_slots() -> asyncio.Semaphore:
    global _SEARCH_SLOTS
    if _SEARCH_SLOTS is None:
        _SEARCH_SLOTS = asyncio.Semaphore(max(1, _env_int("DEMO_MAX_CONCURRENT", 2)))
    return _SEARCH_SLOTS


# ---------------------------------------------------------------------------
# Request policy
# ---------------------------------------------------------------------------

class DemoLimitError(Exception):
    def __init__(self, message: str, status: int = 429) -> None:
        super().__init__(message)
        self.status = status


def has_own_llm_key(cfg: ApiKeysConfig) -> bool:
    return bool((cfg.llm_api_key or "").strip())


def _is_private_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        return False


def apply_demo_config(cfg: ApiKeysConfig) -> Tuple[ApiKeysConfig, bool]:
    """
    Returns (config to use, uses_server_key). Visitors with their own key keep their settings (local providers
    excepted: the server cannot reach a visitor's machine and must not be pointed at its own network).
    Everyone else is moved onto the server's key and an allowed NVIDIA free model.
    """
    if cfg.llm_provider == "ollama" or (cfg.llm_base_url and _is_private_url(cfg.llm_base_url)):
        raise DemoLimitError("Local models (Ollama or a local server address) cannot be used on the public demo. "
                             "Install CiteLume on your own computer to use a local model.", status=400)
    if has_own_llm_key(cfg):
        return cfg, False
    if not server_key_configured():
        raise DemoLimitError("The demo's shared model key is not configured. Add your own LLM API key in "
                             "⚙ API Keys & Sources Setup to run a search.", status=503)
    model = cfg.llm_model if cfg.llm_model in allowed_models() else default_model()
    out = cfg.model_copy(deep=True)
    out.llm_provider = "openrouter"
    out.llm_api_key = os.environ["DEMO_OPENROUTER_API_KEY"].strip()
    out.llm_model = model
    out.llm_base_url = ""
    out.llm_fallback_models = [m for m in allowed_models() if m != model][:2]
    # Optional server-side literature-source keys, only when the visitor gave none
    for field, env in (("openalex_api_key", "DEMO_OPENALEX_API_KEY"), ("s2_api_key", "DEMO_S2_API_KEY"),
                       ("pubmed_api_key", "DEMO_PUBMED_API_KEY")):
        if not getattr(out, field) and os.environ.get(env):
            setattr(out, field, os.environ[env].strip())
    return out, True


def bucket_for(mode: str, uses_server_key: bool) -> str:
    if not uses_server_key:
        return "own_key"
    return "deep_research" if mode == "deep_research" else "standard"


def take_quota(ip: str, bucket: str) -> None:
    limit = demo_limits()[bucket]
    if USAGE.try_take(ip, bucket, limit):
        return
    if bucket == "own_key":
        raise DemoLimitError(f"Daily limit of {limit} searches reached for this address. Please try again tomorrow, "
                             "or install CiteLume locally for unlimited use.")
    label = "Deep Research review" if bucket == "deep_research" else "Standard search"
    raise DemoLimitError(
        f"The free demo allows {limit} {label}{'s' if limit != 1 else ''} per day per address, and you have used "
        f"{'it' if limit == 1 else 'them'}. Add your own LLM API key in ⚙ API Keys & Sources Setup to keep going, "
        "or try again tomorrow (limits reset at 00:00 UTC).")


def status_for(ip: str) -> Dict[str, object]:
    limits = demo_limits()
    used = USAGE.used(ip)
    return {
        "demo": True,
        "server_key_configured": server_key_configured(),
        "limits": {"standard": limits["standard"], "deep_research": limits["deep_research"]},
        "remaining": {b: max(0, limits[b] - used.get(b, 0)) for b in ("standard", "deep_research")},
        "allowed_models": allowed_models(),
        "default_model": default_model(),
        "resets": "00:00 UTC",
        # Shown back to the visitor so the per-address limit can be checked behind the host's proxy
        "address": ip,
    }
