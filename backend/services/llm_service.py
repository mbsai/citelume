import asyncio
import json
from contextvars import ContextVar
import re
import time
from typing import Optional, Dict, Any, List, Type, Callable, Tuple
import httpx
from pydantic import BaseModel, ValidationError
from models import (
    ApiKeysConfig,
    TestLlmRequest,
    ModelCapabilityProfile,
    TaskExecutionRecord,
    VerificationStatusType,
)


# Live metadata cache populated from provider model discovery endpoints (OpenRouter, Gemini, Ollama, etc.)
_LIVE_MODEL_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}

# Optional test hook for deterministic cross-model benchmarking
_TEST_LLM_RESPONDER: Optional[Callable[[str, str, ApiKeysConfig, int], Optional[str]]] = None
# Last provider error for the current asyncio task (ContextVar so concurrent searches/batches never overwrite it)
_LAST_LLM_ERROR: ContextVar[Optional[str]] = ContextVar("last_llm_error", default=None)

# Model that actually answered the last call (differs from the configured one when an OpenRouter fallback was used)
_LAST_LLM_MODEL_USED: ContextVar[Optional[str]] = ContextVar("last_llm_model_used", default=None)

# True when the last call stopped at its output token limit (the answer is cut off and unusable as JSON)
_LAST_LLM_TRUNCATED: ContextVar[bool] = ContextVar("last_llm_truncated", default=False)

# Rough token estimate (≈3.5 characters per token for English/scientific text; errs on the high side)
CHARS_PER_TOKEN = 3.5
# Tokens kept free in the context window for chat-template overhead and estimation error
CONTEXT_SAFETY_MARGIN = 512
# Local servers (Ollama) only respond once the whole answer is generated, which can take minutes on a laptop
LOCAL_LLM_TIMEOUT_S = 300.0
# Below this many answer tokens a structured task cannot produce anything useful
MIN_USEFUL_OUTPUT_TOKENS = 600


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


# Per-search provider health used by the circuit breaker. None outside a search (tests, connection checks).
_REQUEST_LLM_HEALTH: ContextVar[Optional[Dict[str, Any]]] = ContextVar("request_llm_health", default=None)
# Consecutive transient failures (rate limits, timeouts, 5xx) after which a search stops calling the provider
TRANSIENT_FAILURES_BEFORE_SKIP = 2

# Extra output tokens granted to reasoning models for hidden reasoning, on top of the answer budget
REASONING_TOKEN_HEADROOM = 8192

# OpenRouter accepts a short ordered list of models to fall back through
MAX_OPENROUTER_FALLBACKS = 3


def set_test_llm_responder(fn: Optional[Callable[[str, str, ApiKeysConfig, int], Optional[str]]]) -> None:
    global _TEST_LLM_RESPONDER
    _TEST_LLM_RESPONDER = fn


# Documented provider & model capability registry (used when live endpoint metadata has not been queried yet)
DOCUMENTED_PROVIDER_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "supports_native_json_schema": True,
        "supports_json_object_mode": True,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "context_window_tokens": 1000000,
        "max_output_tokens": 8192,
        "recommended_study_batch_size": 20,
        "provider_constraints": ["Supports native responseMimeType=application/json"],
        "observed_quality_tier": "tier_1_high_structured",
    },
    "openai": {
        "supports_native_json_schema": True,
        "supports_json_object_mode": True,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "context_window_tokens": 128000,
        "max_output_tokens": 4096,
        "recommended_study_batch_size": 15,
        "provider_constraints": ["Supports native response_format={type: json_object}"],
        "observed_quality_tier": "tier_1_high_structured",
    },
    "anthropic": {
        "supports_native_json_schema": False,
        "supports_json_object_mode": False,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "context_window_tokens": 200000,
        "max_output_tokens": 4096,
        "recommended_study_batch_size": 15,
        "provider_constraints": ["Uses strict system-prompt JSON schema enforcement"],
        "observed_quality_tier": "tier_1_high_structured",
    },
    "openrouter": {
        "supports_native_json_schema": False,
        "supports_json_object_mode": True,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "context_window_tokens": 64000,
        "max_output_tokens": 4096,
        "recommended_study_batch_size": 12,
        "provider_constraints": ["Aggregator routing across heterogeneous backends; verify JSON and numerical integrity"],
        "observed_quality_tier": "tier_2_standard",
    },
    "huggingface": {
        "supports_native_json_schema": False,
        "supports_json_object_mode": False,
        "supports_tool_calling": False,
        "supports_streaming": True,
        "context_window_tokens": 16384,
        "max_output_tokens": 2048,
        "recommended_study_batch_size": 6,
        "provider_constraints": ["2048 max_tokens cap on Serverless Inference API; requires smaller study batches"],
        "observed_quality_tier": "tier_2_standard",
    },
    "ollama": {
        "supports_native_json_schema": False,
        "supports_json_object_mode": True,
        "supports_tool_calling": False,
        "supports_streaming": True,
        "context_window_tokens": 8192,
        "max_output_tokens": 2048,
        "recommended_study_batch_size": 5,
        "provider_constraints": ["Local quantized weights; smaller context window requires chunked study batching"],
        "observed_quality_tier": "tier_3_constrained_local",
    },
    "custom": {
        "supports_native_json_schema": False,
        "supports_json_object_mode": True,
        "supports_tool_calling": False,
        "supports_streaming": True,
        "context_window_tokens": 32768,
        "max_output_tokens": 4096,
        "recommended_study_batch_size": 10,
        "provider_constraints": ["OpenAI-compatible custom endpoint; automatic fallback if response_format is rejected"],
        "observed_quality_tier": "tier_2_standard",
    },
}


# Name hints for model scale. Deep Research needs a large model; these drive the UI guidance and report disclaimer.
# Hints must be a separate part of the id ("gpt-4o-mini", not the "mini" inside "gemini")
_SMALL_MODEL_HINTS = re.compile(r"(?:^|[-_:/.\s])(mini|nano|lite|lightning|tiny|small|haiku|flash-lite)(?![a-z])", re.I)
_LARGE_MODEL_HINTS = re.compile(
    r"(gpt-4o(?!-mini)|gpt-4\.1(?!-(mini|nano))|gpt-5(?!-(mini|nano))|\bo1\b|\bo3(?!-mini)|claude.*(sonnet|opus)|"
    r"gemini.*pro|deepseek|ultra|super|mistral-large|command-r-plus|grok)",
    re.I,
)
# Below this many billion parameters a model is "small"; at or above LARGE_MODEL_MIN_B it is "large"
SMALL_MODEL_MAX_B = 30
LARGE_MODEL_MIN_B = 65


def classify_model_scale(provider: str, model_id: str) -> Tuple[str, str]:
    """
    Rough model scale from its id: ("small" | "medium" | "large" | "unknown", explanation). Parameter counts in the id
    ("qwen3.5:9b", "llama-3.3-70b") win over name hints; local models of unknown size are treated as small.
    """
    mid = (model_id or "").lower()
    sizes = [float(x) for x in re.findall(r"(?<![\d.a-z])(\d+(?:\.\d+)?)\s*b(?![a-z])", mid)]
    if sizes:
        size = max(sizes)
        if size < SMALL_MODEL_MAX_B or _SMALL_MODEL_HINTS.search(mid):
            return "small", f"about {size:g}B parameters"
        if size >= LARGE_MODEL_MIN_B:
            return "large", f"about {size:g}B parameters"
        return "medium", f"about {size:g}B parameters"
    if _SMALL_MODEL_HINTS.search(mid):
        return "small", "a compact model variant"
    if _LARGE_MODEL_HINTS.search(mid):
        return "large", "a frontier-scale model"
    if re.search(r"(?:^|[-_:/.])flash(?![a-z])", mid):
        return "medium", "a fast mid-sized model"
    if provider == "ollama":
        return "small", "a local model of unspecified size"
    return "unknown", "unknown size"


def resolve_model_capabilities(config: ApiKeysConfig) -> ModelCapabilityProfile:
    """
    Section 2 — Model Capability Abstraction Layer:
    Resolves capabilities for the selected provider and model using:
    1. Live provider metadata cached from `/api/provider-models` (`_LIVE_MODEL_METADATA_CACHE`) when available.
    2. Documented provider configuration (`DOCUMENTED_PROVIDER_CAPABILITIES`).
    3. Conservative defaults for unrecognized custom endpoints.
    """
    provider = (config.llm_provider or "gemini").lower()
    model_id = (config.llm_model or "").strip() or PROVIDER_DEFAULT_MODELS.get(provider, ["default"])[0]
    cache_key = f"{provider}:{model_id}"

    base_doc = DOCUMENTED_PROVIDER_CAPABILITIES.get(provider, DOCUMENTED_PROVIDER_CAPABILITIES["custom"]).copy()
    cap_source: str = "documented_config" if provider in DOCUMENTED_PROVIDER_CAPABILITIES else "conservative_default"

    if cache_key in _LIVE_MODEL_METADATA_CACHE:
        meta = _LIVE_MODEL_METADATA_CACHE[cache_key]
        cap_source = "provider_metadata"
        if meta.get("context_length"):
            base_doc["context_window_tokens"] = int(meta["context_length"])
        if meta.get("max_output_tokens"):
            base_doc["max_output_tokens"] = int(meta["max_output_tokens"])
        if "supports_json_object_mode" in meta:
            base_doc["supports_json_object_mode"] = bool(meta["supports_json_object_mode"])

    ctx = base_doc["context_window_tokens"]
    if ctx <= 8192:
        batch_size = 5
    elif ctx <= 32768:
        batch_size = 8
    elif ctx <= 64000:
        batch_size = 12
    else:
        batch_size = base_doc.get("recommended_study_batch_size", 15)

    return ModelCapabilityProfile(
        provider=provider,
        model_id=model_id,
        capability_source=cap_source,
        supports_native_json_schema=base_doc["supports_native_json_schema"],
        supports_json_object_mode=base_doc["supports_json_object_mode"],
        supports_tool_calling=base_doc["supports_tool_calling"],
        supports_streaming=base_doc["supports_streaming"],
        context_window_tokens=ctx,
        max_output_tokens=base_doc["max_output_tokens"],
        recommended_study_batch_size=batch_size,
        provider_constraints=list(base_doc["provider_constraints"]),
        observed_quality_tier=base_doc["observed_quality_tier"],
    )


# Section 8: Versioned Task Prompt Registry for Tasks A through J
TASK_PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "Task_A_B": {
        "version": "v2.2",
        "purpose": "Task A (Interpret research question) & Task B (Identify required evidence types & compile 5-API queries)",
    },
    "Task_C": {
        "version": "v2.2",
        "purpose": "Task C (Extract factual findings, study design, sample status, and limitations from study batches)",
    },
    "Task_DEFG": {
        "version": "v3.0-section1",
        "purpose": "Tasks D-G (Section 1): qualify claim wording for pre-built comparable evidence clusters without changing their evidential basis",
    },
    "Task_HI": {
        "version": "v2.2",
        "purpose": "Tasks H-I (Identify candidate research gaps across domain axes and formulate targeted F5 verification queries)",
    },
    "Task_J": {
        "version": "v3.0-section1",
        "purpose": "Task J (Section 1): write the A-E narrative strictly from verified claim records",
    },
}


PROVIDER_DEFAULT_MODELS: Dict[str, List[str]] = {
    "gemini": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    "openai": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
    "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"],
    "openrouter": [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "google/gemma-4-31b-it:free",
        "google/gemini-2.5-flash",
    ],
    "huggingface": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-72B-Instruct",
    ],
    "ollama": ["deepseek-r1:latest", "llama3.2:latest", "mistral:latest", "qwen2.5:latest"],
    "custom": ["deepseek-chat", "gpt-4o-mini", "llama-3.3-70b-versatile"],
}


def _is_free_model(model_id: str, model_name: str = "", pricing: Optional[Dict[str, Any]] = None, provider: str = "") -> bool:
    mid_low = (model_id or "").lower()
    name_low = (model_name or "").lower()
    if provider == "ollama":
        return True
    if mid_low.endswith(":free") or "(free)" in name_low or " free" in name_low:
        return True
    if isinstance(pricing, dict) and pricing:
        try:
            prompt_cost = float(pricing.get("prompt", 1))
            comp_cost = float(pricing.get("completion", 1))
            if prompt_cost == 0.0 and comp_cost == 0.0:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _sort_models_free_then_alpha(models_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts models with:
    1. Free models at the top (sorted alphabetically A -> Z by display name / id)
    2. All other models below (sorted alphabetically A -> Z by display name / id)
    """
    return sorted(
        models_list,
        key=lambda m: (
            0 if m.get("is_free") else 1,
            (m.get("clean_name") or m.get("name") or m.get("id") or "").lower(),
            (m.get("id") or "").lower(),
        ),
    )


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = candidate[start : end + 1]
        try:
            return json.loads(raw)
        except Exception:
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                return json.loads(cleaned)
            except Exception:
                return None
    return None


def _normalize_base_url(base_url: str, default_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    return url if url else default_url


def _ollama_num_ctx(system_prompt: str, user_prompt: str, max_output: int, model_context: int) -> int:
    needed = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + max_output + CONTEXT_SAFETY_MARGIN
    rounded = ((needed + 2047) // 2048) * 2048  # round up so Ollama can reuse the loaded model across calls
    return max(2048, min(rounded, model_context))


def _chat_completion_text(data: Any) -> Optional[str]:
    """
    Extracts assistant text from an OpenAI-compatible chat completion body. Records a diagnostic in
    `_LAST_LLM_ERROR` when the body carries an error (OpenRouter can return HTTP 200 with an `error`
    object) or when the model produced no text, e.g. a reasoning model that spent its whole token
    budget on hidden reasoning (`finish_reason=length`, empty `content`).
    """
    if not isinstance(data, dict):
        _LAST_LLM_ERROR.set("Provider returned a non-JSON-object response.")
        return None
    if data.get("error"):
        err = data["error"]
        _LAST_LLM_ERROR.set((err.get("message") if isinstance(err, dict) else None) or str(err))
        return None
    choices = data.get("choices") or []
    if not choices:
        _LAST_LLM_ERROR.set("Provider returned no completion choices.")
        return None
    choice = choices[0] or {}
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    text = (content or "").strip()
    if (choice.get("finish_reason") or "") == "length" and text:
        # A cut-off answer is broken JSON; report it so callers can shrink the job instead of resending it
        _LAST_LLM_TRUNCATED.set(True)
        _LAST_LLM_ERROR.set("Output was cut off at the token limit (finish_reason=length).")
        return None
    if not text:
        if (choice.get("finish_reason") or "") == "length":
            _LAST_LLM_TRUNCATED.set(True)
        finish = choice.get("finish_reason") or choice.get("native_finish_reason") or "unknown"
        if msg.get("reasoning") or msg.get("reasoning_content"):
            _LAST_LLM_ERROR.set(
                f"Model returned reasoning but no answer text (finish_reason={finish}); "
                "it likely exhausted its output token budget while reasoning."
            )
        else:
            _LAST_LLM_ERROR.set(f"Model returned an empty answer (finish_reason={finish}).")
        return None
    return text


async def _ensure_ollama_metadata(config: ApiKeysConfig) -> None:
    """Caches a local Ollama model's real context window (from /api/show) so budgets fit the actual model."""
    model = (config.llm_model or "").strip()
    cache_key = f"ollama:{model}"
    if not model or cache_key in _LIVE_MODEL_METADATA_CACHE:
        return
    base = _normalize_base_url(config.llm_base_url, "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{base}/api/show", json={"model": model})
            resp.raise_for_status()
            info = resp.json().get("model_info") or {}
        ctx = next((int(v) for k, v in info.items() if k.endswith(".context_length")), 0)
    except Exception:
        ctx = 0
    if ctx:
        _LIVE_MODEL_METADATA_CACHE[cache_key] = _ollama_metadata(ctx)


def _ollama_metadata(context_length: int) -> Dict[str, Any]:
    # Local generation is slow, so cap answers well below the window even for long-context models
    return {"context_length": context_length, "max_output_tokens": min(context_length // 2, 8192)}


_OPENROUTER_CATALOG_LOCK: Optional[asyncio.Lock] = None
_OPENROUTER_CATALOG_ATTEMPTED: set = set()


async def _ensure_openrouter_catalog(config: ApiKeysConfig) -> None:
    """
    Loads OpenRouter's model catalog once per server process. Reasoning support, context window and JSON mode come
    from it; without it a freshly started server would send reasoning models no reasoning settings, and they spend
    the whole token budget thinking and return no answer.
    """
    global _OPENROUTER_CATALOG_LOCK
    if config.llm_provider != "openrouter" or _TEST_LLM_RESPONDER is not None:
        return
    base = _normalize_base_url(config.llm_base_url, "https://openrouter.ai/api/v1")
    if base in _OPENROUTER_CATALOG_ATTEMPTED:
        return
    if _OPENROUTER_CATALOG_LOCK is None:
        _OPENROUTER_CATALOG_LOCK = asyncio.Lock()
    async with _OPENROUTER_CATALOG_LOCK:
        if base in _OPENROUTER_CATALOG_ATTEMPTED:
            return
        _OPENROUTER_CATALOG_ATTEMPTED.add(base)
        try:
            await fetch_provider_models("openrouter", (config.llm_api_key or "").strip(), config.llm_base_url)
        except Exception as exc:
            print(f"[OpenRouter Catalog Warning]: {exc}", flush=True)


_LOCAL_LLM_LOCKS: Dict[str, asyncio.Lock] = {}


def _local_llm_lock(config: ApiKeysConfig) -> asyncio.Lock:
    base = _normalize_base_url(config.llm_base_url, "http://localhost:11434")
    if base not in _LOCAL_LLM_LOCKS:
        _LOCAL_LLM_LOCKS[base] = asyncio.Lock()
    return _LOCAL_LLM_LOCKS[base]


def begin_llm_request_health() -> None:
    """Starts per-search provider health tracking so the circuit breaker can skip a dead provider."""
    _REQUEST_LLM_HEALTH.set({"transient_failures": 0, "skip_reason": None})


def _record_provider_failure(status_code: Optional[int], err_msg: str) -> None:
    health = _REQUEST_LLM_HEALTH.get()
    if health is None:
        return
    if status_code in (401, 402, 403):
        # Bad key, no credits, or forbidden: every later call in this search would fail the same way
        health["skip_reason"] = f"Skipped: the provider rejected an earlier call in this search ({err_msg})"
    elif status_code is None or status_code in (408, 429) or status_code >= 500:
        health["transient_failures"] += 1
        if health["transient_failures"] >= TRANSIENT_FAILURES_BEFORE_SKIP:
            health["skip_reason"] = (
                f"Skipped: the provider failed {health['transient_failures']} times in a row earlier in this "
                f"search (last error: {err_msg})"
            )


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    config: ApiKeysConfig,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    expect_json: bool = False,
    attempt_idx: int = 1,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Circuit-breaker wrapper around `_call_provider`: once the provider has failed hard (401/402/403) or
    repeatedly (rate limits, timeouts, 5xx) within the current search, later calls return immediately
    instead of each waiting on the same failure.
    """
    _LAST_LLM_MODEL_USED.set(None)
    await _ensure_openrouter_catalog(config)
    _LAST_LLM_TRUNCATED.set(False)
    health = _REQUEST_LLM_HEALTH.get()
    if health is not None and health.get("skip_reason") and _TEST_LLM_RESPONDER is None:
        _LAST_LLM_ERROR.set(health["skip_reason"])
        return None
    if config.llm_provider == "ollama":
        # One local GPU serves calls one at a time; queueing here keeps each call's timeout for its own generation
        async with _local_llm_lock(config):
            result = await _call_provider(
                system_prompt, user_prompt, config, temperature, max_tokens, expect_json, attempt_idx, json_schema
            )
    else:
        result = await _call_provider(
            system_prompt, user_prompt, config, temperature, max_tokens, expect_json, attempt_idx, json_schema
        )
    if result and health is not None:
        health["transient_failures"] = 0
    return result


async def _call_provider(
    system_prompt: str,
    user_prompt: str,
    config: ApiKeysConfig,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    expect_json: bool = False,
    attempt_idx: int = 1,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Calls the configured LLM provider with capability-aware structured output flags:
    - gemini, openai, anthropic, openrouter, huggingface
    - ollama (local LLM via http://localhost:11434 or custom base URL, no API key required)
    - custom (any OpenAI-compatible endpoint / aggregator such as Groq, Together, vLLM, LM Studio, DeepSeek, LiteLLM)
    """
    _LAST_LLM_ERROR.set(None)

    if _TEST_LLM_RESPONDER is not None:
        return _TEST_LLM_RESPONDER(system_prompt, user_prompt, config, attempt_idx)

    provider = config.llm_provider
    key = (config.llm_api_key or "").strip()
    if not key and provider == "huggingface" and config.hf_api_key:
        key = config.hf_api_key.strip()

    if provider not in ("ollama", "custom") and not key:
        return None
    if provider == "custom" and not key and not (config.llm_base_url or "").strip():
        return None

    cap = resolve_model_capabilities(config)
    effective_max_tokens = min(max_tokens, cap.max_output_tokens)
    model = (config.llm_model or "").strip()

    timeout_s = LOCAL_LLM_TIMEOUT_S if provider in ("ollama", "custom") else 55.0
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
        try:
            if provider == "gemini":
                model_name = model or "gemini-2.5-flash"
                if model_name.startswith("models/"):
                    model_name = model_name.replace("models/", "", 1)
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                gen_cfg: Dict[str, Any] = {
                    "temperature": temperature,
                    "maxOutputTokens": effective_max_tokens,
                }
                if expect_json and cap.supports_json_object_mode:
                    gen_cfg["responseMimeType"] = "application/json"
                payload = {
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": gen_cfg,
                }
                resp = await client.post(url, params={"key": key}, json=payload)
                if resp.status_code == 400 and "responseMimeType" in gen_cfg:
                    gen_cfg.pop("responseMimeType", None)
                    resp = await client.post(url, params={"key": key}, json=payload)
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates") or []
                if candidates:
                    if candidates[0].get("finishReason") == "MAX_TOKENS":
                        _LAST_LLM_TRUNCATED.set(True)
                        _LAST_LLM_ERROR.set("Output was cut off at the token limit (finishReason=MAX_TOKENS).")
                        return None
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    return "".join(p.get("text", "") for p in parts).strip()

            elif provider == "openai":
                model_name = model or "gpt-4o-mini"
                base = _normalize_base_url(config.llm_base_url, "https://api.openai.com/v1")
                url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                }
                if expect_json and cap.supports_json_object_mode:
                    payload["response_format"] = {"type": "json_object"}
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format", None)
                    resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return _chat_completion_text(resp.json())

            elif provider == "anthropic":
                model_name = model or "claude-3-5-sonnet-latest"
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model_name,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                }
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if data.get("stop_reason") == "max_tokens":
                    _LAST_LLM_TRUNCATED.set(True)
                    _LAST_LLM_ERROR.set("Output was cut off at the token limit (stop_reason=max_tokens).")
                    return None
                content_blocks = data.get("content") or []
                return "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text").strip()

            elif provider == "openrouter":
                model_name = model or "google/gemini-2.5-flash"
                base = _normalize_base_url(config.llm_base_url, "https://openrouter.ai/api/v1")
                url = f"{base}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "CiteLume",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                }
                fallbacks = [
                    m.strip() for m in (config.llm_fallback_models or []) if m and m.strip() and m.strip() != model_name
                ]
                fallbacks = list(dict.fromkeys(fallbacks))[:MAX_OPENROUTER_FALLBACKS]
                if expect_json and cap.supports_json_object_mode:
                    payload["response_format"] = {"type": "json_object"}
                if any(
                    _LIVE_MODEL_METADATA_CACHE.get(f"openrouter:{m}", {}).get("supports_reasoning")
                    for m in [model_name] + fallbacks
                ):
                    # Hidden reasoning tokens count against max_tokens; on long task prompts they can use the whole
                    # budget and leave an empty answer (finish_reason=length). These are structured extraction tasks,
                    # so turn reasoning off (≈3x faster), and keep headroom for models that reason regardless.
                    payload["reasoning"] = {"enabled": False}
                    payload["max_tokens"] = min(effective_max_tokens + REASONING_TOKEN_HEADROOM, cap.max_output_tokens)
                # Try the primary model alone first: backups can be much slower, so only use them on a real failure
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 429 or resp.status_code in (404, 408) or resp.status_code >= 500:
                    if resp.status_code == 429:
                        # Free models are often rate-limited in short bursts upstream; a short wait usually clears it
                        try:
                            wait_s = min(float(resp.headers.get("retry-after") or 6), 15.0)
                        except ValueError:
                            wait_s = 6.0
                        await asyncio.sleep(wait_s)
                    if fallbacks:
                        # OpenRouter routing: retries the primary, then each backup in order if it is still failing
                        payload["models"] = [model_name] + fallbacks
                    resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (400, 422) and ("response_format" in payload or "reasoning" in payload):
                    payload.pop("response_format", None)
                    payload.pop("reasoning", None)
                    resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = _chat_completion_text(data)
                if text is None and _LAST_LLM_TRUNCATED.get() and "reasoning" not in payload:
                    # A reasoning model the catalog did not describe used the whole budget thinking: retry once with
                    # reasoning off and room for the answer
                    payload["reasoning"] = {"enabled": False}
                    payload["max_tokens"] = min(effective_max_tokens + REASONING_TOKEN_HEADROOM, cap.max_output_tokens)
                    _LAST_LLM_TRUNCATED.set(False)
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = _chat_completion_text(data)
                if isinstance(data, dict) and data.get("model"):
                    _LAST_LLM_MODEL_USED.set(data["model"])
                return text

            elif provider == "huggingface":
                model_name = model or "Qwen/Qwen2.5-72B-Instruct"
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": min(effective_max_tokens, 2048),
                }
                for url in [
                    "https://router.huggingface.co/v1/chat/completions",
                    f"https://api-inference.huggingface.co/models/{model_name}/v1/chat/completions",
                ]:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        return _chat_completion_text(resp.json())
                resp.raise_for_status()

            elif provider == "ollama":
                model_name = model or "llama3.2:latest"
                base = _normalize_base_url(config.llm_base_url, "http://localhost:11434")
                headers = {"Content-Type": "application/json"}
                if key:
                    headers["Authorization"] = f"Bearer {key}"

                ollama_chat_url = f"{base}/api/chat"
                ollama_payload: Dict[str, Any] = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    # Without num_ctx Ollama uses its small default window (2k-4k) and silently drops the start of
                    # longer prompts; size the window to this prompt + answer (rounded up, capped at the model's limit).
                    "options": {
                        "temperature": temperature,
                        "num_ctx": _ollama_num_ctx(system_prompt, user_prompt, effective_max_tokens, cap.context_window_tokens),
                        "num_predict": effective_max_tokens,
                    },
                    # Structured extraction does not need hidden reasoning; thinking models are much faster without it
                    "think": False,
                }
                if expect_json:
                    # A JSON schema constrains decoding so even small models can only emit valid, schema-shaped JSON
                    ollama_payload["format"] = json_schema or "json"
                resp = await client.post(ollama_chat_url, headers=headers, json=ollama_payload)
                if resp.status_code == 400 and "think" in ollama_payload:
                    # Older Ollama versions or models without a thinking switch reject the flag
                    ollama_payload.pop("think", None)
                    resp = await client.post(ollama_chat_url, headers=headers, json=ollama_payload)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("done_reason") == "length":
                        _LAST_LLM_TRUNCATED.set(True)
                        _LAST_LLM_ERROR.set("Output was cut off at the token limit (done_reason=length).")
                        return None
                    msg = (data.get("message") or {}).get("content")
                    if msg:
                        return msg.strip()

                v1_url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
                v1_payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                resp2 = await client.post(v1_url, headers=headers, json=v1_payload)
                resp2.raise_for_status()
                return _chat_completion_text(resp2.json())

            elif provider == "custom":
                model_name = model or "gpt-4o-mini"
                base = _normalize_base_url(config.llm_base_url, "https://api.openai.com/v1")
                headers = {"Content-Type": "application/json"}
                if key:
                    headers["Authorization"] = f"Bearer {key}"

                candidate_urls = []
                if base.endswith("/chat/completions"):
                    candidate_urls.append(base)
                elif base.endswith("/v1") or base.endswith("/api/v1") or base.endswith("/openai/v1"):
                    candidate_urls.append(f"{base}/chat/completions")
                else:
                    candidate_urls.append(f"{base}/v1/chat/completions")
                    candidate_urls.append(f"{base}/chat/completions")

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                }
                last_resp = None
                for url in candidate_urls:
                    last_resp = await client.post(url, headers=headers, json=payload)
                    if last_resp.status_code == 200:
                        return _chat_completion_text(last_resp.json())
                if last_resp is not None:
                    last_resp.raise_for_status()

        except httpx.HTTPStatusError as exc:
            err_body = ""
            try:
                err_data = exc.response.json()
                if isinstance(err_data, dict):
                    err_obj = err_data.get("error") or {}
                    err_body = err_obj.get("message") or str(err_data)
                    # OpenRouter wraps upstream failures as "Provider returned error" with details in metadata.raw
                    raw_detail = (err_obj.get("metadata") or {}).get("raw")
                    if raw_detail:
                        err_body = f"{err_body} — {str(raw_detail)[:220]}"
            except Exception:
                err_body = exc.response.text[:220]
            err_msg = f"HTTP {exc.response.status_code}: {err_body}" if err_body else f"HTTP {exc.response.status_code}"
            _LAST_LLM_ERROR.set(err_msg)
            _record_provider_failure(exc.response.status_code, err_msg)
            print(f"[LLM Router Warning ({provider} / {model})]: {err_msg}", flush=True)
            return None
        except Exception as exc:
            # str() of httpx timeouts is often empty, so keep the exception type
            _LAST_LLM_ERROR.set(f"{type(exc).__name__}: {exc}".rstrip(": "))
            _record_provider_failure(None, _LAST_LLM_ERROR.get() or "")
            print(f"[LLM Router Warning ({provider} / {model})]: {_LAST_LLM_ERROR.get()}", flush=True)
            return None

    return None


async def execute_validated_llm_task(
    task_id: str,
    task_name: str,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[BaseModel],
    config: ApiKeysConfig,
    custom_validator: Optional[Callable[[Any], List[str]]] = None,
    max_retries: int = 2,
    temperature: float = 0.15,
    max_tokens: int = 3000,
    grow_on_truncation: bool = True,
) -> Tuple[Optional[Any], TaskExecutionRecord]:
    """
    Section 3, 6 & 7 — Model-Independent Structured Task Runner with Controlled Error Correction:
    1. Invokes the user-selected LLM with capability-appropriate JSON instructions.
    2. Parses the JSON block and validates against `response_model` (Pydantic schema).
    3. Runs `custom_validator(validated_obj)` for source-fact, citation-bound, numerical, and causal-inference checks.
    4. If validation fails, returns the specific validation errors and original source evidence to the model
       for bounded corrective retries (`max_retries` <= 2).
    5. Never silently switches to another paid LLM provider; if all retries fail, returns `(None, record)`
       so the caller can degrade gracefully to deterministic source-grounded extraction/synthesis with
       transparent disclosure (`failed_verification` / `passed_with_disclosed_limitations`).
    """
    reg_info = TASK_PROMPT_REGISTRY.get(task_id, {"version": "v2.2", "purpose": task_name})
    cap = resolve_model_capabilities(config)
    model_label = f"{cap.provider}:{cap.model_id}"

    has_llm = (
        _TEST_LLM_RESPONDER is not None
        or bool((config.llm_api_key or "").strip())
        or (config.llm_provider == "huggingface" and bool((config.hf_api_key or "").strip()))
        or config.llm_provider in ("ollama", "custom")
    )

    if not has_llm:
        return None, TaskExecutionRecord(
            task_id=task_id,
            task_name=task_name,
            prompt_version=reg_info["version"],
            model_used="deterministic_offline_engine",
            attempts_used=0,
            validation_passed=True,
            verification_status="passed_with_disclosed_limitations",
            validation_errors=["No external LLM configured; executed deterministic source-grounded pipeline."],
            degraded_to_deterministic=True,
        )

    if config.llm_provider == "ollama" and _TEST_LLM_RESPONDER is None:
        await _ensure_ollama_metadata(config)
        cap = resolve_model_capabilities(config)
    elif config.llm_provider == "openrouter" and _TEST_LLM_RESPONDER is None:
        await _ensure_openrouter_catalog(config)
        cap = resolve_model_capabilities(config)

    # Fit the job to the model: the prompt plus a useful answer must fit in the context window. Small models get
    # a smaller answer budget; if even a minimal answer cannot fit, skip the call and keep the rule-based output.
    prompt_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    room = cap.context_window_tokens - prompt_tokens - CONTEXT_SAFETY_MARGIN
    if room < min(max_tokens, MIN_USEFUL_OUTPUT_TOKENS):
        return None, TaskExecutionRecord(
            task_id=task_id,
            task_name=task_name,
            prompt_version=reg_info["version"],
            model_used=model_label,
            attempts_used=0,
            validation_passed=False,
            verification_status="passed_with_disclosed_limitations",
            validation_errors=[
                f"Skipped: this step's prompt (~{prompt_tokens} tokens) does not leave room for an answer in "
                f"{cap.model_id}'s {cap.context_window_tokens}-token context window; rule-based output used."
            ],
            degraded_to_deterministic=True,
        )
    max_tokens = min(max_tokens, room)
    json_schema = response_model.model_json_schema()

    current_user_prompt = user_prompt
    all_errors: List[str] = []
    last_candidate: Optional[Dict[str, Any]] = None
    truncated = False
    total_attempts = 1 + max(0, min(max_retries, 2))

    for attempt in range(1, total_attempts + 1):
        raw_text = await call_llm(
            system_prompt=system_prompt,
            user_prompt=current_user_prompt,
            config=config,
            temperature=temperature if attempt == 1 else 0.05,
            max_tokens=max_tokens,
            expect_json=True,
            attempt_idx=attempt,
            json_schema=json_schema,
        )
        if not raw_text:
            truncated = _LAST_LLM_TRUNCATED.get()
            reason = f" Reason: {_LAST_LLM_ERROR.get()}" if _LAST_LLM_ERROR.get() else ""
            err_msg = f"Attempt {attempt}: Empty or unreachable response from {model_label}.{reason}"
            all_errors.append(err_msg)
            # A cut-off answer fails the same way if resent unchanged. Tasks that cannot be split get one retry with
            # double the answer budget while it still fits the context window; splittable tasks return to the caller.
            grown = min(max_tokens * 2, room, cap.max_output_tokens)
            if truncated and grow_on_truncation and grown > max_tokens and attempt < total_attempts:
                max_tokens = grown
                grow_on_truncation = False
                truncated = False
                continue
            break

        parsed_dict = extract_json_block(raw_text)
        if parsed_dict is None:
            err_msg = f"Attempt {attempt}: Output did not contain a valid JSON object."
            all_errors.append(err_msg)
            current_user_prompt = (
                f"{user_prompt}\n\n"
                f"=== CONTROLLED ERROR CORRECTION (RETRY {attempt}) ===\n"
                f"Your previous output failed JSON parsing: {err_msg}\n"
                f"Respond ONLY with a valid JSON object matching the required schema. Do not include markdown prose outside JSON."
            )
            continue

        try:
            validated_obj = response_model.model_validate(parsed_dict)
        except ValidationError as ve:
            short_ve = "; ".join(f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in ve.errors()[:4])
            err_msg = f"Attempt {attempt} Schema Error: {short_ve}"
            all_errors.append(err_msg)
            current_user_prompt = (
                f"{user_prompt}\n\n"
                f"=== CONTROLLED ERROR CORRECTION (RETRY {attempt}) ===\n"
                f"Your previous JSON failed schema validation: {short_ve}\n"
                f"Fix the exact fields above using ONLY the provided source records and return valid JSON."
            )
            continue

        if custom_validator is not None:
            semantic_errors = custom_validator(validated_obj)
            if semantic_errors:
                last_candidate = validated_obj.model_dump()
                err_joined = "; ".join(semantic_errors[:4])
                all_errors.append(f"Attempt {attempt} Source/Quality Gate Error: {err_joined}")
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"=== CONTROLLED ERROR CORRECTION (RETRY {attempt}) ===\n"
                    f"Your previous output violated scientific quality gates:\n"
                    + "\n".join(f"- {se}" for se in semantic_errors[:5])
                    + "\nCorrect these errors strictly using the provided study abstracts/metadata and return valid JSON."
                )
                continue

        v_status: VerificationStatusType = (
            "passed_validation" if attempt == 1 else "passed_with_disclosed_limitations"
        )
        answered_by = _LAST_LLM_MODEL_USED.get()
        # OpenRouter may report the id without its ":free"-style variant suffix, so compare base ids
        if answered_by and answered_by.split(":")[0] == cap.model_id.split(":")[0]:
            answered_by = None
        if answered_by:
            # Disclose that an OpenRouter fallback model answered instead of the configured one
            all_errors.append(f"Primary model {cap.model_id} unavailable; answered by fallback model {answered_by}.")
        return validated_obj, TaskExecutionRecord(
            task_id=task_id,
            task_name=task_name,
            prompt_version=reg_info["version"],
            model_used=f"{cap.provider}:{answered_by}" if answered_by else model_label,
            attempts_used=attempt,
            validation_passed=True,
            verification_status=v_status,
            validation_errors=all_errors,
            degraded_to_deterministic=False,
        )

    return None, TaskExecutionRecord(
        task_id=task_id,
        task_name=task_name,
        prompt_version=reg_info["version"],
        model_used=model_label,
        attempts_used=len(all_errors) or 1,
        validation_passed=False,
        verification_status="failed_verification",
        validation_errors=all_errors,
        degraded_to_deterministic=True,
        output_truncated=truncated,
        last_candidate=last_candidate,
    )


def detect_provider_from_credentials(
    current_provider: str,
    api_key: str,
    base_url: str,
) -> Dict[str, str]:
    k = (api_key or "").strip()
    b = (base_url or "").strip().lower()

    valid_providers = {"gemini", "openai", "anthropic", "openrouter", "huggingface", "ollama", "custom"}
    if current_provider in valid_providers:
        detected_base = base_url.strip() if base_url else ""
        return {"provider": current_provider, "base_url": detected_base}

    detected = current_provider
    detected_base = base_url.strip() if base_url else ""

    if k.startswith("sk-or-v1-") or "openrouter.ai" in b:
        detected = "openrouter"
    elif k.startswith("sk-ant-") or "anthropic.com" in b:
        detected = "anthropic"
    elif k.startswith("hf_") or "huggingface.co" in b:
        detected = "huggingface"
    elif k.startswith("AIza"):
        detected = "gemini"
    elif "11434" in b or "ollama" in b:
        detected = "ollama"
    elif k.startswith("gsk_") and not detected_base:
        detected = "custom"
        detected_base = "https://api.groq.com/openai/v1"
    elif "together.xyz" in b or "fireworks.ai" in b or "deepseek.com" in b or "localhost:1234" in b:
        detected = "custom"

    return {"provider": detected, "base_url": detected_base}


async def fetch_provider_models(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    hf_api_key: str = "",
) -> Dict[str, Any]:
    """
    Dynamically queries the selected provider or aggregator (OpenRouter, Ollama, Custom OpenAI-compatible,
    OpenAI, Gemini, Anthropic, Hugging Face) to refresh its available models list.
    Models are sorted with FREE models at the top (A-Z), followed by all other models (A-Z).
    """
    valid_providers = {"gemini", "openai", "anthropic", "openrouter", "huggingface", "ollama", "custom"}
    if provider in valid_providers:
        effective_provider = provider
        effective_base = base_url.strip() if base_url else ""
    else:
        rec = detect_provider_from_credentials(provider, api_key, base_url)
        effective_provider = rec["provider"]
        effective_base = rec["base_url"]
    key = (api_key or "").strip()
    if not key and effective_provider == "huggingface" and hf_api_key:
        key = hf_api_key.strip()

    models_list: List[Dict[str, Any]] = []
    source_label = effective_provider
    live_fetched = False
    error_detail = ""

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        try:
            if effective_provider == "ollama":
                base = _normalize_base_url(effective_base, "http://localhost:11434")
                headers = {}
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                resp = await client.get(f"{base}/api/tags", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("models") or []:
                        name = m.get("name") or m.get("model")
                        if name:
                            details = m.get("details") or {}
                            if details.get("context_length"):
                                _LIVE_MODEL_METADATA_CACHE[f"ollama:{name}"] = _ollama_metadata(int(details["context_length"]))
                            param_size = details.get("parameter_size") or ""
                            quant = details.get("quantization_level") or ""
                            disp = f"{name} ({param_size} {quant})".strip(" ()")
                            models_list.append(
                                {
                                    "id": name,
                                    "clean_name": name,
                                    "name": f"★ [FREE Local] {disp}",
                                    "provider": "ollama",
                                    "is_free": True,
                                }
                            )
                    live_fetched = True
                    source_label = f"Ollama ({base})"
                else:
                    v1_url = f"{base}/v1/models" if not base.endswith("/v1") else f"{base}/models"
                    resp2 = await client.get(v1_url, headers=headers)
                    resp2.raise_for_status()
                    data2 = resp2.json()
                    for m in data2.get("data") or []:
                        mid = m.get("id")
                        if mid:
                            models_list.append(
                                {
                                    "id": mid,
                                    "clean_name": mid,
                                    "name": f"★ [FREE Local] {mid}",
                                    "provider": "ollama",
                                    "is_free": True,
                                }
                            )
                    live_fetched = True
                    source_label = f"Ollama ({base})"

            elif effective_provider == "openrouter":
                base = _normalize_base_url(effective_base, "https://openrouter.ai/api/v1")
                headers = {"User-Agent": "CiteLume/1.0"}
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                resp = await client.get(f"{base}/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("data") or []:
                    mid = m.get("id")
                    if not mid:
                        continue
                    ctx = m.get("context_length")
                    top_prov = m.get("top_provider") or {}
                    max_comp = top_prov.get("max_completion_tokens")
                    supp_params = m.get("supported_parameters") or []
                    _LIVE_MODEL_METADATA_CACHE[f"openrouter:{mid}"] = {
                        "context_length": ctx or 64000,
                        "max_output_tokens": max_comp or 4096,
                        "supports_json_object_mode": ("response_format" in supp_params or "structured_outputs" in supp_params) if supp_params else True,
                        "supports_reasoning": "reasoning" in supp_params,
                    }
                    raw_label = m.get("name") or mid
                    pricing = m.get("pricing") or {}
                    is_free = _is_free_model(mid, raw_label, pricing, "openrouter")
                    ctx_str = f" [{ctx // 1000}k ctx]" if isinstance(ctx, int) and ctx >= 1000 else ""
                    prefix = "★ [FREE] " if is_free else ""
                    models_list.append(
                        {
                            "id": mid,
                            "clean_name": raw_label,
                            "name": f"{prefix}{raw_label} ({mid}){ctx_str}",
                            "provider": "openrouter",
                            "is_free": is_free,
                        }
                    )
                live_fetched = True
                source_label = "OpenRouter Live Catalog"

            elif effective_provider == "custom":
                base = _normalize_base_url(effective_base, "http://localhost:1234/v1")
                headers = {"User-Agent": "CiteLume/1.0"}
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                candidate_urls = []
                if base.endswith("/models"):
                    candidate_urls.append(base)
                elif base.endswith("/v1") or base.endswith("/api/v1") or base.endswith("/openai/v1"):
                    candidate_urls.append(f"{base}/models")
                else:
                    candidate_urls.append(f"{base}/v1/models")
                    candidate_urls.append(f"{base}/models")
                    candidate_urls.append(f"{base}/api/tags")

                for url in candidate_urls:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data.get("data") or data.get("models") or (data if isinstance(data, list) else [])
                        for item in items:
                            if isinstance(item, dict):
                                mid = item.get("id") or item.get("name") or item.get("model")
                                if mid:
                                    raw_label = item.get("name") or mid
                                    is_free = _is_free_model(mid, raw_label, item.get("pricing"), "custom")
                                    prefix = "★ [FREE] " if is_free else ""
                                    models_list.append(
                                        {
                                            "id": mid,
                                            "clean_name": raw_label,
                                            "name": f"{prefix}{raw_label}",
                                            "provider": "custom",
                                            "is_free": is_free,
                                        }
                                    )
                            elif isinstance(item, str):
                                is_free = _is_free_model(item, item, None, "custom")
                                models_list.append(
                                    {
                                        "id": item,
                                        "clean_name": item,
                                        "name": f"★ [FREE] {item}" if is_free else item,
                                        "provider": "custom",
                                        "is_free": is_free,
                                    }
                                )
                        live_fetched = True
                        source_label = f"Custom Provider ({base})"
                        break

            elif effective_provider == "openai" and key:
                base = _normalize_base_url(effective_base, "https://api.openai.com/v1")
                url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
                resp = await client.get(url, headers={"Authorization": f"Bearer {key}"})
                resp.raise_for_status()
                data = resp.json()
                raw_ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
                chat_ids = [
                    mid for mid in raw_ids
                    if any(p in mid for p in ("gpt-", "o1", "o3", "o4", "chatgpt"))
                ] or raw_ids
                for mid in chat_ids:
                    models_list.append(
                        {
                            "id": mid,
                            "clean_name": mid,
                            "name": mid,
                            "provider": "openai",
                            "is_free": False,
                        }
                    )
                live_fetched = True
                source_label = "OpenAI API"

            elif effective_provider == "gemini" and key:
                resp = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": key},
                )
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models") or []:
                    methods = m.get("supportedGenerationMethods") or []
                    if "generateContent" in methods:
                        raw_name = (m.get("name") or "").replace("models/", "", 1)
                        disp = m.get("displayName") or raw_name
                        if raw_name:
                            models_list.append(
                                {
                                    "id": raw_name,
                                    "clean_name": disp,
                                    "name": f"{disp} ({raw_name})",
                                    "provider": "gemini",
                                    "is_free": "flash" in raw_name.lower(),
                                }
                            )
                live_fetched = True
                source_label = "Google Gemini API"

            elif effective_provider == "anthropic" and key:
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                )
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("data") or []:
                    mid = m.get("id")
                    if mid:
                        disp = m.get("display_name") or mid
                        models_list.append(
                            {
                                "id": mid,
                                "clean_name": disp,
                                "name": f"{disp} ({mid})",
                                "provider": "anthropic",
                                "is_free": False,
                            }
                        )
                live_fetched = True
                source_label = "Anthropic API"

            elif effective_provider == "huggingface":
                hf_models = await fetch_available_hf_llm_models(hf_api_key=key)
                for m in hf_models:
                    models_list.append(
                        {
                            "id": m["id"],
                            "clean_name": m["id"],
                            "name": f"★ [FREE HF Tier] {m['id']} ({m.get('downloads', 0):,} dl)",
                            "provider": "huggingface",
                            "is_free": True,
                        }
                    )
                live_fetched = True
                source_label = "Hugging Face Hub"

        except Exception as exc:
            error_detail = str(exc)
            print(f"[Provider Models Refresh Warning ({effective_provider})]: {exc}")

    if not models_list:
        defaults = PROVIDER_DEFAULT_MODELS.get(effective_provider, PROVIDER_DEFAULT_MODELS["gemini"])
        for m in defaults:
            is_free = _is_free_model(m, m, None, effective_provider)
            models_list.append(
                {
                    "id": m,
                    "clean_name": m,
                    "name": f"{'★ [FREE] ' if is_free else ''}{m} (Preset)",
                    "provider": effective_provider,
                    "is_free": is_free,
                }
            )

    sorted_models = _sort_models_free_then_alpha(models_list)
    free_count = sum(1 for m in sorted_models if m.get("is_free"))

    return {
        "detected_provider": effective_provider,
        "detected_base_url": effective_base,
        "source_label": source_label,
        "live_fetched": live_fetched,
        "count": len(sorted_models),
        "free_count": free_count,
        "models": sorted_models,
        "warning": error_detail,
    }


async def test_llm_connection(req: TestLlmRequest) -> Dict[str, Any]:
    """
    Tests the configured LLM provider, API key, base URL, and selected model with a lightweight
    verification prompt and returns success status, latency, and response preview.
    """
    t0 = time.perf_counter()
    cfg = ApiKeysConfig(
        llm_provider=req.llm_provider,  # type: ignore
        llm_api_key=req.llm_api_key,
        llm_model=req.llm_model or PROVIDER_DEFAULT_MODELS.get(req.llm_provider, ["gemini-2.5-flash"])[0],
        llm_base_url=req.llm_base_url,
        hf_api_key=req.hf_api_key,
    )

    if cfg.llm_provider not in ("ollama", "custom") and not (cfg.llm_api_key or cfg.hf_api_key).strip():
        return {
            "success": False,
            "latency_ms": 0,
            "message": f"Please enter a valid API key for {cfg.llm_provider.upper()} before testing.",
        }

    reply = await call_llm(
        system_prompt="You are a diagnostic connectivity check.",
        user_prompt="Reply with the single word: CONNECTED",
        config=cfg,
        temperature=0.0,
        # Reasoning models (e.g. NVIDIA Nemotron, DeepSeek R1) spend tokens thinking before answering;
        # a tiny budget makes them return empty content and look like a failed connection.
        max_tokens=512,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if reply:
        snippet = reply.strip().replace("\n", " ")[:60]
        return {
            "success": True,
            "latency_ms": elapsed_ms,
            "message": f"Verified {cfg.llm_provider.upper()} ({cfg.llm_model}) in {elapsed_ms}ms — Response: \"{snippet}\"",
        }
    diagnostic = f": {_LAST_LLM_ERROR.get()}" if _LAST_LLM_ERROR.get() else ". Check your API key, model availability, or endpoint URL."
    return {
        "success": False,
        "latency_ms": elapsed_ms,
        "message": f"Connection failed for {cfg.llm_provider.upper()} ({cfg.llm_model}){diagnostic}",
    }


async def test_source_api_key(source: str, api_key: str = "") -> Dict[str, Any]:
    """
    Individually tests the API key and connectivity for PubMed, Semantic Scholar, arXiv, bioRxiv, or Hugging Face Hub.
    """
    t0 = time.perf_counter()
    key = (api_key or "").strip()

    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        try:
            if source == "pubmed":
                params = {"db": "pubmed", "term": "crispr", "retmax": "1", "retmode": "json"}
                if key:
                    params["api_key"] = key
                resp = await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
                elapsed = int((time.perf_counter() - t0) * 1000)
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:
                        return {"success": False, "latency_ms": elapsed, "message": f"NCBI Error: {data['error']}"}
                    mode_lbl = "Authenticated API Key (10 req/s)" if key else "Public Access (3 req/s)"
                    return {"success": True, "latency_ms": elapsed, "message": f"PubMed verified ({mode_lbl}) in {elapsed}ms"}
                return {"success": False, "latency_ms": elapsed, "message": f"PubMed returned HTTP {resp.status_code} (invalid NCBI API key)."}

            elif source == "s2":
                headers = {"User-Agent": "CiteLume/1.0"}
                if key:
                    headers["x-api-key"] = key
                resp = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": "transformer", "limit": "1"},
                    headers=headers,
                )
                elapsed = int((time.perf_counter() - t0) * 1000)
                if resp.status_code == 200:
                    mode_lbl = "Authenticated S2 Key" if key else "Public Endpoint"
                    return {"success": True, "latency_ms": elapsed, "message": f"Semantic Scholar verified ({mode_lbl}) in {elapsed}ms"}
                elif resp.status_code in (401, 403) and key:
                    return {"success": False, "latency_ms": elapsed, "message": f"Invalid Semantic Scholar API key (HTTP {resp.status_code})."}
                elif resp.status_code == 429 and not key:
                    return {
                        "success": True,
                        "latency_ms": elapsed,
                        "message": f"Public S2 rate-limited (HTTP 429) — automatic OpenAlex graph fallback is active & ready ({elapsed}ms).",
                    }
                return {"success": False, "latency_ms": elapsed, "message": f"Semantic Scholar returned HTTP {resp.status_code}"}

            elif source == "arxiv":
                resp = await client.get(
                    "https://export.arxiv.org/api/query",
                    params={"search_query": "all:protein", "max_results": "1"},
                )
                elapsed = int((time.perf_counter() - t0) * 1000)
                if resp.status_code == 200:
                    return {"success": True, "latency_ms": elapsed, "message": f"arXiv API verified & saved ({elapsed}ms)"}
                return {"success": False, "latency_ms": elapsed, "message": f"arXiv API returned HTTP {resp.status_code}"}

            elif source == "biorxiv":
                resp = await client.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params={"query": "(crispr) AND (SRC:PPR)", "pageSize": "1", "format": "json"},
                )
                elapsed = int((time.perf_counter() - t0) * 1000)
                if resp.status_code == 200:
                    return {"success": True, "latency_ms": elapsed, "message": f"bioRxiv / medRxiv preprint index verified & saved ({elapsed}ms)"}
                return {"success": False, "latency_ms": elapsed, "message": f"bioRxiv endpoint returned HTTP {resp.status_code}"}

            elif source == "huggingface":
                headers = {"User-Agent": "CiteLume/1.0"}
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                    resp = await client.get("https://huggingface.co/api/whoami-v2", headers=headers)
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    if resp.status_code == 200:
                        user_info = resp.json()
                        uname = user_info.get("name") or "Authenticated User"
                        return {"success": True, "latency_ms": elapsed, "message": f"Hugging Face Token verified for '{uname}' ({elapsed}ms)"}
                    return {"success": False, "latency_ms": elapsed, "message": f"Invalid Hugging Face Token (HTTP {resp.status_code})"}
                else:
                    resp = await client.get("https://huggingface.co/api/models", params={"limit": "1"}, headers=headers)
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    if resp.status_code == 200:
                        return {"success": True, "latency_ms": elapsed, "message": f"Hugging Face Hub public API verified ({elapsed}ms)"}
                    return {"success": False, "latency_ms": elapsed, "message": f"Hugging Face Hub returned HTTP {resp.status_code}"}

        except Exception as exc:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {"success": False, "latency_ms": elapsed, "message": f"Connection error: {exc}"}

    return {"success": False, "latency_ms": 0, "message": f"Unknown source: {source}"}


async def fetch_available_hf_llm_models(hf_api_key: str = "", search_filter: str = "") -> List[Dict[str, Any]]:
    headers = {"User-Agent": "CiteLume/1.0"}
    if hf_api_key and hf_api_key.strip():
        headers["Authorization"] = f"Bearer {hf_api_key.strip()}"

    params = {
        "pipeline_tag": "text-generation",
        "sort": "downloads",
        "direction": "-1",
        "limit": "30",
    }
    if search_filter and search_filter.strip():
        params["search"] = search_filter.strip()

    models_out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=12.0, headers=headers, follow_redirects=True) as client:
        try:
            resp = await client.get("https://huggingface.co/api/models", params=params)
            if resp.status_code == 200:
                for item in resp.json():
                    mid = item.get("id") or item.get("modelId")
                    if not mid:
                        continue
                    models_out.append(
                        {
                            "id": mid,
                            "pipeline_tag": item.get("pipeline_tag") or "text-generation",
                            "downloads": int(item.get("downloads") or 0),
                            "likes": int(item.get("likes") or 0),
                            "tags": (item.get("tags") or [])[:6],
                        }
                    )
        except Exception as exc:
            print(f"[HF Available Models Fetch Warning] {exc}")

    if not models_out:
        models_out = [
            {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "pipeline_tag": "text-generation", "downloads": 1920000, "likes": 1750, "tags": ["reasoning"]},
            {"id": "meta-llama/Llama-3.3-70B-Instruct", "pipeline_tag": "text-generation", "downloads": 2400000, "likes": 2150, "tags": ["llama-3", "instruct"]},
            {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "pipeline_tag": "text-generation", "downloads": 1600000, "likes": 1980, "tags": ["moe", "instruct"]},
            {"id": "Qwen/Qwen2.5-72B-Instruct", "pipeline_tag": "text-generation", "downloads": 1850000, "likes": 1420, "tags": ["instruct", "chat"]},
        ]
    models_out.sort(key=lambda x: x["id"].lower())
    return models_out
