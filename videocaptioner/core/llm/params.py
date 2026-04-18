"""Utilities for optional LLM request parameters."""

import json
from typing import Any

CHAT_COMPLETION_KWARGS = {
    "audio",
    "extra_body",
    "extra_headers",
    "extra_query",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning_effort",
    "response_format",
    "safety_identifier",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "timeout",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "verbosity",
    "web_search_options",
}


def parse_llm_extra_params(value: Any) -> dict[str, Any]:
    """Parse user-provided LLM extra parameters.

    Accepts either a dict or a JSON object string. Empty values produce an empty dict.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM custom parameters must be a JSON object")
        return parsed
    raise ValueError("LLM custom parameters must be a dict or JSON object string")


def prepare_llm_request_params(
    defaults: dict[str, Any] | None = None,
    extra: Any = None,
    protected_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Merge defaults and custom params for OpenAI-compatible Chat Completions.

    Unknown top-level keys are moved into extra_body so compatible providers such
    as OpenRouter can receive provider-specific fields like {"reasoning": {...}}.
    """
    protected_keys = protected_keys or set()
    merged = dict(defaults or {})
    for key, value in parse_llm_extra_params(extra).items():
        if key not in protected_keys:
            merged[key] = value

    extra_body = dict(merged.get("extra_body") or {})
    normalized: dict[str, Any] = {}

    for key, value in merged.items():
        if key == "extra_body":
            continue
        if key in CHAT_COMPLETION_KWARGS:
            normalized[key] = value
        else:
            extra_body[key] = value

    if extra_body:
        normalized["extra_body"] = extra_body

    return normalized


def serialize_llm_extra_params(value: Any) -> str:
    """Return a stable string for cache keys and logs."""
    params = parse_llm_extra_params(value)
    if not params:
        return ""
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
