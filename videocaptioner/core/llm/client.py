"""Unified LLM client for the application."""

import os
import threading
from typing import Any, Callable, List, Optional
from urllib.parse import urlparse, urlunparse

import openai
from openai import OpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from videocaptioner.core.utils.cache import get_llm_cache, memoize
from videocaptioner.core.utils.logger import setup_logger

from .request_logger import create_logging_http_client, log_llm_response

_global_client: Optional[OpenAI] = None
_global_client_signature: Optional[tuple[str, str]] = None
_client_lock = threading.Lock()

logger = setup_logger("llm_client")


def normalize_base_url(base_url: str) -> str:
    """Normalize API base URL by ensuring /v1 suffix when needed."""
    url = base_url.strip()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    if not path:
        path = "/v1"

    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return normalized


def _get_env_llm_config() -> tuple[str, str]:
    """Read and validate LLM config from environment variables."""
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    base_url = normalize_base_url(base_url)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not base_url or not api_key:
        raise ValueError(
            "OPENAI_BASE_URL and OPENAI_API_KEY environment variables must be set"
        )

    return base_url, api_key


def reset_llm_client() -> None:
    """Reset cached global client so next call rebuilds it from env."""
    global _global_client, _global_client_signature
    with _client_lock:
        _global_client = None
        _global_client_signature = None


def get_llm_client() -> OpenAI:
    """Get global LLM client instance.

    The singleton is rebuilt automatically if OPENAI_BASE_URL/OPENAI_API_KEY changed.
    """
    global _global_client, _global_client_signature

    base_url, api_key = _get_env_llm_config()
    signature = (base_url, api_key)

    if _global_client is None or _global_client_signature != signature:
        with _client_lock:
            if _global_client is None or _global_client_signature != signature:
                _global_client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    http_client=create_logging_http_client(),
                )
                _global_client_signature = signature

    return _global_client


def before_sleep_log(retry_state: RetryCallState) -> None:
    logger.warning(
        "Rate Limit Error, sleeping and retrying... Please lower your thread concurrency or use better OpenAI API."
    )


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
    before_sleep=before_sleep_log,
)
def _call_llm_api(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """实际调用 LLM API（带重试）"""
    client = get_llm_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,  # pyright: ignore[reportArgumentType]
        temperature=temperature,
        **kwargs,
    )

    # 记录响应内容
    log_llm_response(response)

    return response


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
    before_sleep=before_sleep_log,
)
def call_llm_stream_text(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    on_text_delta: Optional[Callable[[str, str], None]] = None,
    **kwargs: Any,
) -> str:
    """Call LLM API in streaming mode and return the accumulated text."""
    client = get_llm_client()
    kwargs.pop("stream", None)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,  # pyright: ignore[reportArgumentType]
        temperature=temperature,
        stream=True,
        **kwargs,
    )

    content_parts: list[str] = []
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")
        if not content:
            continue
        content_parts.append(content)
        full_content = "".join(content_parts)
        if on_text_delta:
            on_text_delta(content, full_content)

    return "".join(content_parts)


@memoize(get_llm_cache(), expire=3600, typed=True)
def call_llm(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """Call LLM API with automatic caching."""
    response = _call_llm_api(messages, model, temperature, **kwargs)

    if not (
        response
        and hasattr(response, "choices")
        and response.choices
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
        and response.choices[0].message.content
    ):
        raise ValueError("Invalid OpenAI API response: empty choices or content")

    return response
