"""Gemini provider - the primary backend.

Uses the ``google-genai`` SDK's native structured output: passing
``response_mime_type="application/json"`` plus a ``response_schema`` constrains
decoding, which is materially more reliable than asking for JSON in the prompt.

Model IDs are not hardcoded here. The router passes whatever
``config/settings.yaml`` maps the tier to.
"""

from __future__ import annotations

import os
import time
from typing import Any

from clipper.llm.base import (
    AuthError,
    LLMResponse,
    ProviderUnavailable,
    RateLimitError,
    TransientError,
    estimate_tokens,
    parse_json_response,
)


def _retry_after_from(exc: Exception) -> float | None:
    """Dig a Retry-After / retryDelay out of whatever the SDK raised."""
    for attr in ("retry_after", "retry_delay"):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    # google-genai surfaces {"error": {"details": [{"retryDelay": "27s"}]}}
    details = getattr(exc, "details", None)
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and "retryDelay" in item:
                text = str(item["retryDelay"]).rstrip("s")
                try:
                    return float(text)
                except ValueError:
                    pass
    return None


def _classify(exc: Exception) -> Exception:
    """Map an SDK exception onto the shared error taxonomy."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc)

    if status == 429 or "RESOURCE_EXHAUSTED" in message or "429" in message:
        return RateLimitError(f"gemini rate limited: {message}", _retry_after_from(exc))
    if status in (401, 403) or "API_KEY_INVALID" in message or "PERMISSION_DENIED" in message:
        return AuthError(f"gemini auth failed: {message}")
    if status in (500, 502, 503, 504) or "UNAVAILABLE" in message or "DEADLINE" in message:
        return TransientError(f"gemini transient failure: {message}")
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return TransientError(f"gemini connection failure: {message}")
    return TransientError(f"gemini call failed: {message}")


class GeminiProvider:
    """Primary provider. Free tier, native JSON mode."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        client: Any = None,
    ) -> None:
        self._client = client
        self._api_key = api_key or os.environ.get(api_key_env)
        self._api_key_env = api_key_env

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ProviderUnavailable(
                f"{self._api_key_env} is not set; cannot use the gemini provider"
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderUnavailable("google-genai is not installed") from exc
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any],
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        timeout_s: float = 120.0,
    ) -> LLMResponse:
        client = self._ensure_client()

        config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": json_schema,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            config["system_instruction"] = system
        if timeout_s:
            # google-genai takes http options in milliseconds.
            config["http_options"] = {"timeout": int(timeout_s * 1000)}

        started = time.monotonic()
        try:
            raw = client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as exc:
            raise _classify(exc) from exc
        latency = time.monotonic() - started

        text = getattr(raw, "text", None) or ""
        usage = getattr(raw, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0
        if not tokens_in:
            tokens_in = estimate_tokens(prompt, system)

        return LLMResponse(
            data=parse_json_response(text),
            text=text,
            provider=self.name,
            model=model,
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out or estimate_tokens(text)),
            latency_s=latency,
        )
