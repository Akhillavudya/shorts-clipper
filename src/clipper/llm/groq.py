"""Groq provider - the failover backend.

Groq serves an OpenAI-compatible API, so this talks to it over plain HTTP with
``httpx`` rather than pulling in another SDK.

Structured output uses ``response_format={"type": "json_object"}``, which
guarantees syntactically valid JSON but does not constrain the shape the way
Gemini's ``response_schema`` does. So the schema is also injected into the
system prompt, and the parsed result is checked for the schema's required keys
before it is returned. A failover path that silently returns a differently
shaped object is worse than one that fails.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from clipper.llm.base import (
    AuthError,
    LLMResponse,
    ProviderUnavailable,
    RateLimitError,
    ResponseFormatError,
    TransientError,
    estimate_tokens,
    parse_json_response,
)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _schema_instruction(json_schema: dict[str, Any]) -> str:
    return (
        "Respond with a single JSON object and nothing else. No markdown, no "
        "commentary. It must validate against this JSON Schema:\n"
        + json.dumps(json_schema, indent=2)
    )


class GroqProvider:
    """OpenAI-compatible client for Groq."""

    name = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "GROQ_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get(api_key_env)
        self._api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self._client = client

    def _ensure_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ProviderUnavailable(
                f"{self._api_key_env} is not set; cannot use the groq provider"
            )
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
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

        system_parts = [p for p in (system, _schema_instruction(json_schema)) if p]
        messages = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": prompt},
        ]
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        started = time.monotonic()
        try:
            resp = client.post("/chat/completions", json=body, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise TransientError(f"groq timed out after {timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"groq connection failure: {exc}") from exc
        latency = time.monotonic() - started

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else None
            except ValueError:
                delay = None
            raise RateLimitError(f"groq rate limited: {resp.text[:200]}", delay)
        if resp.status_code in (401, 403):
            raise AuthError(f"groq auth failed: {resp.text[:200]}")
        if resp.status_code >= 500:
            raise TransientError(f"groq {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise TransientError(f"groq {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ResponseFormatError(f"unexpected groq payload: {payload}") from exc

        data = parse_json_response(text)

        # Groq's json_object mode guarantees valid JSON, not the right shape.
        required = json_schema.get("required") or []
        missing = [key for key in required if key not in data]
        if missing:
            raise ResponseFormatError(f"groq response is missing required keys {missing}")

        usage = payload.get("usage") or {}
        return LLMResponse(
            data=data,
            text=text,
            provider=self.name,
            model=model,
            tokens_in=int(usage.get("prompt_tokens") or estimate_tokens(prompt, system)),
            tokens_out=int(usage.get("completion_tokens") or estimate_tokens(text)),
            latency_s=latency,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
