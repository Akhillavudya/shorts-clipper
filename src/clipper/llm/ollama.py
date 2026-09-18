"""Ollama provider - local models. Stub for later.

Wired into the router and callable, but not the default for any tier: at the
sizes that run comfortably on CPU, output quality on the rubric task is not
good enough to trust. This becomes interesting alongside local WhisperX, once
there is a GPU worth pointing both at (FULL_PLAN.md section 10).

Ollama's ``format`` parameter accepts a JSON Schema, which is the one structured
output guarantee worth having here.

Not implemented:
- streaming
- keep_alive / model preloading, which matters for cold-start latency
- any real handling of context-length overflow on long transcript chunks
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from clipper.llm.base import (
    LLMResponse,
    ResponseFormatError,
    TransientError,
    estimate_tokens,
    parse_json_response,
)

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """Local Ollama daemon. No quota, no key, no rate limit."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url)
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

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Ollama takes a JSON Schema directly here, not just "json".
            "format": json_schema,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            body["system"] = system

        started = time.monotonic()
        try:
            resp = client.post("/api/generate", json=body, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise TransientError(f"ollama timed out after {timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"ollama unreachable at {self.base_url}: {exc}") from exc
        latency = time.monotonic() - started

        if resp.status_code >= 400:
            raise TransientError(f"ollama {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        text = payload.get("response")
        if not text:
            raise ResponseFormatError(f"empty ollama response: {payload}")

        return LLMResponse(
            data=parse_json_response(text),
            text=text,
            provider=self.name,
            model=model,
            tokens_in=int(payload.get("prompt_eval_count") or estimate_tokens(prompt, system)),
            tokens_out=int(payload.get("eval_count") or estimate_tokens(text)),
            latency_s=latency,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
