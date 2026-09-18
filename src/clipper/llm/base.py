"""The provider contract, plus the error taxonomy the router retries on.

JSON mode is required, not optional. Every call in this pipeline returns
structured output - candidate spans, rubric scores, discovery blurbs - so a
provider that cannot constrain output to a schema is not usable here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """Base for every error this layer raises."""


class AuthError(LLMError):
    """Missing or rejected credentials. Never retried - failover immediately."""


class RateLimitError(LLMError):
    """429. Carries ``Retry-After`` when the provider sent one."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(LLMError):
    """5xx, timeout, or connection failure. Retried with backoff."""


class ResponseFormatError(LLMError):
    """The provider returned something that is not the requested JSON.

    Retried, because with a non-zero temperature a rerun often succeeds, but it
    is not a quota problem so it does not trigger failover on its own.
    """


class ProviderUnavailable(LLMError):
    """The provider cannot be constructed at all (no key, no SDK, no daemon)."""


@dataclass(slots=True)
class LLMResponse:
    """One completed call, with everything the log and the cache need."""

    data: dict[str, Any]
    text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cached: bool = False
    attempts: int = 1
    tier: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


@runtime_checkable
class Provider(Protocol):
    """What the router needs from a backend.

    Implementations must map their SDK's failures onto the error taxonomy
    above; the router's retry and failover logic keys off those types and
    nothing else.
    """

    name: str

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
        """Run one completion and return the parsed JSON object.

        ``json_schema`` is a JSON Schema object describing the required
        response shape. Providers that support native structured output pass it
        to the API; the rest inject it into the prompt and validate on the way
        out. Either way, ``LLMResponse.data`` is a parsed dict or the call
        raises.
        """
        ...


# --- helpers shared by provider implementations --------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a model's JSON reply, tolerating the usual mangling.

    Even in JSON mode, models sometimes wrap output in a markdown fence or add
    a leading sentence. Strip what is safely strippable and fail loudly on the
    rest rather than silently returning a partial object.
    """
    if not text or not text.strip():
        raise ResponseFormatError("provider returned an empty response")

    cleaned = _FENCE.sub("", text.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Last resort: take the outermost {...} span. Helps when a model adds a
        # trailing explanation despite being asked not to.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ResponseFormatError(f"response is not JSON: {text[:200]!r}") from exc
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            raise ResponseFormatError(f"response is not JSON: {text[:200]!r}") from exc

    if not isinstance(parsed, dict):
        raise ResponseFormatError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def estimate_tokens(*parts: str | None) -> int:
    """Rough token count, used to reserve TPM budget before a call.

    Deliberately crude: ~4 characters per token, rounded up, with a floor. The
    limiter reconciles against real usage once the response comes back, so this
    only needs to be close enough to stop a burst of large prompts blowing the
    per-minute token budget.
    """
    chars = sum(len(p) for p in parts if p)
    return max(16, (chars + 3) // 4)
