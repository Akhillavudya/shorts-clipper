"""Tier routing, retry, failover, logging, and caching.

Pipeline code calls ``router.complete("recall", prompt, json_schema=SCHEMA)``
and never names a model. ``config/settings.yaml`` maps tiers to concrete
provider/model pairs, so switching provider is a config edit.

Order of operations for one call:

1. cache lookup      - a hit costs nothing and skips everything below
2. limiter.acquire   - self-throttle so we do not earn a 429
3. provider.complete - the actual call
4. on failure        - backoff with jitter, honouring Retry-After, up to
                       ``max_attempts``; then fail over to the fallback provider
5. log + cache       - append to jobs/<id>/llm.log, write the cache entry
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from clipper.llm.base import (
    AuthError,
    LLMError,
    LLMResponse,
    Provider,
    ProviderUnavailable,
    RateLimitError,
    ResponseFormatError,
    TransientError,
    estimate_tokens,
)
from clipper.llm.cache import CACHE_DIRNAME, LLMCache, make_key
from clipper.llm.limiter import Limits, RateLimiter

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")
LOG_FILENAME = "llm.log"


class NoProviderAvailable(LLMError):
    """Both the primary and the fallback failed, or neither could be built."""


@dataclass(frozen=True)
class TierConfig:
    """What a logical tier resolves to."""

    provider: str
    model: str

    @classmethod
    def from_config(cls, cfg: dict, *, tier: str) -> TierConfig:
        try:
            return cls(provider=cfg["provider"], model=cfg["model"])
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"tier {tier!r} in settings.yaml needs both 'provider' and 'model'"
            ) from exc


def load_settings(path: Path | str = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"settings file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_provider(name: str, cfg: dict[str, Any]) -> Provider:
    """Construct a provider from its settings.yaml block."""
    if name == "gemini":
        from clipper.llm.gemini import GeminiProvider

        return GeminiProvider(api_key_env=cfg.get("api_key_env", "GEMINI_API_KEY"))
    if name == "groq":
        from clipper.llm.groq import DEFAULT_BASE_URL, GroqProvider

        return GroqProvider(
            api_key_env=cfg.get("api_key_env", "GROQ_API_KEY"),
            base_url=cfg.get("base_url", DEFAULT_BASE_URL),
        )
    if name == "ollama":
        from clipper.llm.ollama import DEFAULT_BASE_URL, OllamaProvider

        return OllamaProvider(base_url=cfg.get("base_url", DEFAULT_BASE_URL))
    raise ProviderUnavailable(f"unknown provider {name!r} in settings.yaml")


class Router:
    """Resolves tiers to providers and runs calls with quota discipline."""

    def __init__(
        self,
        settings: dict[str, Any] | Path | str = DEFAULT_SETTINGS_PATH,
        *,
        job_dir: Path | str | None = None,
        providers: dict[str, Provider] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings if isinstance(settings, dict) else load_settings(settings)
        llm_cfg = self.settings.get("llm") or {}

        self.max_attempts = int(llm_cfg.get("max_attempts", 4))
        self.backoff_base_s = float(llm_cfg.get("backoff_base_s", 1.0))
        self.backoff_max_s = float(llm_cfg.get("backoff_max_s", 60.0))
        self.backoff_jitter = float(llm_cfg.get("backoff_jitter", 0.3))
        self.timeout_s = float(llm_cfg.get("timeout_s", 120.0))
        self.failover_enabled = bool(llm_cfg.get("failover_enabled", True))
        self.log_calls = bool(llm_cfg.get("log_calls", True))

        self._sleep = sleep_fn
        self._rng = rng or random.Random()
        self._time = time_fn

        # Injected providers are used as-is; anything else is built on demand so
        # a missing GROQ_API_KEY does not break a run that never fails over.
        self._provider_overrides = providers or {}
        self._providers: dict[str, Provider] = dict(self._provider_overrides)
        self._limiters: dict[str, RateLimiter] = {}

        self.job_dir = Path(job_dir) if job_dir else None
        cache_enabled = bool(llm_cfg.get("cache_enabled", True)) and self.job_dir is not None
        self.cache = LLMCache(
            (self.job_dir / CACHE_DIRNAME) if self.job_dir else Path(CACHE_DIRNAME),
            enabled=cache_enabled,
        )

    # -- configuration --------------------------------------------------

    def tier(self, name: str) -> TierConfig:
        tiers = self.settings.get("tiers") or {}
        if name not in tiers:
            raise KeyError(f"unknown tier {name!r}; settings.yaml defines {sorted(tiers)}")
        return TierConfig.from_config(tiers[name], tier=name)

    def fallback(self) -> TierConfig | None:
        cfg = self.settings.get("fallback")
        return TierConfig.from_config(cfg, tier="fallback") if cfg else None

    def provider(self, name: str) -> Provider:
        if name not in self._providers:
            provider_cfg = (self.settings.get("providers") or {}).get(name) or {}
            self._providers[name] = _build_provider(name, provider_cfg)
        return self._providers[name]

    def limiter(self, name: str) -> RateLimiter:
        if name not in self._limiters:
            provider_cfg = (self.settings.get("providers") or {}).get(name) or {}
            self._limiters[name] = RateLimiter(
                Limits.from_config(provider_cfg.get("limits")),
                sleep_fn=self._sleep,
                time_fn=self._time,
            )
        return self._limiters[name]

    # -- backoff --------------------------------------------------------

    def _backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with jitter, floored by any Retry-After.

        Retry-After is authoritative when present - the provider knows when the
        window reopens better than our exponent does - but jitter is still
        applied on top so parallel jobs do not all retry on the same tick.
        """
        delay = min(self.backoff_base_s * (2 ** (attempt - 1)), self.backoff_max_s)
        if retry_after is not None and retry_after > delay:
            delay = min(retry_after, self.backoff_max_s)
        if self.backoff_jitter:
            spread = delay * self.backoff_jitter
            delay = max(0.0, delay + self._rng.uniform(-spread, spread))
        return delay

    # -- logging --------------------------------------------------------

    @property
    def log_path(self) -> Path | None:
        return (self.job_dir / LOG_FILENAME) if self.job_dir else None

    def _log(self, record: dict[str, Any]) -> None:
        """Append one JSON line to jobs/<id>/llm.log.

        JSONL rather than prose so spend can be totalled with a one-liner:
            jq -s 'map(.tokens_in + .tokens_out) | add' jobs/<id>/llm.log
        Logging must never break a run, so write failures are swallowed.
        """
        if not self.log_calls:
            return
        path = self.log_path
        if path is None:
            return
        record = {"ts": datetime.now(UTC).isoformat(), **record}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # -- the call -------------------------------------------------------

    def complete(
        self,
        tier: str,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any],
        max_tokens: int = 8192,
        temperature: float = 0.2,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Run a call on ``tier`` and return the parsed JSON object."""
        return self.complete_full(
            tier,
            prompt,
            system=system,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=use_cache,
        ).data

    def complete_full(
        self,
        tier: str,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any],
        max_tokens: int = 8192,
        temperature: float = 0.2,
        use_cache: bool = True,
    ) -> LLMResponse:
        """As ``complete``, but returns the full response with usage metadata."""
        primary = self.tier(tier)
        targets: list[tuple[TierConfig, bool]] = [(primary, False)]

        fallback = self.fallback()
        if self.failover_enabled and fallback and fallback != primary:
            targets.append((fallback, True))

        last_error: Exception | None = None

        for target, is_fallback in targets:
            key = make_key(
                prompt,
                target.model,
                system=system,
                json_schema=json_schema,
                temperature=temperature,
            )
            if use_cache:
                hit = self.cache.get(key)
                if hit is not None:
                    hit.tier = tier
                    self._log(
                        {
                            "tier": tier,
                            "provider": hit.provider,
                            "model": hit.model,
                            "cached": True,
                            "tokens_in": hit.tokens_in,
                            "tokens_out": hit.tokens_out,
                            "latency_s": 0.0,
                            "status": "cache_hit",
                        }
                    )
                    return hit

            try:
                response = self._call_with_retries(
                    target,
                    tier=tier,
                    prompt=prompt,
                    system=system,
                    json_schema=json_schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    is_fallback=is_fallback,
                )
            except (LLMError, ProviderUnavailable) as exc:
                last_error = exc
                self._log(
                    {
                        "tier": tier,
                        "provider": target.provider,
                        "model": target.model,
                        "status": "exhausted",
                        "error": f"{type(exc).__name__}: {exc}",
                        "failing_over": not is_fallback and len(targets) > 1,
                    }
                )
                continue

            if use_cache:
                self.cache.put(key, response)
            return response

        raise NoProviderAvailable(
            f"tier {tier!r} failed on every configured provider"
        ) from last_error

    def _call_with_retries(
        self,
        target: TierConfig,
        *,
        tier: str,
        prompt: str,
        system: str | None,
        json_schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
        is_fallback: bool,
    ) -> LLMResponse:
        """Try one provider up to ``max_attempts`` times, then give up on it."""
        provider = self.provider(target.provider)
        limiter = self.limiter(target.provider)
        estimated = estimate_tokens(prompt, system)

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            waited = limiter.acquire(estimated)
            started = self._time()
            try:
                response = provider.complete(
                    prompt,
                    system=system,
                    json_schema=json_schema,
                    model=target.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=self.timeout_s,
                )
            except RateLimitError as exc:
                # We were throttled despite self-throttling, so our limits are
                # wrong. Drain the bucket as well as backing off.
                last_error = exc
                limiter.penalize(exc.retry_after)
                delay = self._backoff_delay(attempt, exc.retry_after)
                self._log(
                    {
                        "tier": tier,
                        "provider": target.provider,
                        "model": target.model,
                        "attempt": attempt,
                        "status": "rate_limited",
                        "retry_after": exc.retry_after,
                        "sleeping_s": round(delay, 3),
                        "error": str(exc),
                    }
                )
                if attempt < self.max_attempts:
                    self._sleep(delay)
                continue
            except AuthError as exc:
                # Credentials will not fix themselves. Stop immediately so the
                # caller fails over rather than burning retries.
                self._log(
                    {
                        "tier": tier,
                        "provider": target.provider,
                        "model": target.model,
                        "attempt": attempt,
                        "status": "auth_error",
                        "error": str(exc),
                    }
                )
                raise
            except (TransientError, ResponseFormatError) as exc:
                last_error = exc
                delay = self._backoff_delay(attempt)
                self._log(
                    {
                        "tier": tier,
                        "provider": target.provider,
                        "model": target.model,
                        "attempt": attempt,
                        "status": type(exc).__name__,
                        "sleeping_s": round(delay, 3),
                        "error": str(exc),
                    }
                )
                if attempt < self.max_attempts:
                    self._sleep(delay)
                continue

            latency = self._time() - started
            limiter.settle(estimated, response.tokens_total)
            response.tier = tier
            response.attempts = attempt
            if not response.latency_s:
                response.latency_s = latency

            self._log(
                {
                    "tier": tier,
                    "provider": response.provider,
                    "model": response.model,
                    "attempt": attempt,
                    "status": "ok",
                    "cached": False,
                    "is_fallback": is_fallback,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "latency_s": round(response.latency_s, 3),
                    "throttled_s": round(waited, 3),
                    "budget": limiter.snapshot(),
                }
            )
            return response

        raise last_error or TransientError(
            f"{target.provider} failed after {self.max_attempts} attempts"
        )

    # -- introspection --------------------------------------------------

    def usage(self) -> dict[str, Any]:
        """Cache and limiter counters, for a run summary."""
        return {
            "cache": self.cache.stats(),
            "limiters": {n: lim.snapshot() for n, lim in self._limiters.items()},
        }


def router_for_job(
    job_dir: Path | str,
    settings_path: Path | str = DEFAULT_SETTINGS_PATH,
) -> Router:
    """Convenience constructor used by the pipeline stages."""
    return Router(settings_path, job_dir=job_dir)


__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "LOG_FILENAME",
    "NoProviderAvailable",
    "Router",
    "TierConfig",
    "load_settings",
    "router_for_job",
]
