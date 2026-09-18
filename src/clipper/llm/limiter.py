"""Token-bucket rate limiting, configured from settings.yaml.

The point is to never see a 429. Backoff handles the ones that arrive anyway,
but a limiter that self-throttles just under the published ceiling is strictly
better: a 429 on a free tier often costs more than the wait would have.

Three limits, because that is what providers publish:

- RPM  requests per minute   token bucket, refills continuously
- TPM  tokens per minute     token bucket, refills continuously
- RPD  requests per day      hard counter, resets at UTC midnight

RPM and TPM are buckets so a burst is allowed up to capacity and then smoothed.
RPD is not a bucket: a daily quota really does reset at a wall-clock boundary,
and modelling it as a slow trickle would let a run drip past the ceiling.

The clock and sleep are injected so tests run instantly and deterministically.
Set any limit to 0 to disable it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

#: Stops a misconfiguration turning into a hang.
MAX_ACQUIRE_ITERATIONS = 64


class TokenBucket:
    """A classic token bucket. Thread-safe, with an injectable clock."""

    def __init__(
        self,
        capacity: float,
        refill_per_s: float,
        *,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._time = time_fn
        self._tokens = float(capacity)
        self._last = time_fn()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    @property
    def tokens(self) -> float:
        """Currently available tokens, after refilling to now."""
        with self._lock:
            self._refill()
            return self._tokens

    def _refill(self) -> None:
        now = self._time()
        elapsed = now - self._last
        if elapsed > 0:
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_s)

    def wait_time(self, amount: float = 1.0) -> float:
        """Seconds until ``amount`` tokens are available. 0 if they are now."""
        if not self.enabled:
            return 0.0
        with self._lock:
            self._refill()
            # A single request larger than the whole bucket would otherwise wait
            # forever. Clamp it: waiting for a full bucket is the best we can do.
            need = min(float(amount), self.capacity)
            if self._tokens >= need:
                return 0.0
            if self.refill_per_s <= 0:
                return float("inf")
            return (need - self._tokens) / self.refill_per_s

    def consume(self, amount: float = 1.0) -> None:
        """Deduct tokens. May go negative if the caller did not wait first."""
        if not self.enabled:
            return
        with self._lock:
            self._refill()
            self._tokens -= float(amount)

    def try_consume(self, amount: float = 1.0) -> bool:
        """Consume only if available. Returns whether it happened."""
        if not self.enabled:
            return True
        with self._lock:
            self._refill()
            if self._tokens < amount:
                return False
            self._tokens -= float(amount)
            return True


class DailyQuota:
    """Requests-per-day counter that resets at UTC midnight."""

    def __init__(
        self,
        limit: int,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.limit = int(limit)
        self._now = now_fn or (lambda: datetime.now(UTC))
        self._used = 0
        self._day = self._now().date()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    @property
    def used(self) -> int:
        with self._lock:
            self._maybe_reset()
            return self._used

    @property
    def remaining(self) -> int:
        if not self.enabled:
            return 2**31
        return max(0, self.limit - self.used)

    def _maybe_reset(self) -> None:
        today = self._now().date()
        if today != self._day:
            self._day = today
            self._used = 0

    def wait_time(self, amount: int = 1) -> float:
        """Seconds until UTC midnight if the quota is spent, else 0."""
        if not self.enabled:
            return 0.0
        with self._lock:
            self._maybe_reset()
            if self._used + amount <= self.limit:
                return 0.0
            now = self._now()
            midnight = datetime.combine(
                now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            )
            return max(0.0, (midnight - now).total_seconds())

    def consume(self, amount: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._maybe_reset()
            self._used += amount


@dataclass(frozen=True)
class Limits:
    """Published provider limits, straight from settings.yaml. 0 disables one."""

    rpm: int = 0
    tpm: int = 0
    rpd: int = 0

    @classmethod
    def from_config(cls, cfg: dict | None) -> Limits:
        cfg = cfg or {}
        return cls(
            rpm=int(cfg.get("rpm", 0) or 0),
            tpm=int(cfg.get("tpm", 0) or 0),
            rpd=int(cfg.get("rpd", 0) or 0),
        )


class RateLimiter:
    """Combines RPM, TPM and RPD into one ``acquire`` gate.

    Usage, once per call:

        limiter.acquire(estimated_tokens)     # blocks until there is budget
        response = provider.complete(...)
        limiter.settle(estimated_tokens, response.tokens_total)

    ``settle`` reconciles the pre-call estimate against real usage, so a run of
    badly-estimated prompts cannot drift the TPM accounting.
    """

    def __init__(
        self,
        limits: Limits,
        *,
        safety_factor: float = 0.9,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.limits = limits
        # Aim just under the published ceiling. Provider-side accounting rarely
        # matches ours exactly, and being 10% low costs far less than a 429.
        self.safety_factor = safety_factor
        self._sleep = sleep_fn

        rpm = limits.rpm * safety_factor
        tpm = limits.tpm * safety_factor

        self.requests = TokenBucket(rpm, rpm / 60.0 if rpm else 0.0, time_fn=time_fn)
        self.tokens = TokenBucket(tpm, tpm / 60.0 if tpm else 0.0, time_fn=time_fn)
        self.daily = DailyQuota(limits.rpd, now_fn=now_fn)

        self.total_waited_s = 0.0
        self.acquisitions = 0

    def wait_time(self, tokens: int) -> float:
        """Longest of the three waits. 0 means the call may go now."""
        return max(
            self.requests.wait_time(1),
            self.tokens.wait_time(tokens),
            self.daily.wait_time(1),
        )

    def acquire(self, tokens: int) -> float:
        """Block until there is budget for one call of ``tokens`` tokens.

        Returns the seconds spent waiting, which the router logs.
        """
        waited = 0.0
        for _ in range(MAX_ACQUIRE_ITERATIONS):
            delay = self.wait_time(tokens)
            if delay <= 0:
                self.requests.consume(1)
                self.tokens.consume(min(tokens, self.tokens.capacity or tokens))
                self.daily.consume(1)
                self.total_waited_s += waited
                self.acquisitions += 1
                return waited
            if delay == float("inf"):
                raise RuntimeError(
                    "rate limiter can never satisfy this request; check the "
                    "rpm/tpm/rpd values in settings.yaml"
                )
            self._sleep(delay)
            waited += delay

        raise RuntimeError(
            f"rate limiter did not settle after {MAX_ACQUIRE_ITERATIONS} waits; "
            "check the rpm/tpm/rpd values in settings.yaml"
        )

    def settle(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Correct the TPM bucket once real usage is known."""
        if not self.tokens.enabled or actual_tokens <= 0:
            return
        delta = actual_tokens - estimated_tokens
        if delta > 0:
            self.tokens.consume(delta)
        elif delta < 0:
            # Hand back what we over-reserved, without exceeding capacity.
            self.tokens.consume(delta)

    def penalize(self, retry_after: float | None) -> None:
        """Drain the request bucket after a 429.

        The provider just told us our accounting is wrong, so stop issuing
        requests for a while regardless of what the bucket thinks.
        """
        if not self.requests.enabled:
            return
        self.requests.consume(self.requests.capacity)

    def snapshot(self) -> dict[str, float]:
        """Current budget, for logging and debugging."""
        return {
            "rpm_available": round(self.requests.tokens, 2),
            "tpm_available": round(self.tokens.tokens, 2),
            "rpd_remaining": self.daily.remaining,
            "total_waited_s": round(self.total_waited_s, 3),
            "acquisitions": self.acquisitions,
        }
