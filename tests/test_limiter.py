"""Rate limiter tests. No network, no real sleeping - the clock is injected."""

from __future__ import annotations

import pytest

from clipper.llm.limiter import DailyQuota, Limits, RateLimiter, TokenBucket


class TestTokenBucket:
    def test_starts_full(self, clock):
        bucket = TokenBucket(10, 1.0, time_fn=clock.time)
        assert bucket.tokens == 10
        assert bucket.wait_time(10) == 0

    def test_consume_reduces_available(self, clock):
        bucket = TokenBucket(10, 1.0, time_fn=clock.time)
        bucket.consume(4)
        assert bucket.tokens == pytest.approx(6)

    def test_refills_over_time(self, clock):
        bucket = TokenBucket(10, 1.0, time_fn=clock.time)  # 1 token/sec
        bucket.consume(10)
        assert bucket.tokens == pytest.approx(0)
        clock.advance(3)
        assert bucket.tokens == pytest.approx(3)

    def test_refill_is_capped_at_capacity(self, clock):
        bucket = TokenBucket(10, 1.0, time_fn=clock.time)
        bucket.consume(5)
        clock.advance(10_000)
        assert bucket.tokens == pytest.approx(10)

    def test_wait_time_when_empty(self, clock):
        bucket = TokenBucket(10, 2.0, time_fn=clock.time)  # 2 tokens/sec
        bucket.consume(10)
        assert bucket.wait_time(4) == pytest.approx(2.0)

    def test_request_larger_than_capacity_is_clamped(self, clock):
        """A prompt bigger than the whole TPM bucket must not wait forever."""
        bucket = TokenBucket(100, 10.0, time_fn=clock.time)
        bucket.consume(100)
        # Asking for 500 against a 100 bucket waits for a full bucket, not 5x it.
        assert bucket.wait_time(500) == pytest.approx(10.0)

    def test_try_consume_respects_availability(self, clock):
        bucket = TokenBucket(5, 1.0, time_fn=clock.time)
        assert bucket.try_consume(5) is True
        assert bucket.try_consume(1) is False
        clock.advance(1)
        assert bucket.try_consume(1) is True

    def test_zero_capacity_disables_the_limit(self, clock):
        bucket = TokenBucket(0, 0, time_fn=clock.time)
        assert bucket.enabled is False
        assert bucket.wait_time(10_000) == 0
        assert bucket.try_consume(10_000) is True


class TestDailyQuota:
    def test_counts_usage(self, wall_clock):
        quota = DailyQuota(3, now_fn=wall_clock)
        assert quota.remaining == 3
        quota.consume()
        quota.consume()
        assert quota.used == 2
        assert quota.remaining == 1

    def test_blocks_when_exhausted(self, wall_clock):
        quota = DailyQuota(2, now_fn=wall_clock)
        quota.consume(2)
        assert quota.wait_time(1) > 0

    def test_wait_is_until_utc_midnight(self, wall_clock):
        quota = DailyQuota(1, now_fn=wall_clock)  # clock starts at 12:00 UTC
        quota.consume()
        assert quota.wait_time(1) == pytest.approx(12 * 3600)

    def test_resets_on_the_next_utc_day(self, wall_clock):
        quota = DailyQuota(2, now_fn=wall_clock)
        quota.consume(2)
        assert quota.remaining == 0
        wall_clock.advance(days=1)
        assert quota.remaining == 2
        assert quota.wait_time(1) == 0

    def test_zero_limit_disables(self, wall_clock):
        quota = DailyQuota(0, now_fn=wall_clock)
        assert quota.enabled is False
        quota.consume(10_000)
        assert quota.wait_time(1) == 0


class TestLimits:
    def test_from_config_reads_settings_block(self):
        limits = Limits.from_config({"rpm": 10, "tpm": 250000, "rpd": 250})
        assert (limits.rpm, limits.tpm, limits.rpd) == (10, 250000, 250)

    def test_missing_and_null_values_disable(self):
        assert Limits.from_config(None) == Limits(0, 0, 0)
        assert Limits.from_config({"rpm": None}).rpm == 0


def make_limiter(clock, wall_clock, **kwargs) -> RateLimiter:
    limits = Limits(**{"rpm": 60, "tpm": 6000, "rpd": 1000, **kwargs})
    return RateLimiter(
        limits,
        safety_factor=1.0,  # exact numbers, so assertions stay readable
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        now_fn=wall_clock,
    )


class TestRateLimiter:
    def test_first_call_does_not_wait(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock)
        assert limiter.acquire(100) == 0
        assert clock.slept == []

    def test_safety_factor_shrinks_the_ceiling(self, clock, wall_clock):
        """We aim under the published limit rather than at it."""
        limiter = RateLimiter(
            Limits(rpm=100, tpm=1000, rpd=0),
            safety_factor=0.9,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
            now_fn=wall_clock,
        )
        assert limiter.requests.capacity == pytest.approx(90)
        assert limiter.tokens.capacity == pytest.approx(900)

    def test_throttles_when_rpm_is_spent(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=2, tpm=0, rpd=0)
        limiter.acquire(1)
        limiter.acquire(1)
        # Bucket is empty; the third call has to wait for a refill.
        limiter.acquire(1)
        assert clock.total_slept > 0

    def test_throttles_on_tokens_not_just_requests(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=0, tpm=1000, rpd=0)
        limiter.acquire(1000)
        limiter.acquire(500)
        # 500 tokens at 1000/min = 30s of refill.
        assert clock.total_slept == pytest.approx(30.0)

    def test_self_throttling_keeps_us_under_the_published_rate(self, clock, wall_clock):
        """The whole point: 10 calls at rpm=10 must span at least a minute."""
        limiter = make_limiter(clock, wall_clock, rpm=10, tpm=0, rpd=0)
        start = clock.now
        for _ in range(20):
            limiter.acquire(1)
        elapsed = clock.now - start
        # 20 requests against a 10/min budget: the first 10 burst, the rest wait.
        assert elapsed >= 60.0

    def test_daily_quota_blocks_until_midnight(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=0, tpm=0, rpd=2)
        limiter.acquire(1)
        limiter.acquire(1)
        limiter.acquire(1)
        assert clock.total_slept == pytest.approx(12 * 3600)

    def test_acquire_reports_time_waited(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=0, tpm=600, rpd=0)
        limiter.acquire(600)
        waited = limiter.acquire(600)
        assert waited == pytest.approx(60.0)

    def test_settle_charges_underestimated_tokens(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=0, tpm=1000, rpd=0)
        limiter.acquire(100)
        assert limiter.tokens.tokens == pytest.approx(900)
        limiter.settle(estimated_tokens=100, actual_tokens=400)
        assert limiter.tokens.tokens == pytest.approx(600)

    def test_settle_refunds_overestimated_tokens(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=0, tpm=1000, rpd=0)
        limiter.acquire(500)
        limiter.settle(estimated_tokens=500, actual_tokens=200)
        assert limiter.tokens.tokens == pytest.approx(800)

    def test_settle_ignores_missing_usage(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=0, tpm=1000, rpd=0)
        limiter.acquire(100)
        limiter.settle(estimated_tokens=100, actual_tokens=0)
        assert limiter.tokens.tokens == pytest.approx(900)

    def test_penalize_drains_the_request_bucket(self, clock, wall_clock):
        """After a 429 our accounting is wrong, so stop sending."""
        limiter = make_limiter(clock, wall_clock, rpm=60, tpm=0, rpd=0)
        limiter.acquire(1)
        limiter.penalize(retry_after=5.0)
        assert limiter.requests.wait_time(1) > 0

    def test_all_limits_disabled_never_waits(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=0, tpm=0, rpd=0)
        for _ in range(500):
            limiter.acquire(10_000)
        assert clock.slept == []

    def test_bucket_that_never_refills_is_reported_not_awaited(self, clock):
        """capacity > 0 with no refill can never be satisfied."""
        bucket = TokenBucket(10, 0.0, time_fn=clock.time)
        bucket.consume(10)
        assert bucket.wait_time(1) == float("inf")

    def test_guard_stops_a_stuck_acquire_from_hanging(self, wall_clock):
        """A sleep that does not advance time must raise, not spin forever."""
        limiter = RateLimiter(
            Limits(rpm=1, tpm=0, rpd=0),
            safety_factor=1.0,
            time_fn=lambda: 0.0,
            sleep_fn=lambda _s: None,  # broken sleep: time never moves
            now_fn=wall_clock,
        )
        limiter.acquire(1)
        with pytest.raises(RuntimeError, match="did not settle"):
            limiter.acquire(1)

    def test_snapshot_reports_budget(self, clock, wall_clock):
        limiter = make_limiter(clock, wall_clock, rpm=10, tpm=1000, rpd=100)
        limiter.acquire(250)
        snap = limiter.snapshot()
        assert snap["rpm_available"] == pytest.approx(9)
        assert snap["tpm_available"] == pytest.approx(750)
        assert snap["rpd_remaining"] == 99
        assert snap["acquisitions"] == 1
