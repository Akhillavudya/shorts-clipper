"""Router tests: tiers, backoff, failover, logging, caching.

Providers are fakes. Nothing here opens a socket.
"""

from __future__ import annotations

import json
import random

import pytest

from clipper.llm.base import (
    AuthError,
    LLMResponse,
    RateLimitError,
    ResponseFormatError,
    TransientError,
)
from clipper.llm.router import NoProviderAvailable, Router

SCHEMA = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}

SETTINGS = {
    "tiers": {
        "recall": {"provider": "primary", "model": "model-lite"},
        "rerank": {"provider": "primary", "model": "model-pro"},
    },
    "fallback": {"provider": "backup", "model": "model-backup"},
    "providers": {
        "primary": {"limits": {"rpm": 0, "tpm": 0, "rpd": 0}},
        "backup": {"limits": {"rpm": 0, "tpm": 0, "rpd": 0}},
    },
    "llm": {
        "max_attempts": 3,
        "backoff_base_s": 1.0,
        "backoff_max_s": 60.0,
        "backoff_jitter": 0.0,
        "failover_enabled": True,
        "cache_enabled": True,
        "log_calls": True,
    },
}


class FakeProvider:
    """Replays a scripted sequence of results or exceptions."""

    def __init__(self, name: str, script=None) -> None:
        self.name = name
        self.script = list(script or [])
        self.calls: list[dict] = []

    def complete(self, prompt, *, system=None, json_schema, model, **kwargs) -> LLMResponse:
        self.calls.append({"prompt": prompt, "model": model, "system": system})
        outcome = self.script.pop(0) if self.script else None
        if isinstance(outcome, Exception):
            raise outcome
        data = outcome if outcome is not None else {"ok": True}
        return LLMResponse(
            data=data,
            text=json.dumps(data),
            provider=self.name,
            model=model,
            tokens_in=100,
            tokens_out=20,
            latency_s=0.5,
        )


def make_router(job_dir, clock, *, primary=None, backup=None, settings=None) -> Router:
    providers = {
        "primary": primary or FakeProvider("primary"),
        "backup": backup or FakeProvider("backup"),
    }
    return Router(
        settings or SETTINGS,
        job_dir=job_dir,
        providers=providers,
        sleep_fn=clock.sleep,
        time_fn=clock.time,
        rng=random.Random(0),
    )


def read_log(job_dir) -> list[dict]:
    path = job_dir / "llm.log"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestTierResolution:
    def test_tiers_map_to_models(self, job_dir, clock):
        router = make_router(job_dir, clock)
        assert router.tier("recall").model == "model-lite"
        assert router.tier("rerank").model == "model-pro"

    def test_unknown_tier_raises(self, job_dir, clock):
        with pytest.raises(KeyError, match="unknown tier"):
            make_router(job_dir, clock).tier("nope")

    def test_pipeline_never_needs_a_model_name(self, job_dir, clock):
        """Callers pass a tier; the router supplies the model."""
        primary = FakeProvider("primary")
        router = make_router(job_dir, clock, primary=primary)
        router.complete("rerank", "score these", json_schema=SCHEMA)
        assert primary.calls[0]["model"] == "model-pro"


class TestRetryAndBackoff:
    def test_succeeds_first_time(self, job_dir, clock):
        primary = FakeProvider("primary", [{"ok": True}])
        router = make_router(job_dir, clock, primary=primary)
        assert router.complete("recall", "p", json_schema=SCHEMA) == {"ok": True}
        assert clock.slept == []

    def test_retries_transient_errors(self, job_dir, clock):
        primary = FakeProvider("primary", [TransientError("503"), {"ok": True}])
        router = make_router(job_dir, clock, primary=primary)
        assert router.complete("recall", "p", json_schema=SCHEMA) == {"ok": True}
        assert len(primary.calls) == 2

    def test_backoff_is_exponential(self, job_dir, clock):
        primary = FakeProvider("primary", [TransientError("a"), TransientError("b"), {"ok": True}])
        router = make_router(job_dir, clock, primary=primary)
        router.complete("recall", "p", json_schema=SCHEMA)
        assert clock.slept == [1.0, 2.0]

    def test_honours_retry_after(self, job_dir, clock):
        """A provider's Retry-After beats our exponent."""
        primary = FakeProvider("primary", [RateLimitError("429", retry_after=27.0), {"ok": True}])
        router = make_router(job_dir, clock, primary=primary)
        router.complete("recall", "p", json_schema=SCHEMA)
        assert clock.slept == [27.0]

    def test_retry_after_is_capped_by_backoff_max(self, job_dir, clock):
        primary = FakeProvider("primary", [RateLimitError("429", retry_after=9999.0), {"ok": True}])
        router = make_router(job_dir, clock, primary=primary)
        router.complete("recall", "p", json_schema=SCHEMA)
        assert clock.slept == [60.0]

    def test_jitter_spreads_retries(self, job_dir, clock):
        settings = {**SETTINGS, "llm": {**SETTINGS["llm"], "backoff_jitter": 0.3}}
        primary = FakeProvider("primary", [TransientError("a"), {"ok": True}])
        router = make_router(job_dir, clock, primary=primary, settings=settings)
        router.complete("recall", "p", json_schema=SCHEMA)
        # Base delay is 1.0; jitter must move it without leaving +/-30%.
        assert clock.slept[0] != 1.0
        assert 0.7 <= clock.slept[0] <= 1.3

    def test_bad_json_is_retried(self, job_dir, clock):
        primary = FakeProvider("primary", [ResponseFormatError("not json"), {"ok": True}])
        router = make_router(job_dir, clock, primary=primary)
        assert router.complete("recall", "p", json_schema=SCHEMA) == {"ok": True}


class TestFailover:
    def test_fails_over_after_exhausting_the_primary(self, job_dir, clock):
        primary = FakeProvider("primary", [TransientError("x")] * 3)
        backup = FakeProvider("backup", [{"ok": "from-backup"}])
        router = make_router(job_dir, clock, primary=primary, backup=backup)

        assert router.complete("recall", "p", json_schema=SCHEMA) == {"ok": "from-backup"}
        assert len(primary.calls) == 3  # max_attempts
        assert len(backup.calls) == 1

    def test_failover_uses_the_fallback_model(self, job_dir, clock):
        primary = FakeProvider("primary", [TransientError("x")] * 3)
        backup = FakeProvider("backup")
        router = make_router(job_dir, clock, primary=primary, backup=backup)
        router.complete("recall", "p", json_schema=SCHEMA)
        assert backup.calls[0]["model"] == "model-backup"

    def test_auth_error_fails_over_without_burning_retries(self, job_dir, clock):
        """Credentials will not fix themselves; retrying is pure waste."""
        primary = FakeProvider("primary", [AuthError("bad key")])
        backup = FakeProvider("backup", [{"ok": True}])
        router = make_router(job_dir, clock, primary=primary, backup=backup)

        assert router.complete("recall", "p", json_schema=SCHEMA) == {"ok": True}
        assert len(primary.calls) == 1
        assert clock.slept == []

    def test_rate_limit_exhaustion_fails_over(self, job_dir, clock):
        primary = FakeProvider("primary", [RateLimitError("429", 1.0)] * 3)
        backup = FakeProvider("backup", [{"ok": True}])
        router = make_router(job_dir, clock, primary=primary, backup=backup)
        assert router.complete("recall", "p", json_schema=SCHEMA) == {"ok": True}

    def test_raises_when_both_providers_fail(self, job_dir, clock):
        primary = FakeProvider("primary", [TransientError("x")] * 3)
        backup = FakeProvider("backup", [TransientError("y")] * 3)
        router = make_router(job_dir, clock, primary=primary, backup=backup)
        with pytest.raises(NoProviderAvailable, match="failed on every configured provider"):
            router.complete("recall", "p", json_schema=SCHEMA)

    def test_failover_can_be_disabled(self, job_dir, clock):
        settings = {**SETTINGS, "llm": {**SETTINGS["llm"], "failover_enabled": False}}
        primary = FakeProvider("primary", [TransientError("x")] * 3)
        backup = FakeProvider("backup", [{"ok": True}])
        router = make_router(job_dir, clock, primary=primary, backup=backup, settings=settings)
        with pytest.raises(NoProviderAvailable):
            router.complete("recall", "p", json_schema=SCHEMA)
        assert backup.calls == []


class TestCaching:
    def test_second_identical_call_does_not_hit_the_provider(self, job_dir, clock):
        primary = FakeProvider("primary")
        router = make_router(job_dir, clock, primary=primary)

        first = router.complete("recall", "same prompt", json_schema=SCHEMA)
        second = router.complete("recall", "same prompt", json_schema=SCHEMA)

        assert first == second
        assert len(primary.calls) == 1

    def test_cache_persists_across_routers(self, job_dir, clock):
        """A rerun in a fresh process must not respend quota."""
        primary_a = FakeProvider("primary")
        make_router(job_dir, clock, primary=primary_a).complete("recall", "p", json_schema=SCHEMA)

        primary_b = FakeProvider("primary")
        make_router(job_dir, clock, primary=primary_b).complete("recall", "p", json_schema=SCHEMA)
        assert primary_b.calls == []

    def test_different_tiers_do_not_share_entries(self, job_dir, clock):
        primary = FakeProvider("primary")
        router = make_router(job_dir, clock, primary=primary)
        router.complete("recall", "p", json_schema=SCHEMA)
        router.complete("rerank", "p", json_schema=SCHEMA)
        assert len(primary.calls) == 2

    def test_cache_can_be_bypassed(self, job_dir, clock):
        primary = FakeProvider("primary")
        router = make_router(job_dir, clock, primary=primary)
        router.complete("recall", "p", json_schema=SCHEMA)
        router.complete("recall", "p", json_schema=SCHEMA, use_cache=False)
        assert len(primary.calls) == 2

    def test_cache_lives_under_the_job_directory(self, job_dir, clock):
        router = make_router(job_dir, clock)
        router.complete("recall", "p", json_schema=SCHEMA)
        assert (job_dir / ".llm_cache").is_dir()


class TestLogging:
    def test_logs_a_successful_call(self, job_dir, clock):
        router = make_router(job_dir, clock)
        router.complete("recall", "p", json_schema=SCHEMA)

        entries = read_log(job_dir)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["status"] == "ok"
        assert entry["tier"] == "recall"
        assert entry["provider"] == "primary"
        assert entry["model"] == "model-lite"
        assert entry["tokens_in"] == 100
        assert entry["tokens_out"] == 20
        assert "latency_s" in entry
        assert "ts" in entry

    def test_logs_cache_hits_separately(self, job_dir, clock):
        router = make_router(job_dir, clock)
        router.complete("recall", "p", json_schema=SCHEMA)
        router.complete("recall", "p", json_schema=SCHEMA)

        statuses = [e["status"] for e in read_log(job_dir)]
        assert statuses == ["ok", "cache_hit"]

    def test_logs_rate_limits_and_failover(self, job_dir, clock):
        primary = FakeProvider("primary", [RateLimitError("429", 2.0)] * 3)
        backup = FakeProvider("backup", [{"ok": True}])
        router = make_router(job_dir, clock, primary=primary, backup=backup)
        router.complete("recall", "p", json_schema=SCHEMA)

        entries = read_log(job_dir)
        statuses = [e["status"] for e in entries]
        assert statuses.count("rate_limited") == 3
        assert "exhausted" in statuses
        assert entries[-1]["status"] == "ok"
        assert entries[-1]["is_fallback"] is True

    def test_log_is_valid_jsonl(self, job_dir, clock):
        router = make_router(job_dir, clock)
        router.complete("recall", "a", json_schema=SCHEMA)
        router.complete("rerank", "b", json_schema=SCHEMA)

        raw = (job_dir / "llm.log").read_text(encoding="utf-8").splitlines()
        assert len(raw) == 2
        spend = sum(json.loads(line)["tokens_in"] for line in raw)
        assert spend == 200
