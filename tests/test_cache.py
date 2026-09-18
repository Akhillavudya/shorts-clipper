"""Response cache tests. No network - responses are constructed by hand."""

from __future__ import annotations

import json

from clipper.llm.base import LLMResponse
from clipper.llm.cache import CACHE_VERSION, LLMCache, make_key


def response(**overrides) -> LLMResponse:
    defaults = dict(
        data={"candidates": [{"start_word_index": 10, "end_word_index": 90}]},
        text='{"candidates": [{"start_word_index": 10, "end_word_index": 90}]}',
        provider="gemini",
        model="gemini-3.5-flash-lite",
        tokens_in=1200,
        tokens_out=340,
        latency_s=1.7,
        tier="recall",
    )
    return LLMResponse(**{**defaults, **overrides})


class TestMakeKey:
    def test_is_deterministic(self):
        assert make_key("p", "m") == make_key("p", "m")

    def test_differs_by_prompt(self):
        assert make_key("a", "m") != make_key("b", "m")

    def test_differs_by_model(self):
        """The whole point of failover is two models; they must not collide."""
        assert make_key("p", "gemini-3.8-flash") != make_key("p", "llama-3.3-70b-versatile")

    def test_differs_by_system_prompt(self):
        assert make_key("p", "m", system="x") != make_key("p", "m", system="y")

    def test_differs_by_schema(self):
        """A changed schema must miss, not return the old shape."""
        a = make_key("p", "m", json_schema={"required": ["a"]})
        b = make_key("p", "m", json_schema={"required": ["b"]})
        assert a != b

    def test_differs_by_temperature(self):
        assert make_key("p", "m", temperature=0.0) != make_key("p", "m", temperature=0.9)

    def test_is_stable_across_schema_key_order(self):
        a = make_key("p", "m", json_schema={"type": "object", "required": ["x"]})
        b = make_key("p", "m", json_schema={"required": ["x"], "type": "object"})
        assert a == b

    def test_is_a_hex_digest(self):
        key = make_key("p", "m")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestLLMCache:
    def test_miss_on_empty_cache(self, tmp_path):
        cache = LLMCache(tmp_path / ".llm_cache")
        assert cache.get(make_key("p", "m")) is None
        assert cache.stats() == {"hits": 0, "misses": 1}

    def test_roundtrip(self, tmp_path):
        cache = LLMCache(tmp_path / ".llm_cache")
        key = make_key("prompt", "gemini-3.5-flash-lite")
        cache.put(key, response())

        hit = cache.get(key)
        assert hit is not None
        assert hit.data == {"candidates": [{"start_word_index": 10, "end_word_index": 90}]}
        assert hit.model == "gemini-3.5-flash-lite"
        assert hit.tokens_in == 1200
        assert hit.tokens_out == 340

    def test_hit_is_flagged_as_cached(self, tmp_path):
        """The router logs this, and it must never be reported as a live call."""
        cache = LLMCache(tmp_path / ".llm_cache")
        key = make_key("p", "m")
        assert response().cached is False
        cache.put(key, response())
        assert cache.get(key).cached is True

    def test_creates_its_directory_on_write(self, tmp_path):
        target = tmp_path / "jobs" / "vid" / ".llm_cache"
        assert not target.exists()
        LLMCache(target).put(make_key("p", "m"), response())
        assert target.is_dir()

    def test_survives_a_new_cache_instance(self, tmp_path):
        """Re-running a stage in a fresh process must still hit."""
        directory = tmp_path / ".llm_cache"
        key = make_key("p", "m")
        LLMCache(directory).put(key, response())
        assert LLMCache(directory).get(key) is not None

    def test_counts_hits_and_misses(self, tmp_path):
        cache = LLMCache(tmp_path / ".llm_cache")
        key = make_key("p", "m")
        cache.get(key)
        cache.put(key, response())
        cache.get(key)
        cache.get(key)
        assert cache.stats() == {"hits": 2, "misses": 1}

    def test_disabled_cache_never_reads_or_writes(self, tmp_path):
        directory = tmp_path / ".llm_cache"
        cache = LLMCache(directory, enabled=False)
        key = make_key("p", "m")
        assert cache.put(key, response()) is None
        assert cache.get(key) is None
        assert not directory.exists()

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        """A half-written file from a kill must not break the next run."""
        directory = tmp_path / ".llm_cache"
        directory.mkdir()
        key = make_key("p", "m")
        (directory / f"{key}.json").write_text('{"data": {"a"', encoding="utf-8")

        cache = LLMCache(directory)
        assert cache.get(key) is None
        # The bad entry is dropped so the next write can replace it cleanly.
        assert not (directory / f"{key}.json").exists()

    def test_stale_cache_version_is_a_miss(self, tmp_path):
        directory = tmp_path / ".llm_cache"
        directory.mkdir()
        key = make_key("p", "m")
        (directory / f"{key}.json").write_text(
            json.dumps({"cache_version": CACHE_VERSION - 1, "data": {"old": True}}),
            encoding="utf-8",
        )
        assert LLMCache(directory).get(key) is None

    def test_writes_are_atomic(self, tmp_path):
        """No .tmp files may survive a completed write."""
        directory = tmp_path / ".llm_cache"
        cache = LLMCache(directory)
        for i in range(5):
            cache.put(make_key(f"p{i}", "m"), response())
        leftovers = [p.name for p in directory.iterdir() if p.name.startswith(".tmp")]
        assert leftovers == []

    def test_overwrite_replaces_the_entry(self, tmp_path):
        cache = LLMCache(tmp_path / ".llm_cache")
        key = make_key("p", "m")
        cache.put(key, response(data={"v": 1}))
        cache.put(key, response(data={"v": 2}))
        assert cache.get(key).data == {"v": 2}

    def test_stored_file_is_readable_json(self, tmp_path):
        """The cache is meant to be inspectable with cat, like the state file."""
        directory = tmp_path / ".llm_cache"
        cache = LLMCache(directory)
        key = make_key("p", "m")
        cache.put(key, response())

        raw = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
        assert raw["provider"] == "gemini"
        assert raw["tier"] == "recall"
        assert raw["cache_version"] == CACHE_VERSION
        assert "cached_at" in raw

    def test_non_ascii_survives_a_roundtrip(self, tmp_path):
        cache = LLMCache(tmp_path / ".llm_cache")
        key = make_key("p", "m")
        payload = {"quote": "it's a \u2014 dash, caf\u00e9, \u65e5\u672c\u8a9e"}
        cache.put(key, response(data=payload))
        assert cache.get(key).data == payload

    def test_clear_removes_everything(self, tmp_path):
        cache = LLMCache(tmp_path / ".llm_cache")
        for i in range(3):
            cache.put(make_key(f"p{i}", "m"), response())
        assert cache.clear() == 3
        assert cache.get(make_key("p0", "m")) is None

    def test_clear_on_missing_directory_is_a_noop(self, tmp_path):
        assert LLMCache(tmp_path / "nope").clear() == 0

    def test_rerunning_a_stage_does_not_respend_quota(self, tmp_path):
        """The reason this cache exists (free tiers, development reruns)."""
        directory = tmp_path / "jobs" / "vid123" / ".llm_cache"
        calls = {"n": 0}

        def expensive_call(prompt: str, model: str) -> LLMResponse:
            cache = LLMCache(directory)
            key = make_key(prompt, model)
            hit = cache.get(key)
            if hit is not None:
                return hit
            calls["n"] += 1
            result = response(data={"answer": calls["n"]})
            cache.put(key, result)
            return result

        first = expensive_call("transcript chunk 1", "gemini-3.5-flash-lite")
        second = expensive_call("transcript chunk 1", "gemini-3.5-flash-lite")

        assert calls["n"] == 1
        assert second.data == first.data
        assert second.cached is True

        # A different chunk is genuinely a new call.
        expensive_call("transcript chunk 2", "gemini-3.5-flash-lite")
        assert calls["n"] == 2
