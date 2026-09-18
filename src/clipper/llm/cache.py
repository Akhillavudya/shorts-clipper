"""On-disk response cache, keyed by the content of the request.

Re-running a stage during development must not respend quota. On a free tier
that is not an optimization - a rerun of the recall pass over a 2-hour
transcript can be a meaningful fraction of a day's requests.

Lives in ``jobs/<video_id>/.llm_cache/`` so it is scoped to one job, is
gitignored along with the rest of ``jobs/``, and is thrown away by deleting the
job directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clipper.llm.base import LLMResponse

CACHE_DIRNAME = ".llm_cache"
CACHE_VERSION = 1


def make_key(
    prompt: str,
    model: str,
    *,
    system: str | None = None,
    json_schema: dict[str, Any] | None = None,
    temperature: float | None = None,
) -> str:
    """Content hash identifying one request.

    Keyed on more than prompt+model on purpose. The schema is part of the key
    because changing the requested shape must not return a cached response in
    the old shape - that failure is silent and extremely annoying to track
    down. Temperature and system prompt are included for the same reason.
    """
    payload = json.dumps(
        {
            "v": CACHE_VERSION,
            "model": model,
            "system": system,
            "prompt": prompt,
            "schema": json_schema,
            "temperature": temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """Read/write cache of ``LLMResponse`` payloads, one JSON file per key."""

    def __init__(self, directory: Path | str, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        """Return the cached response, or None. Never raises on bad cache data."""
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt or half-written entry is a miss, not a crash. Drop it so
            # the next run rewrites it cleanly.
            path.unlink(missing_ok=True)
            self.misses += 1
            return None

        if raw.get("cache_version") != CACHE_VERSION:
            self.misses += 1
            return None

        self.hits += 1
        return LLMResponse(
            data=raw["data"],
            text=raw.get("text", ""),
            provider=raw.get("provider", "?"),
            model=raw.get("model", "?"),
            tokens_in=raw.get("tokens_in", 0),
            tokens_out=raw.get("tokens_out", 0),
            latency_s=raw.get("latency_s", 0.0),
            cached=True,
            tier=raw.get("tier"),
        )

    def put(self, key: str, response: LLMResponse) -> Path | None:
        """Write a response to the cache. Atomic, so a kill cannot truncate it."""
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        payload = json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "key": key,
                "cached_at": datetime.now(UTC).isoformat(),
                "provider": response.provider,
                "model": response.model,
                "tier": response.tier,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "latency_s": response.latency_s,
                "text": response.text,
                "data": response.data,
            },
            indent=2,
            ensure_ascii=False,
        )

        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=".tmp.", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed."""
        if not self.directory.is_dir():
            return 0
        removed = 0
        for path in self.directory.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
