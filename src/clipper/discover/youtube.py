"""Quota-aware YouTube Data API v3 client (FULL_PLAN.md section 3).

Responsibility: turn a watchlist of channels into a list of recent video
records, spending as little quota as possible.

The quota shape is the whole design constraint: 10,000 units/day per Cloud
project, `search.list` costs 100 units, `playlistItems.list` costs 1. So:

1. Resolve each channel's uploads playlist ID once (`channels.list`, 1 unit)
   and cache it forever.
2. Daily, walk `playlistItems.list` per uploads playlist (1 unit each).
3. Batch the resulting video IDs into `videos.list` 50 at a time (1 unit) for
   view counts and durations.

Never call `search.list` on a schedule. Track spend per run and refuse to
exceed the daily ceiling configured in settings.yaml.

Two things are enforced here rather than left to discipline:

- :meth:`YouTubeClient.get` refuses the ``search`` resource outright. CLAUDE.md
  hard rule 3 is the kind of rule broken by a well-meaning one-line change six
  months from now, so it is a raised exception rather than a comment.
- The ledger is persisted and keyed on the UTC date, so units spent by an
  earlier run today still count. A per-process counter would reset on every
  invocation and the ceiling would mean nothing.

This talks HTTP with ``httpx`` rather than ``google-api-python-client``. The
surface used here is four read-only endpoints, the repo already speaks httpx in
``llm/groq.py`` and ``llm/ollama.py``, and a plain client is something a test
can fake with ``httpx.MockTransport`` without a network round trip to fetch a
discovery document first.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

API_BASE = "https://www.googleapis.com/youtube/v3"

#: Quota cost in units per endpoint, from the YouTube Data API quota table.
#: ``search`` is listed so the refusal can quote the number it is saving.
QUOTA_COST: dict[str, int] = {
    "channels": 1,
    "playlistItems": 1,
    "videos": 1,
    "search": 100,
}

#: ``videos.list`` accepts up to 50 ids per call and costs the same 1 unit for
#: one id as for fifty. Batching is free money.
MAX_IDS_PER_CALL = 50

#: ``playlistItems.list`` page size ceiling.
MAX_PAGE_SIZE = 50

CACHE_FILENAME = "uploads_playlists.json"
LEDGER_FILENAME = "quota.json"

_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class DiscoveryError(RuntimeError):
    """The API refused, or was asked for something discovery must not ask for."""


class QuotaExceeded(DiscoveryError):
    """The configured daily ceiling would be passed by this call."""


def parse_duration(raw: str | None) -> float | None:
    """ISO 8601 ``PT1H2M3S`` -> seconds.

    YouTube reports durations in this format and nothing else in the pipeline
    speaks it. A live stream reports ``P0D``, which parses to 0 - correct, and
    the duration filter then drops it.
    """
    if not raw:
        return None
    match = _ISO_DURATION.match(raw.strip())
    if not match:
        return None
    parts = {k: int(v) for k, v in match.groupdict().items() if v}
    return float(
        parts.get("days", 0) * 86400
        + parts.get("hours", 0) * 3600
        + parts.get("minutes", 0) * 60
        + parts.get("seconds", 0)
    )


def parse_timestamp(raw: str | None) -> datetime | None:
    """RFC 3339 as the API emits it, normalized to aware UTC."""
    if not raw:
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _int(value: Any) -> int | None:
    """Statistics come back as strings, and are absent when a channel hides them."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class VideoRecord:
    """One upload, as much as ``videos.list`` will tell us about it.

    ``view_count`` is ``None``, not 0, when a channel hides its stats. Velocity
    scoring skips those rather than treating them as zero-view flops.
    """

    video_id: str
    channel_id: str | None = None
    channel_title: str | None = None
    title: str | None = None
    description: str | None = None
    published_at: datetime | None = None
    duration_s: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    licence: str | None = None
    live: str | None = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def is_live(self) -> bool:
        """Live or upcoming. Never clippable, and its stats mean nothing yet."""
        return self.live in {"live", "upcoming"}

    def age_hours(self, now: datetime | None = None) -> float | None:
        if self.published_at is None:
            return None
        now = now or datetime.now(UTC)
        return max(0.0, (now - self.published_at).total_seconds() / 3600.0)

    @classmethod
    def from_api(cls, item: dict[str, Any]) -> VideoRecord:
        snippet = item.get("snippet") or {}
        details = item.get("contentDetails") or {}
        stats = item.get("statistics") or {}
        status = item.get("status") or {}
        return cls(
            video_id=str(item.get("id") or ""),
            channel_id=snippet.get("channelId"),
            channel_title=snippet.get("channelTitle"),
            title=snippet.get("title"),
            description=snippet.get("description"),
            published_at=parse_timestamp(snippet.get("publishedAt")),
            duration_s=parse_duration(details.get("duration")),
            view_count=_int(stats.get("viewCount")),
            like_count=_int(stats.get("likeCount")),
            comment_count=_int(stats.get("commentCount")),
            licence=status.get("license"),
            live=snippet.get("liveBroadcastContent"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "channel_id": self.channel_id,
            "channel_title": self.channel_title,
            "title": self.title,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "duration_s": self.duration_s,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "licence": self.licence,
        }


class QuotaLedger:
    """Units spent today, persisted and keyed on the UTC date.

    Persisted because the ceiling is a *daily* budget and a scan may run more
    than once a day. An in-memory counter would reset at every invocation, so
    the ceiling would only catch a runaway loop inside one run - which is the
    single failure mode a watchlist-driven design already prevents.

    A ceiling of 0 disables the check, matching how ``llm/limiter.py`` reads a
    limit of 0.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        ceiling: int = 2000,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.ceiling = int(ceiling)
        self._now = now_fn or (lambda: datetime.now(UTC))
        self.spent_this_run = 0
        self.calls: dict[str, int] = {}
        self._day, self._used = self._read()

    def _read(self) -> tuple[date, int]:
        today = self._now().date()
        if self.path is None or not self.path.is_file():
            return today, 0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt ledger means the count is unknown. Starting from zero
            # risks overspending once; refusing to run risks never running
            # again. Against a 10,000-unit budget that a scan uses ~100 of,
            # overspending once is the cheaper mistake.
            return today, 0
        if data.get("day") != today.isoformat():
            return today, 0
        return today, int(data.get("used") or 0)

    def _write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"day": self._day.isoformat(), "used": self._used}, indent=2),
            encoding="utf-8",
        )

    def _roll(self) -> None:
        today = self._now().date()
        if today != self._day:
            self._day = today
            self._used = 0

    @property
    def used(self) -> int:
        self._roll()
        return self._used

    @property
    def remaining(self) -> int:
        if self.ceiling <= 0:
            return 2**31
        return max(0, self.ceiling - self.used)

    def check(self, units: int) -> None:
        """Raise if spending ``units`` would pass the ceiling."""
        if self.ceiling <= 0:
            return
        if self.used + units > self.ceiling:
            raise QuotaExceeded(
                f"discovery has spent {self.used} of {self.ceiling} quota units "
                f"today and the next call costs {units}. Raise "
                f"discovery.daily_quota_ceiling in config/settings.yaml, or wait "
                f"for the UTC-midnight reset. A watchlist scan should cost about "
                f"one unit per channel - if it is costing more, something is "
                f"calling search.list."
            )

    def spend(self, units: int, *, resource: str = "") -> None:
        self.check(units)
        self._roll()
        self._used += units
        self.spent_this_run += units
        if resource:
            self.calls[resource] = self.calls.get(resource, 0) + 1
        self._write()

    def snapshot(self) -> dict[str, Any]:
        return {
            "day": self._day.isoformat(),
            "used_today": self.used,
            "spent_this_run": self.spent_this_run,
            "ceiling": self.ceiling,
            "remaining": self.remaining,
            "calls": dict(self.calls),
        }


class PlaylistCache:
    """channel id -> uploads playlist id, cached forever.

    A channel's uploads playlist id never changes, so this is the one lookup in
    the pipeline safe to cache with no expiry. 80 channels costs 80 units the
    first time and nothing every day after.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, str] = {}
        if self.path and self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = {str(k): str(v) for k, v in loaded.items()}
            except (OSError, json.JSONDecodeError, AttributeError):
                self._data = {}

    def get(self, channel_id: str) -> str | None:
        return self._data.get(channel_id)

    def put(self, channel_id: str, playlist_id: str) -> None:
        self._data[channel_id] = playlist_id
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class ScanResult:
    """What one watchlist scan found, plus what it cost."""

    records: list[VideoRecord] = field(default_factory=list)
    quota: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class YouTubeClient:
    """Read-only Data API v3 client that counts every unit it spends."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "YOUTUBE_API_KEY",
        base_url: str = API_BASE,
        client: httpx.Client | None = None,
        ledger: QuotaLedger | None = None,
        playlist_cache: PlaylistCache | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get(api_key_env)
        self._api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.timeout_s = timeout_s
        self.ledger = ledger if ledger is not None else QuotaLedger(ceiling=0)
        self.playlists = playlist_cache if playlist_cache is not None else PlaylistCache()

    # -- transport ------------------------------------------------------

    def _ensure_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise DiscoveryError(
                f"{self._api_key_env} is not set; discovery cannot call the "
                f"YouTube Data API. Create a key in the Google Cloud console "
                f"and put it in .env."
            )
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    def get(self, resource: str, params: dict[str, Any]) -> dict[str, Any]:
        """One API call, charged to the ledger before it is made.

        Charging first means a call refused for quota is never made, and a call
        that fails still counts - Google charges for a 4xx the same as a 200, so
        a ledger that only counted successes would drift low.
        """
        if resource == "search":
            raise DiscoveryError(
                "search.list costs 100 quota units against a 10,000/day budget, "
                "and discovery is watchlist-driven for exactly that reason "
                "(CLAUDE.md hard rule 3). Use playlistItems.list - 1 unit."
            )

        cost = QUOTA_COST.get(resource)
        if cost is None:
            raise DiscoveryError(
                f"unknown YouTube resource {resource!r}; refusing to guess what it costs in quota"
            )

        client = self._ensure_client()
        self.ledger.spend(cost, resource=resource)

        query = dict(params)
        if self._api_key:
            query["key"] = self._api_key

        try:
            resp = client.get(f"/{resource}", params=query, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"youtube {resource}.list failed: {exc}") from exc

        if resp.status_code == 403:
            raise DiscoveryError(
                f"youtube {resource}.list returned 403 - usually the daily quota "
                f"is exhausted or the key is restricted: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise DiscoveryError(f"youtube {resource}.list {resp.status_code}: {resp.text[:300]}")

        return resp.json()

    # -- endpoints ------------------------------------------------------

    def uploads_playlist_id(self, channel_id: str) -> str:
        """The channel's uploads playlist. 1 unit, then cached forever."""
        cached = self.playlists.get(channel_id)
        if cached:
            return cached

        payload = self.get("channels", {"part": "contentDetails", "id": channel_id})
        items = payload.get("items") or []
        if not items:
            raise DiscoveryError(
                f"channel {channel_id!r} returned no items - check the id in "
                f"config/watchlist.yaml. It must be the UC... channel id, not a "
                f"@handle or a custom URL."
            )
        try:
            playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except (KeyError, TypeError) as exc:
            raise DiscoveryError(f"channel {channel_id!r} has no uploads playlist") from exc

        self.playlists.put(channel_id, str(playlist_id))
        return str(playlist_id)

    def recent_upload_ids(self, playlist_id: str, limit: int = 10) -> list[str]:
        """Newest video ids from an uploads playlist. 1 unit per page of 50."""
        wanted = max(0, int(limit))
        ids: list[str] = []
        page_token: str | None = None

        while len(ids) < wanted:
            params: dict[str, Any] = {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": min(MAX_PAGE_SIZE, wanted - len(ids)),
            }
            if page_token:
                params["pageToken"] = page_token

            payload = self.get("playlistItems", params)
            items = payload.get("items") or []
            for item in items:
                video_id = (item.get("contentDetails") or {}).get("videoId") or ""
                if video_id:
                    ids.append(str(video_id))

            page_token = payload.get("nextPageToken")
            if not page_token or not items:
                break

        return ids[:wanted]

    def videos(self, video_ids: list[str]) -> list[VideoRecord]:
        """Stats and durations, batched 50 per unit."""
        unique = list(dict.fromkeys(v for v in video_ids if v))
        records: list[VideoRecord] = []

        for start in range(0, len(unique), MAX_IDS_PER_CALL):
            batch = unique[start : start + MAX_IDS_PER_CALL]
            payload = self.get(
                "videos",
                {
                    "part": "snippet,contentDetails,statistics,status",
                    "id": ",".join(batch),
                    "maxResults": MAX_IDS_PER_CALL,
                },
            )
            records.extend(VideoRecord.from_api(item) for item in payload.get("items") or [])

        return records

    def most_popular(self, *, region_code: str = "US", limit: int = 25) -> list[VideoRecord]:
        """``videos.list?chart=mostPopular`` - 1 unit (section 3, step 6).

        Supplementary, not the backbone: what trends nationally is rarely the
        long-form talking-head material this pipeline clips well. It is one unit
        for a read on what the moment is, which is worth having.
        """
        payload = self.get(
            "videos",
            {
                "part": "snippet,contentDetails,statistics,status",
                "chart": "mostPopular",
                "regionCode": region_code,
                "maxResults": min(MAX_PAGE_SIZE, max(1, int(limit))),
            },
        )
        return [VideoRecord.from_api(item) for item in payload.get("items") or []]

    # -- the scan -------------------------------------------------------

    def scan_channels(
        self,
        channel_ids: list[str],
        *,
        per_channel: int = 5,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> ScanResult:
        """Walk a watchlist: uploads playlist, recent ids, then one stats batch.

        A channel that fails is recorded and skipped rather than aborting the
        scan - one dead id in an 80-channel watchlist should not cost the other
        79, and the error is reported so the entry gets fixed. Quota exhaustion
        is the exception: that is not this channel's problem, it is the run's.
        """
        result = ScanResult()
        pending: list[str] = []

        for channel_id in channel_ids:
            try:
                playlist_id = self.uploads_playlist_id(channel_id)
                ids = self.recent_upload_ids(playlist_id, limit=per_channel)
            except QuotaExceeded:
                raise
            except DiscoveryError as exc:
                result.errors[channel_id] = str(exc)
                continue
            pending.extend(ids)
            if on_progress is not None:
                on_progress(channel_id, len(ids))

        if pending:
            result.records = self.videos(pending)

        result.quota = self.ledger.snapshot()
        return result

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def cache_dir(root: Path | str = "jobs") -> Path:
    """Where discovery keeps its ledger, playlist cache, and LLM cache.

    Under ``jobs/`` and dot-prefixed: discovery is not a job, but its artifacts
    are run state rather than config, and ``jobs/`` is already gitignored.
    """
    return Path(root) / ".discover"


def ledger_for(root: Path | str = "jobs", *, ceiling: int = 2000) -> QuotaLedger:
    return QuotaLedger(cache_dir(root) / LEDGER_FILENAME, ceiling=ceiling)


def playlist_cache_for(root: Path | str = "jobs") -> PlaylistCache:
    return PlaylistCache(cache_dir(root) / CACHE_FILENAME)


__all__ = [
    "API_BASE",
    "CACHE_FILENAME",
    "LEDGER_FILENAME",
    "MAX_IDS_PER_CALL",
    "MAX_PAGE_SIZE",
    "QUOTA_COST",
    "DiscoveryError",
    "PlaylistCache",
    "QuotaExceeded",
    "QuotaLedger",
    "ScanResult",
    "VideoRecord",
    "YouTubeClient",
    "cache_dir",
    "ledger_for",
    "parse_duration",
    "parse_timestamp",
    "playlist_cache_for",
]
