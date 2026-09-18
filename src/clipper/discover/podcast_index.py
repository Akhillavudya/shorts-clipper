"""Podcast Index API client - RSS-side discovery (FULL_PLAN.md section 3).

Responsibility: supplement YouTube discovery with the free Podcast Index API,
which has no per-day quota worth worrying about. Catches episodes whose
YouTube upload lags the RSS feed.

Taddy and Listen Notes are the paid alternatives if transcripts or chart data
are ever needed; not required for MVP.

Auth is four headers, computed per request:

    X-Auth-Key    the API key
    X-Auth-Date   unix seconds
    Authorization sha1(key + secret + date), hex
    User-Agent    required; the API rejects requests without one

The date is part of the signature, so a cached header block goes stale. The
signature is recomputed on every call for that reason, not out of caution.

What this feed can and cannot do for the pipeline: it finds *episodes*, and an
episode's enclosure is audio. The pipeline needs video to clip, so an episode
found here is a lead - it tells you an episode exists before the YouTube upload
appears, and the ``link`` field usually points at the episode page where the
video lives. Nothing here bypasses the rights gate: an episode surfaced from
RSS still has to resolve to a watchlist channel before anything renders.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

API_BASE = "https://api.podcastindex.org/api/1.0"

#: The API rejects a request with no User-Agent. Naming the project is also
#: what their docs ask callers to do.
USER_AGENT = "shorts-clipper/0.1"


class PodcastIndexError(RuntimeError):
    """The API refused, or credentials are missing."""


@dataclass(frozen=True)
class Episode:
    """One episode from the index."""

    episode_id: str
    feed_id: str | None = None
    feed_title: str | None = None
    title: str | None = None
    description: str | None = None
    published_at: datetime | None = None
    duration_s: float | None = None
    enclosure_url: str | None = None
    link: str | None = None
    explicit: bool = False

    def age_hours(self, now: datetime | None = None) -> float | None:
        if self.published_at is None:
            return None
        now = now or datetime.now(UTC)
        return max(0.0, (now - self.published_at).total_seconds() / 3600.0)

    @classmethod
    def from_api(cls, item: dict[str, Any]) -> Episode:
        published = item.get("datePublished")
        when: datetime | None = None
        if isinstance(published, (int, float)) and published > 0:
            when = datetime.fromtimestamp(float(published), tz=UTC)

        duration = item.get("duration")
        try:
            duration_s = float(duration) if duration not in (None, "") else None
        except (TypeError, ValueError):
            duration_s = None

        return cls(
            episode_id=str(item.get("id") or ""),
            feed_id=str(item.get("feedId")) if item.get("feedId") is not None else None,
            feed_title=item.get("feedTitle"),
            title=item.get("title"),
            description=item.get("description"),
            published_at=when,
            duration_s=duration_s,
            enclosure_url=item.get("enclosureUrl"),
            link=item.get("link"),
            explicit=bool(item.get("explicit")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "feed_id": self.feed_id,
            "feed_title": self.feed_title,
            "title": self.title,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "duration_s": self.duration_s,
            "link": self.link,
            "enclosure_url": self.enclosure_url,
        }


def auth_headers(key: str, secret: str, *, now: int | None = None) -> dict[str, str]:
    """The four headers every Podcast Index request needs.

    Exposed separately because the signature is the only part worth testing, and
    testing it through an HTTP call would prove less about it.
    """
    stamp = str(int(now if now is not None else time.time()))
    digest = hashlib.sha1(f"{key}{secret}{stamp}".encode()).hexdigest()  # noqa: S324
    return {
        "X-Auth-Key": key,
        "X-Auth-Date": stamp,
        "Authorization": digest,
        "User-Agent": USER_AGENT,
    }


class PodcastIndexClient:
    """Read-only client for the free tier of the Podcast Index API.

    No quota ledger here, deliberately: the index has no per-day budget of the
    kind that shapes ``youtube.py``, so counting units would be ceremony. It is
    still polite to keep the scan small, which the callers do by asking for a
    bounded ``limit``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        key_env: str = "PODCAST_INDEX_KEY",
        secret_env: str = "PODCAST_INDEX_SECRET",
        base_url: str = API_BASE,
        client: httpx.Client | None = None,
        timeout_s: float = 30.0,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._key = api_key or os.environ.get(key_env) or ""
        self._secret = api_secret or os.environ.get(secret_env) or ""
        self._key_env = key_env
        self._secret_env = secret_env
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.timeout_s = timeout_s
        self._time = time_fn

    @property
    def configured(self) -> bool:
        """Whether credentials are present. Discovery treats this as optional."""
        return bool(self._key and self._secret)

    def _ensure_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        if not self.configured:
            raise PodcastIndexError(
                f"{self._key_env} and {self._secret_env} must both be set to use "
                f"Podcast Index. Keys are free at podcastindex.org/api-keys."
            )
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        client = self._ensure_client()
        headers = auth_headers(self._key, self._secret, now=int(self._time()))
        try:
            resp = client.get(path, params=params or {}, headers=headers, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise PodcastIndexError(f"podcast index {path} failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise PodcastIndexError(
                f"podcast index rejected the signature ({resp.status_code}). The "
                f"signed timestamp must be close to real time - check the system "
                f"clock: {resp.text[:200]}"
            )
        if resp.status_code >= 400:
            raise PodcastIndexError(f"podcast index {path} {resp.status_code}: {resp.text[:200]}")

        return resp.json()

    # -- endpoints ------------------------------------------------------

    def recent_episodes(self, *, limit: int = 40, since: datetime | None = None) -> list[Episode]:
        """Newest episodes across the whole index."""
        params: dict[str, Any] = {"max": max(1, int(limit))}
        if since is not None:
            params["since"] = int(since.timestamp())
        payload = self.get("/recent/episodes", params)
        return [Episode.from_api(item) for item in payload.get("items") or []]

    def episodes_by_feed(self, feed_id: str | int, *, limit: int = 20) -> list[Episode]:
        """Episodes from one feed, newest first."""
        payload = self.get("/episodes/byfeedid", {"id": str(feed_id), "max": max(1, int(limit))})
        return [Episode.from_api(item) for item in payload.get("items") or []]

    def search_feeds(self, term: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Feed search, for finding the ``feed_id`` to watch.

        Interactive only - you run it once to look a show up and then record the
        feed id. It is not part of the daily scan, which walks known feeds.
        """
        payload = self.get("/search/byterm", {"q": term, "max": max(1, int(limit))})
        return list(payload.get("feeds") or [])

    def scan_feeds(
        self,
        feed_ids: list[str | int],
        *,
        per_feed: int = 5,
        min_duration_s: float = 0.0,
        max_age_hours: float | None = 72.0,
        now: datetime | None = None,
    ) -> list[Episode]:
        """Recent episodes across the watched feeds, filtered the way the
        YouTube side filters uploads.

        Same two filters as the video scan - long enough to clip, recent enough
        to matter - so the two halves of discovery produce comparable lists.
        """
        now = now or datetime.now(UTC)
        found: list[Episode] = []

        for feed_id in feed_ids:
            for episode in self.episodes_by_feed(feed_id, limit=per_feed):
                if min_duration_s and (episode.duration_s or 0) < min_duration_s:
                    continue
                age = episode.age_hours(now)
                if max_age_hours is not None and (age is None or age > max_age_hours):
                    continue
                found.append(episode)

        found.sort(key=lambda e: e.published_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return found

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


__all__ = [
    "API_BASE",
    "USER_AGENT",
    "Episode",
    "PodcastIndexClient",
    "PodcastIndexError",
    "auth_headers",
]
