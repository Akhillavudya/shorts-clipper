"""Discovery end to end, with the API and the model faked.

This is where the rights tiers meet discovery. A red channel can never be
rendered, so shortlisting one is an invitation to override the gate - the
default is to drop it, and ``--include-blocked`` is the deliberate opt-in.
"""

from __future__ import annotations

import json
import textwrap
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from clipper import pipeline
from clipper.discover.youtube import ScanResult, VideoRecord

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        textwrap.dedent(
            """
            paths:
              jobs_root: jobs
            tiers:
              recall:
                provider: gemini
                model: test-model
            discovery:
              daily_quota_ceiling: 2000
              max_candidates: 3
              min_video_age_hours: 6
              min_channel_uploads: 2
            """
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def watchlist(tmp_path):
    path = tmp_path / "watchlist.yaml"
    path.write_text(
        textwrap.dedent(
            """
            defaults:
              max_per_day: 2
              min_duration_s: 900
            channels:
              - id: UC_GREEN
                name: Green Channel
                rights_status: green
                note: Public clipping policy.
              - id: UC_RED
                name: Red Channel
                rights_status: red
                note: Runs its own clips channel.
            """
        ),
        encoding="utf-8",
    )
    return path


def record(video_id: str, channel: str, *, views: int, age_hours: float = 24.0) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        channel_id=channel,
        channel_title=channel,
        title=f"Episode {video_id}",
        description="An interview.",
        published_at=NOW - timedelta(hours=age_hours),
        duration_s=5400.0,
        view_count=views,
    )


class FakeYouTube:
    """Stands in for YouTubeClient. Records which channels it was asked for."""

    def __init__(self, by_channel: dict[str, list[VideoRecord]]) -> None:
        self.by_channel = by_channel
        self.asked: list[str] = []

    def scan_channels(self, channel_ids, *, per_channel=5, on_progress=None):
        self.asked = list(channel_ids)
        records: list[VideoRecord] = []
        for channel_id in channel_ids:
            found = self.by_channel.get(channel_id, [])
            records.extend(found)
            if on_progress:
                on_progress(channel_id, len(found))
        return ScanResult(
            records=records,
            quota={"spent_this_run": len(channel_ids) * 2 + 1, "used_today": 7, "ceiling": 2000},
        )


def ctx_for(settings, tmp_path) -> pipeline.Context:
    return pipeline.Context.build(config_path=settings, jobs_root=tmp_path / "jobs")


class TestDiscover:
    def test_red_channels_are_not_scanned_by_default(self, settings, watchlist, tmp_path) -> None:
        youtube = FakeYouTube(
            {
                "UC_GREEN": [
                    record("g1", "UC_GREEN", views=1000),
                    record("g2", "UC_GREEN", views=1000),
                    record("g3", "UC_GREEN", views=9000),
                ]
            }
        )
        result = pipeline.discover(
            ctx_for(settings, tmp_path),
            watchlist_path=watchlist,
            youtube=youtube,
            summarize=False,
            now=NOW,
        )

        assert youtube.asked == ["UC_GREEN"], "a red channel costs quota for nothing"
        assert result.blocked_hidden == 1
        assert all(c.rights_status == "green" for c in result.candidates)

    def test_include_blocked_scans_them_and_labels_them(
        self, settings, watchlist, tmp_path
    ) -> None:
        youtube = FakeYouTube(
            {
                "UC_GREEN": [record(f"g{i}", "UC_GREEN", views=1000 + i) for i in range(3)],
                "UC_RED": [record(f"r{i}", "UC_RED", views=90_000 + i) for i in range(3)],
            }
        )
        result = pipeline.discover(
            ctx_for(settings, tmp_path),
            watchlist_path=watchlist,
            youtube=youtube,
            summarize=False,
            include_blocked=True,
            now=NOW,
        )

        assert set(youtube.asked) == {"UC_GREEN", "UC_RED"}
        assert {c.rights_status for c in result.candidates} == {"green", "red"}

    def test_the_shortlist_is_capped_by_max_candidates(self, settings, watchlist, tmp_path) -> None:
        youtube = FakeYouTube(
            {"UC_GREEN": [record(f"g{i}", "UC_GREEN", views=1000 + i * 100) for i in range(8)]}
        )
        result = pipeline.discover(
            ctx_for(settings, tmp_path),
            watchlist_path=watchlist,
            youtube=youtube,
            summarize=False,
            now=NOW,
        )
        # max_candidates is 3 in the fixture, but max_per_day caps this channel
        # at 2 first - one creator must not fill the shortlist.
        assert len(result.candidates) == 2

    def test_an_empty_watchlist_explains_itself(self, settings, tmp_path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("channels: []\n", encoding="utf-8")

        result = pipeline.discover(
            ctx_for(settings, tmp_path), watchlist_path=empty, summarize=False, now=NOW
        )

        assert result.candidates == []
        assert "watchlist" in result.errors

    def test_the_report_lands_under_the_jobs_root(self, settings, watchlist, tmp_path) -> None:
        youtube = FakeYouTube(
            {"UC_GREEN": [record(f"g{i}", "UC_GREEN", views=1000 + i * 500) for i in range(4)]}
        )
        result = pipeline.discover(
            ctx_for(settings, tmp_path),
            watchlist_path=watchlist,
            youtube=youtube,
            summarize=False,
            now=NOW,
        )

        assert result.report_path is not None
        assert result.report_path.parent == tmp_path / "jobs" / ".discover"

        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        assert payload["quota"]["spent_this_run"] == 3
        assert len(payload["candidates"]) == len(result.candidates)

    def test_a_failing_summarize_does_not_lose_the_shortlist(
        self, settings, watchlist, tmp_path, monkeypatch
    ) -> None:
        # The blurb is optional; the ranked list is the deliverable. A dead API
        # key at the last step must not throw away the quota already spent.
        from clipper.discover import summarize as summarize_module

        def boom(*args: Any, **kwargs: Any):
            raise RuntimeError("no API key")

        monkeypatch.setattr(summarize_module, "run", boom)

        youtube = FakeYouTube(
            {"UC_GREEN": [record(f"g{i}", "UC_GREEN", views=1000 + i * 500) for i in range(4)]}
        )
        result = pipeline.discover(
            ctx_for(settings, tmp_path),
            watchlist_path=watchlist,
            youtube=youtube,
            summarize=True,
            now=NOW,
        )

        assert len(result.candidates) == 2
        assert "summarize" in result.errors

    def test_too_short_videos_never_reach_the_shortlist(
        self, settings, watchlist, tmp_path
    ) -> None:
        short = [
            VideoRecord(
                video_id=f"s{i}",
                channel_id="UC_GREEN",
                title=f"short {i}",
                published_at=NOW - timedelta(hours=24),
                duration_s=120.0,
                view_count=100_000,
            )
            for i in range(4)
        ]
        result = pipeline.discover(
            ctx_for(settings, tmp_path),
            watchlist_path=watchlist,
            youtube=FakeYouTube({"UC_GREEN": short}),
            summarize=False,
            now=NOW,
        )

        assert result.candidates == []
        assert result.skipped["too_short"] == 4
