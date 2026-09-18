"""Velocity scoring - the deterministic half of discovery.

The interesting cases are the ones where a naive implementation gives a
plausible-looking wrong answer: a breakout video inflating the baseline it is
being measured against, a channel that hides its view count scoring zero, and
a two-hour-old upload looking like a phenomenon because notifications went out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from clipper.discover.velocity import (
    cap_per_channel,
    channel_baseline,
    classify,
    score_records,
    summarize_skips,
    views_per_hour,
)
from clipper.discover.youtube import VideoRecord

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def record(
    video_id: str,
    *,
    views: int | None = 1000,
    age_hours: float = 48.0,
    channel: str = "UC_A",
    duration_s: float = 3600.0,
    live: str | None = None,
) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        channel_id=channel,
        channel_title=channel,
        title=video_id,
        published_at=NOW - timedelta(hours=age_hours),
        duration_s=duration_s,
        view_count=views,
        live=live,
    )


class TestViewsPerHour:
    def test_simple_arithmetic(self) -> None:
        assert views_per_hour(record("v", views=1000, age_hours=10), now=NOW) == 100.0

    def test_too_new_is_none(self) -> None:
        assert views_per_hour(record("v", age_hours=2), now=NOW, min_age_hours=6) is None

    def test_hidden_views_is_none(self) -> None:
        assert views_per_hour(record("v", views=None), now=NOW) is None


class TestClassify:
    def test_a_scorable_record_classifies_as_none(self) -> None:
        assert classify(record("v"), now=NOW) is None

    def test_reasons_are_distinguished(self) -> None:
        assert classify(record("v", age_hours=1), now=NOW) == "too_new"
        assert classify(record("v", views=None), now=NOW) == "no_views"
        assert classify(record("v", live="live"), now=NOW) == "live"
        assert classify(record("v", duration_s=120), now=NOW, min_duration_s=900) == "too_short"


class TestBaseline:
    def test_median_not_mean(self) -> None:
        # 10, 10, 10 vph and one outlier at 1000. A mean would be ~257.
        records = [
            record("a", views=100, age_hours=10),
            record("b", views=100, age_hours=10),
            record("c", views=100, age_hours=10),
            record("viral", views=10_000, age_hours=10),
        ]
        baseline = channel_baseline(records, now=NOW, min_uploads=3)
        assert baseline.median_vph == 10.0

    def test_the_scored_video_is_excluded_from_its_own_baseline(self) -> None:
        # One ordinary upload at 10 vph and one breakout at 1000. Including the
        # breakout in its own baseline makes the median 505 and the breakout
        # scores 1.98x - ordinary. Excluding it, the baseline is 10 and the
        # breakout scores 100x, which is the truth. Small samples are where
        # this bias bites hardest, and a young watchlist is all small samples.
        records = [
            record("a", views=100, age_hours=10),
            record("breakout", views=10_000, age_hours=10),
        ]
        with_self = channel_baseline(records, now=NOW, min_uploads=1)
        without_self = channel_baseline(
            records, now=NOW, exclude_video_id="breakout", min_uploads=1
        )
        assert with_self.median_vph == 505.0
        assert without_self.median_vph == 10.0

        scored, _ = score_records(records, now=NOW, min_channel_uploads=1)
        assert scored[0].video_id == "breakout"
        assert scored[0].velocity == 100.0

    def test_thin_sample_is_untrusted_but_still_a_number(self) -> None:
        baseline = channel_baseline([record("a", views=100, age_hours=10)], now=NOW, min_uploads=5)
        assert baseline.sample_size == 1
        assert baseline.trusted is False
        assert baseline.median_vph == 10.0

    def test_no_usable_samples_gives_zero_and_untrusted(self) -> None:
        baseline = channel_baseline([record("a", views=None)], now=NOW)
        assert baseline.median_vph == 0.0
        assert baseline.trusted is False


class TestScoreRecords:
    def test_the_outlier_ranks_first(self) -> None:
        records = [
            record("ordinary_1", views=1000, age_hours=10),
            record("ordinary_2", views=1000, age_hours=10),
            record("ordinary_3", views=1000, age_hours=10),
            record("breakout", views=5000, age_hours=10),
        ]
        scored, _ = score_records(records, now=NOW, min_channel_uploads=3)

        assert scored[0].video_id == "breakout"
        assert scored[0].velocity == 5.0
        assert scored[1].velocity == 1.0

    def test_channels_are_normalized_against_themselves(self) -> None:
        # A small channel doubling its median must outrank a huge channel
        # sitting exactly on its own median, despite far fewer raw views.
        small = [record(f"s{i}", views=100, age_hours=10, channel="UC_SMALL") for i in range(3)]
        small.append(record("s_hit", views=200, age_hours=10, channel="UC_SMALL"))
        big = [record(f"b{i}", views=1_000_000, age_hours=10, channel="UC_BIG") for i in range(4)]

        scored, _ = score_records(small + big, now=NOW, min_channel_uploads=3)

        assert scored[0].video_id == "s_hit"
        assert scored[0].record.view_count == 200

    def test_skips_are_reported_not_silently_dropped(self) -> None:
        records = [
            record("fine", views=1000, age_hours=10),
            record("fine2", views=1000, age_hours=10),
            record("fine3", views=1000, age_hours=10),
            record("new", age_hours=1),
            record("hidden", views=None),
            record("short", duration_s=60),
            record("streaming", live="live"),
        ]
        scored, skipped = score_records(records, now=NOW, min_channel_uploads=2, min_duration_s=900)

        assert {s.video_id for s in scored} == {"fine", "fine2", "fine3"}
        counts = summarize_skips(skipped)
        assert counts == {"too_new": 1, "no_views": 1, "too_short": 1, "live": 1}

    def test_a_channel_with_no_baseline_is_reported_not_ranked(self) -> None:
        scored, skipped = score_records([record("only", views=1000)], now=NOW)
        assert scored == []
        assert summarize_skips(skipped) == {"no_baseline": 1}

    def test_per_channel_duration_floor_overrides_the_global_one(self) -> None:
        records = [
            record(f"a{i}", views=1000, age_hours=10, duration_s=600, channel="UC_A")
            for i in range(3)
        ]
        scored, _ = score_records(
            records,
            now=NOW,
            min_channel_uploads=2,
            min_duration_s=900,
            per_channel_min_duration={"UC_A": 300},
        )
        assert len(scored) == 3


class TestCapPerChannel:
    def test_one_channel_cannot_fill_the_shortlist(self) -> None:
        records = [record(f"a{i}", views=1000 + i * 100, age_hours=10) for i in range(6)]
        scored, _ = score_records(records, now=NOW, min_channel_uploads=2)

        capped = cap_per_channel(scored, {"UC_A": 2}, default=2)

        assert len(capped) == 2
        assert capped[0].velocity >= capped[1].velocity, "the cap keeps the best, not the first"

    def test_the_default_applies_to_channels_with_no_entry(self) -> None:
        records = [record(f"a{i}", views=1000 + i, age_hours=10) for i in range(5)]
        scored, _ = score_records(records, now=NOW, min_channel_uploads=2)
        assert len(cap_per_channel(scored, {}, default=1)) == 1

    def test_a_limit_of_zero_drops_the_channel(self) -> None:
        records = [record(f"a{i}", views=1000 + i, age_hours=10) for i in range(3)]
        scored, _ = score_records(records, now=NOW, min_channel_uploads=2)
        assert cap_per_channel(scored, {"UC_A": 0}) == []
