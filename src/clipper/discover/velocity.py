"""Deterministic outlier scoring for discovery candidates (FULL_PLAN.md section 3).

Responsibility: rank freshly-published videos by how far they are beating their
own channel's baseline. No model involved - arithmetic on view counts is both
cheaper and better than an LLM here.

    velocity = views_per_hour_since_publish / channel_median_views_per_hour

Maintains the per-channel median so the score is normalized across channels of
very different sizes. Needs a minimum age (a few hours) before the score means
anything, and a minimum sample of prior uploads before the median is trusted.

Three details that decide whether the number is worth anything:

- **The baseline excludes the video being scored.** A breakout upload drags the
  median towards itself, and on a channel with a handful of uploads it can drag
  it far enough to score 1.0 and look ordinary. Scoring a video against its
  peers, not against a set containing itself, is the difference between an
  outlier detector and a slow-moving average.
- **Hidden stats are skipped, not zeroed.** ``view_count`` is ``None`` when a
  channel hides it; treating that as 0 views would rank every such video last
  and quietly remove the channel from discovery.
- **Age floor before scoring.** Views per hour in the first hour of an upload is
  dominated by subscriber notifications, not by the video. Below
  ``min_age_hours`` the score is not computed at all.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from clipper.discover.youtube import VideoRecord

#: Reasons a record produced no score. Reported rather than silently dropped,
#: because "no candidates today" and "every candidate was too new" are very
#: different things to read in a scan summary.
SKIP_REASONS = (
    "too_new",
    "no_views",
    "no_published_at",
    "live",
    "too_short",
    "no_baseline",
)


@dataclass(frozen=True)
class ChannelBaseline:
    """A channel's typical views-per-hour, and whether to believe it."""

    channel_id: str
    median_vph: float
    sample_size: int
    trusted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "median_vph": round(self.median_vph, 4),
            "sample_size": self.sample_size,
            "trusted": self.trusted,
        }


@dataclass(frozen=True)
class VelocityScore:
    """One scored candidate, with the arithmetic left visible.

    ``vph`` and ``baseline`` are kept alongside ``velocity`` so a surprising
    ranking can be read straight off the JSON rather than recomputed.
    """

    record: VideoRecord
    vph: float
    baseline: ChannelBaseline
    velocity: float
    age_hours: float

    @property
    def video_id(self) -> str:
        return self.record.video_id

    @property
    def trusted(self) -> bool:
        return self.baseline.trusted

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record.to_dict(),
            "age_hours": round(self.age_hours, 2),
            "views_per_hour": round(self.vph, 2),
            "velocity": round(self.velocity, 3),
            "baseline": self.baseline.to_dict(),
        }


@dataclass(frozen=True)
class Skipped:
    """A record that could not be scored, and why."""

    record: VideoRecord
    reason: str


def views_per_hour(
    record: VideoRecord,
    *,
    now: datetime | None = None,
    min_age_hours: float = 6.0,
) -> float | None:
    """Views per hour since publish, or ``None`` if the number is not yet real.

    ``None`` covers three distinct cases - too new, no publish date, hidden view
    count - and the caller distinguishes them via :func:`classify`. This
    function is the arithmetic only.
    """
    now = now or datetime.now(UTC)
    if record.published_at is None or record.view_count is None:
        return None
    age = (now - record.published_at).total_seconds() / 3600.0
    if age < min_age_hours or age <= 0:
        return None
    return record.view_count / age


def classify(
    record: VideoRecord,
    *,
    now: datetime | None = None,
    min_age_hours: float = 6.0,
    min_duration_s: float = 0.0,
) -> str | None:
    """Why this record cannot be scored, or ``None`` if it can."""
    now = now or datetime.now(UTC)
    if record.is_live:
        return "live"
    if record.published_at is None:
        return "no_published_at"
    if record.view_count is None:
        return "no_views"
    if min_duration_s and (record.duration_s or 0) < min_duration_s:
        return "too_short"
    age = (now - record.published_at).total_seconds() / 3600.0
    if age < min_age_hours or age <= 0:
        return "too_new"
    return None


def channel_baseline(
    records: list[VideoRecord],
    *,
    now: datetime | None = None,
    exclude_video_id: str | None = None,
    min_uploads: int = 5,
    min_age_hours: float = 6.0,
) -> ChannelBaseline:
    """Median views-per-hour across a channel's other uploads.

    Median rather than mean: one viral upload in the sample would pull a mean up
    far enough that the channel's next genuine outlier scores below 1.0. The
    median is what makes this an outlier detector rather than a trend line.

    ``trusted`` is false when the sample is thin. A thin sample still produces a
    number - it is better than nothing for ranking within one scan - but it is
    flagged so the summary can say the ranking is provisional.
    """
    now = now or datetime.now(UTC)
    channel_id = next((r.channel_id for r in records if r.channel_id), "") or ""

    samples: list[float] = []
    for record in records:
        if exclude_video_id and record.video_id == exclude_video_id:
            continue
        if record.is_live:
            continue
        vph = views_per_hour(record, now=now, min_age_hours=min_age_hours)
        if vph is not None and vph > 0:
            samples.append(vph)

    if not samples:
        return ChannelBaseline(channel_id=channel_id, median_vph=0.0, sample_size=0, trusted=False)

    return ChannelBaseline(
        channel_id=channel_id,
        median_vph=float(statistics.median(samples)),
        sample_size=len(samples),
        trusted=len(samples) >= int(min_uploads),
    )


def score_records(
    records: list[VideoRecord],
    *,
    now: datetime | None = None,
    min_age_hours: float = 6.0,
    min_channel_uploads: int = 5,
    min_duration_s: float = 0.0,
    per_channel_min_duration: dict[str, float] | None = None,
) -> tuple[list[VelocityScore], list[Skipped]]:
    """Score every record against its own channel's baseline.

    Returns the scored candidates ranked highest-velocity first, plus the
    records that could not be scored with the reason for each.

    ``per_channel_min_duration`` lets the watchlist override the global floor
    per channel, which is how ``config/watchlist.yaml`` already expresses it.
    """
    now = now or datetime.now(UTC)
    per_channel_min_duration = per_channel_min_duration or {}

    by_channel: dict[str, list[VideoRecord]] = {}
    for record in records:
        by_channel.setdefault(record.channel_id or "", []).append(record)

    scored: list[VelocityScore] = []
    skipped: list[Skipped] = []

    for channel_id, channel_records in by_channel.items():
        floor = float(per_channel_min_duration.get(channel_id, min_duration_s))

        for record in channel_records:
            reason = classify(record, now=now, min_age_hours=min_age_hours, min_duration_s=floor)
            if reason is not None:
                skipped.append(Skipped(record=record, reason=reason))
                continue

            vph = views_per_hour(record, now=now, min_age_hours=min_age_hours)
            if vph is None:
                skipped.append(Skipped(record=record, reason="too_new"))
                continue

            baseline = channel_baseline(
                channel_records,
                now=now,
                exclude_video_id=record.video_id,
                min_uploads=min_channel_uploads,
                min_age_hours=min_age_hours,
            )
            if baseline.median_vph <= 0:
                # Nothing to normalize against. The video may still be worth
                # clipping, so it is reported rather than dropped - it just has
                # no comparative claim to make.
                skipped.append(Skipped(record=record, reason="no_baseline"))
                continue

            age = (now - record.published_at).total_seconds() / 3600.0  # type: ignore[operator]
            scored.append(
                VelocityScore(
                    record=record,
                    vph=vph,
                    baseline=baseline,
                    velocity=vph / baseline.median_vph,
                    age_hours=age,
                )
            )

    scored.sort(key=lambda s: (s.velocity, s.vph), reverse=True)
    return scored, skipped


def cap_per_channel(
    scores: list[VelocityScore], limits: dict[str, int], default: int = 2
) -> list[VelocityScore]:
    """Keep at most ``max_per_day`` candidates from any one channel.

    The watchlist sets this per channel. Without it a channel that uploads daily
    can fill the whole shortlist, and a clip feed made of one creator is the
    reused-content pattern the rights tiers exist to avoid.
    """
    kept: list[VelocityScore] = []
    counts: dict[str, int] = {}
    for score in scores:
        channel_id = score.record.channel_id or ""
        limit = int(limits.get(channel_id, default))
        if limit <= 0:
            continue
        if counts.get(channel_id, 0) >= limit:
            continue
        counts[channel_id] = counts.get(channel_id, 0) + 1
        kept.append(score)
    return kept


def summarize_skips(skipped: list[Skipped]) -> dict[str, int]:
    """Counts per reason, for the scan summary."""
    counts: dict[str, int] = {}
    for item in skipped:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return counts


__all__ = [
    "SKIP_REASONS",
    "ChannelBaseline",
    "Skipped",
    "VelocityScore",
    "cap_per_channel",
    "channel_baseline",
    "classify",
    "score_records",
    "summarize_skips",
    "views_per_hour",
]
