"""Read-side view models for the review UI (FULL_PLAN.md section 12, Phase 4).

Everything the page shows is assembled here, and nothing here imports Flask.
The templates get plain dataclasses with the strings already formatted, so a
test can assert on what a reviewer would see without going through HTTP, and a
change to the page cannot quietly change what counts as "reviewed".

The one rule this module keeps for itself: it builds *relative* media paths
(``clips/clip_01/final.mp4``), never URLs. Routing belongs to ``app.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from clipper.moments.score import RUBRIC_AXES
from clipper.state import ClipState, JobState, JobStore, StageStatus, utcnow
from clipper.transcribe.schema import Transcript, transcript_path

#: Previews, best first. A rendered Short is what actually goes up, so it wins;
#: the intermediate files are worth watching when a render looks wrong; the
#: source span is the fallback, and the normal case - review happens *before*
#: render, so usually nothing has been encoded yet.
PREVIEW_ORDER: tuple[tuple[str, str], ...] = (
    ("final.mp4", "final"),
    ("reframed.mp4", "reframed"),
    ("cut.mp4", "cut"),
)

PREVIEW_LABELS = {
    "final": "final render",
    "reframed": "reframed, before captions",
    "cut": "raw cut, before reframing",
    "source": "source video, seeked to the moment",
}

#: Suffixes the media route will serve. Everything else in a job directory -
#: state.json, llm.log, the LLM cache - is not a preview and is not served.
MEDIA_SUFFIXES: frozenset[str] = frozenset({".mp4", ".m4a", ".mp3", ".wav", ".webm"})


def timecode(seconds: float | None) -> str:
    """``h:mm:ss`` for anything over an hour, ``m:ss`` below it."""
    if seconds is None:
        return "-"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


@dataclass(frozen=True)
class Axis:
    """One rubric axis, with the weight that turned it into the total."""

    name: str
    label: str
    score: float
    weight: float | None

    @property
    def percent(self) -> float:
        """Bar width. The rubric is scored 1-10, so the score is the percent."""
        return max(0.0, min(100.0, self.score * 10.0))


@dataclass(frozen=True)
class Preview:
    """What to play for a clip, and which span of it matters.

    ``start``/``end`` are seconds into *this file*, not into the source: for a
    rendered clip they are the whole file, and for the source fallback they are
    the cut points. The page seeks and stops on them either way, so one code
    path covers both.
    """

    path: str
    kind: str
    start: float
    end: float

    @property
    def label(self) -> str:
        return PREVIEW_LABELS.get(self.kind, self.kind)

    @property
    def is_source(self) -> bool:
        return self.kind == "source"


@dataclass(frozen=True)
class ClipCard:
    """One candidate, as a reviewer sees it."""

    clip_id: str
    index: int
    score: float | None
    type: str | None
    reason: str | None
    suggested_title: str | None
    hook_rewrite: str | None
    emphasis_words: list[str]
    start: float | None
    end: float | None
    duration_s: float | None
    rubric: list[Axis]
    transcript: str
    approved: bool | None
    review_note: str
    reviewed_at: datetime | None
    preview: Preview | None
    rendered: bool
    render_note: str | None
    youtube_video_id: str | None
    source_link: str | None

    @property
    def verdict(self) -> str:
        """``approved`` | ``rejected`` | ``pending``. What the CSS keys off."""
        if self.approved is True:
            return "approved"
        if self.approved is False:
            return "rejected"
        return "pending"

    @property
    def score_label(self) -> str:
        return f"{self.score:.2f}" if self.score is not None else "-"

    @property
    def span_label(self) -> str:
        if self.start is None or self.end is None:
            return "not snapped"
        return f"{timecode(self.start)} - {timecode(self.end)}"

    @property
    def duration_label(self) -> str:
        return f"{self.duration_s:.0f}s" if self.duration_s else "-"


@dataclass(frozen=True)
class Counts:
    """Review progress for one job."""

    total: int = 0
    approved: int = 0
    rejected: int = 0

    @property
    def pending(self) -> int:
        return self.total - self.approved - self.rejected

    @property
    def done(self) -> bool:
        return self.total > 0 and self.pending == 0


@dataclass(frozen=True)
class JobSummary:
    """One row of the job list."""

    video_id: str
    title: str
    channel_title: str | None
    rights_status: str
    rights_override: bool
    exportable: bool
    stage: str
    stage_failed: bool
    counts: Counts
    updated_at: datetime | None


@dataclass(frozen=True)
class JobView:
    """A whole job's review page."""

    video_id: str
    title: str
    channel_title: str | None
    source_url: str | None
    duration_s: float | None
    rights_status: str
    rights_note: str | None
    rights_override: bool
    exportable: bool
    counts: Counts
    clips: list[ClipCard] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_label(self) -> str:
        return timecode(self.duration_s)


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def _counts(state: JobState) -> Counts:
    return Counts(
        total=len(state.clips),
        approved=sum(1 for c in state.clips if c.approved is True),
        rejected=sum(1 for c in state.clips if c.approved is False),
    )


def _rubric(clip: ClipState, weights: dict[str, Any]) -> list[Axis]:
    """Rubric axes in the canonical order, with any unrecognised ones appended.

    The order is ``RUBRIC_AXES`` because that is the order the prompt asks for
    them in and the order the plan reads them in. An axis a future prompt adds
    still shows up rather than vanishing from the page.
    """
    configured = weights.get("weights") if isinstance(weights, dict) else None
    configured = configured if isinstance(configured, dict) else {}

    names = list(RUBRIC_AXES) + [k for k in clip.rubric if k not in RUBRIC_AXES]
    axes = []
    for name in names:
        if name not in clip.rubric:
            continue
        weight = configured.get(name)
        axes.append(
            Axis(
                name=name,
                label=name.replace("_", " "),
                score=float(clip.rubric[name]),
                weight=float(weight) if isinstance(weight, int | float) else None,
            )
        )
    return axes


def _preview(job_dir: Path, clip: ClipState) -> Preview | None:
    """The best available preview for a clip.

    Falls back to the source video with the cut points attached, because review
    happens before render: there is normally no encoded clip to play yet, and a
    page that said "render it first" would invert the whole point of reviewing
    before the expensive stage runs.
    """
    clip_dir = job_dir / "clips" / clip.clip_id
    for filename, kind in PREVIEW_ORDER:
        if (clip_dir / filename).is_file():
            return Preview(
                path=f"clips/{clip.clip_id}/{filename}",
                kind=kind,
                start=0.0,
                end=clip.duration_s or 0.0,
            )

    if (job_dir / "source.mp4").is_file() and clip.start is not None and clip.end is not None:
        return Preview(path="source.mp4", kind="source", start=clip.start, end=clip.end)
    return None


def _source_link(state: JobState, clip: ClipState) -> str | None:
    """The source URL seeked to the clip, for checking it against the original."""
    if not state.source_url or clip.start is None:
        return None
    separator = "&" if "?" in state.source_url else "?"
    return f"{state.source_url}{separator}t={int(clip.start)}s"


def _transcript_for(transcript: Transcript | None, clip: ClipState) -> str:
    if transcript is None or clip.start_word_index is None or clip.end_word_index is None:
        return ""
    return transcript.text_for(clip.start_word_index, clip.end_word_index)


def load_transcript(job_dir: Path) -> Transcript | None:
    """The job's transcript, or ``None`` if it has not been produced yet.

    A missing or unreadable transcript degrades the page to scores and titles
    rather than failing it: the rubric is still reviewable without the words.
    """
    path = transcript_path(job_dir)
    if not path.is_file():
        return None
    try:
        return Transcript.load(path)
    except Exception:  # noqa: BLE001 - a broken transcript degrades the page, not a 500
        return None


def build_clip(
    state: JobState,
    clip: ClipState,
    *,
    job_dir: Path,
    transcript: Transcript | None,
    weights: dict[str, Any],
) -> ClipCard:
    """One card. Split out so a test can build it without a whole job on disk."""
    track_note = clip.stage("reframe").detail.get("track_note")
    return ClipCard(
        clip_id=clip.clip_id,
        index=clip.index,
        score=clip.score,
        type=clip.type,
        reason=clip.reason,
        suggested_title=clip.suggested_title,
        hook_rewrite=clip.hook_rewrite,
        emphasis_words=list(clip.emphasis_words),
        start=clip.start,
        end=clip.end,
        duration_s=clip.duration_s,
        rubric=_rubric(clip, weights),
        transcript=_transcript_for(transcript, clip),
        approved=clip.approved,
        review_note=clip.review_note or "",
        reviewed_at=clip.reviewed_at,
        preview=_preview(job_dir, clip),
        rendered=clip.stage("render").status is StageStatus.done,
        render_note=track_note if isinstance(track_note, str) else None,
        youtube_video_id=clip.youtube_video_id,
        source_link=_source_link(state, clip),
    )


def build_job(store: JobStore, video_id: str, *, weights: dict[str, Any] | None = None) -> JobView:
    """Assemble one job's review page."""
    state = store.load(video_id)
    job_dir = store.job_dir(video_id)
    transcript = load_transcript(job_dir)
    weights = weights or {}

    warnings: list[str] = []
    if not state.clips:
        warnings.append(f"No scored candidates yet. Run `clipper moments {video_id}` and reload.")
    if not state.is_exportable:
        warnings.append(
            f"rights_status is '{state.rights_status.value}': render and publish will both "
            f"refuse this job. Approving a clip here does not clear that - classify the "
            f"channel in config/watchlist.yaml."
        )
    if transcript is None and state.clips:
        warnings.append("No transcript on disk, so the words of each clip are not shown.")

    clips = [
        build_clip(state, clip, job_dir=job_dir, transcript=transcript, weights=weights)
        for clip in state.clips
    ]

    return JobView(
        video_id=state.video_id,
        title=state.title or state.video_id,
        channel_title=state.channel_title,
        source_url=state.source_url,
        duration_s=state.duration_s,
        rights_status=state.rights_status.value,
        rights_note=state.rights_note,
        rights_override=state.rights_override,
        exportable=state.is_exportable,
        counts=_counts(state),
        clips=clips,
        warnings=warnings,
    )


def build_index(store: JobStore) -> list[JobSummary]:
    """Every job with a state file, most recently updated first."""
    summaries: list[JobSummary] = []
    for video_id in store.list_jobs():
        try:
            state = store.load(video_id)
        except Exception:  # noqa: BLE001 - one unreadable job must not blank the index
            continue
        pending = state.next_pending_stage()
        summaries.append(
            JobSummary(
                video_id=state.video_id,
                title=state.title or state.video_id,
                channel_title=state.channel_title,
                rights_status=state.rights_status.value,
                rights_override=state.rights_override,
                exportable=state.is_exportable,
                stage=pending or "complete",
                stage_failed=(
                    pending is not None and state.stage(pending).status is StageStatus.failed
                ),
                counts=_counts(state),
                updated_at=state.updated_at,
            )
        )
    summaries.sort(key=lambda s: s.updated_at or datetime.min, reverse=True)
    return summaries


# --------------------------------------------------------------------------
# the one write
# --------------------------------------------------------------------------


class ClipNotFound(LookupError):
    """The job exists; the clip id does not."""


def apply_review(
    store: JobStore,
    video_id: str,
    clip_id: str,
    *,
    approved: bool | None,
    note: str | None = None,
    now: datetime | None = None,
) -> ClipState:
    """Record a verdict on one clip and flush the state file.

    Reads state fresh on every call rather than holding a copy in the app: the
    CLI is expected to be running against the same job in another terminal, and
    the review page must not write back a stale copy of a stage record it never
    touched.

    ``reviewed_at`` is stamped even when the verdict is reset to undecided, so
    "looked at, still unsure" stays distinguishable from "never opened".
    """
    state = store.load(video_id)
    clip = state.clip(clip_id)
    if clip is None:
        raise ClipNotFound(f"job {video_id!r} has no clip {clip_id!r}")

    clip.approved = approved
    if note is not None:
        clip.review_note = note.strip() or None
    clip.reviewed_at = now or utcnow()

    store.save(state)
    return clip


__all__ = [
    "MEDIA_SUFFIXES",
    "PREVIEW_LABELS",
    "PREVIEW_ORDER",
    "Axis",
    "ClipCard",
    "ClipNotFound",
    "Counts",
    "JobSummary",
    "JobView",
    "Preview",
    "apply_review",
    "build_clip",
    "build_index",
    "build_job",
    "load_transcript",
    "timecode",
]
