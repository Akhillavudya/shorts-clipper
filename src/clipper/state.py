"""Job state - the single source of truth for one video (FULL_PLAN.md section 1).

There is no orchestration framework here on purpose. Every stage reads
``jobs/<video_id>/state.json``, does its work, and writes it back. That makes
the pipeline resumable (rerun the stage that failed, not the pipeline),
debuggable (``cat state.json``), and inspectable without running anything.

The contract each stage honours:

- before work:  status -> running, ``started_at`` stamped, ``attempts`` bumped
- on success:   status -> done,   ``finished_at`` stamped
- on failure:   status -> failed, ``error`` recorded, exception re-raised

State is flushed to disk at each transition, so a hard kill mid-stage still
leaves an accurate record: the stage reads ``running``, which is how a crashed
run is told apart from one that never started.

Writes are atomic (temp file in the same directory, then ``os.replace``) so an
interrupted write cannot truncate an existing state file.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

DEFAULT_JOBS_ROOT = Path("jobs")

#: Per-job stages, in pipeline order. Mirrors the CLI subcommands in section 11.
JOB_STAGES: tuple[str, ...] = ("ingest", "transcribe", "moments", "render")

#: Per-clip stages, in the order section 6 requires them to run.
CLIP_STAGES: tuple[str, ...] = ("cut", "reframe", "audio_mix", "captions", "render")


def utcnow() -> datetime:
    """Timezone-aware UTC now. Every timestamp in the state file uses this."""
    return datetime.now(UTC)


class StageStatus(StrEnum):
    """Lifecycle of a single stage."""

    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class RightsStatus(StrEnum):
    """Source tier from FULL_PLAN.md section 0.

    ``unknown`` is deliberately distinct from ``red`` but gated identically:
    an unclassified source has not been cleared, and the export gate treats
    "not yet checked" as "not allowed".
    """

    green = "green"
    amber = "amber"
    red = "red"
    unknown = "unknown"


#: Statuses the renderer will export without an explicit override.
EXPORTABLE_RIGHTS: frozenset[RightsStatus] = frozenset({RightsStatus.green, RightsStatus.amber})


class RightsBlocked(RuntimeError):
    """Raised when a job is exported despite a non-clearable rights status."""


class StageState(BaseModel):
    """Status and timing for one stage of one job (or one clip)."""

    status: StageStatus = StageStatus.pending
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int = 0
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        return self.status is StageStatus.done

    @property
    def duration_s(self) -> float | None:
        """Wall-clock seconds the last attempt took, if it finished."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


def _new_stage_map(names: tuple[str, ...]) -> dict[str, StageState]:
    return {name: StageState() for name in names}


class ClipState(BaseModel):
    """One candidate clip within a job.

    Cut points are stored only after ``moments/snap.py`` has run: the model
    supplies word indices, never timestamps, so ``start``/``end`` stay ``None``
    until the deterministic snapping step has resolved them.
    """

    clip_id: str
    index: int

    # Set by moments/recall.py (word indices) then moments/snap.py (seconds).
    start_word_index: int | None = None
    end_word_index: int | None = None
    start: float | None = None
    end: float | None = None

    # Set by moments/score.py.
    type: str | None = None
    reason: str | None = None
    rubric: dict[str, float] = Field(default_factory=dict)
    score: float | None = None
    suggested_title: str | None = None
    emphasis_words: list[str] = Field(default_factory=list)
    hook_rewrite: str | None = None

    # Human review (FULL_PLAN.md section 12, Phase 4). Rejections are training
    # data: a rejected clip keeps its rubric in scored.json, and the note is the
    # only record of *why* the score was wrong. ``approved is None`` means "not
    # looked at yet", which is distinct from a deliberate reset to undecided
    # only by ``reviewed_at`` being set.
    approved: bool | None = None
    review_note: str | None = None
    reviewed_at: datetime | None = None

    # Set by publish/ once the Short is live. ``youtube_video_id`` is the join
    # key the calibration loop needs: without it there is no way to line a
    # retention figure up with the rubric that predicted it, and the rubric
    # archive is worth nothing on its own (CLAUDE.md hard rule 8).
    youtube_video_id: str | None = None
    published_at: datetime | None = None

    stages: dict[str, StageState] = Field(default_factory=lambda: _new_stage_map(CLIP_STAGES))

    @property
    def duration_s(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def stage(self, name: str) -> StageState:
        return self.stages.setdefault(name, StageState())


class JobState(BaseModel):
    """Everything known about one source video and the clips cut from it."""

    model_config = ConfigDict(validate_assignment=True)

    schema_version: int = SCHEMA_VERSION
    video_id: str

    # Source identity. Exactly one of these is set by ingest.
    source_url: str | None = None
    source_path: str | None = None

    # Populated by ingest/download.py from yt-dlp metadata.
    title: str | None = None
    channel_id: str | None = None
    channel_title: str | None = None
    duration_s: float | None = None
    published_at: datetime | None = None
    licence: str | None = None

    # FULL_PLAN.md section 0: first-class, never derived at render time.
    rights_status: RightsStatus = RightsStatus.unknown
    rights_note: str | None = None
    rights_override: bool = False
    rights_override_reason: str | None = None

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    stages: dict[str, StageState] = Field(default_factory=lambda: _new_stage_map(JOB_STAGES))
    clips: list[ClipState] = Field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = utcnow()

    # -- stages ---------------------------------------------------------

    def stage(self, name: str) -> StageState:
        """Get a stage record, creating it if this state predates the stage."""
        return self.stages.setdefault(name, StageState())

    def is_done(self, name: str) -> bool:
        return self.stage(name).is_done

    def next_pending_stage(self) -> str | None:
        """First stage in pipeline order that has not completed.

        This is what ``clipper run`` resumes from.
        """
        for name in JOB_STAGES:
            if not self.is_done(name):
                return name
        return None

    # -- clips ----------------------------------------------------------

    def clip(self, clip_id: str) -> ClipState | None:
        return next((c for c in self.clips if c.clip_id == clip_id), None)

    def clip_by_index(self, index: int) -> ClipState | None:
        return next((c for c in self.clips if c.index == index), None)

    def add_clip(self, **fields: Any) -> ClipState:
        """Append a clip, assigning the next index and a ``clip_NN`` id."""
        index = fields.pop("index", None)
        if index is None:
            index = max((c.index for c in self.clips), default=0) + 1
        clip = ClipState(
            clip_id=fields.pop("clip_id", f"clip_{index:02d}"),
            index=index,
            **fields,
        )
        self.clips.append(clip)
        self.touch()
        return clip

    # -- rights gate ----------------------------------------------------

    @property
    def is_exportable(self) -> bool:
        """Whether render may write an uploadable file for this job."""
        return self.rights_status in EXPORTABLE_RIGHTS or self.rights_override

    def assert_exportable(self) -> None:
        """Raise unless this job may be exported.

        Called by edit/render.py before it writes anything. Enforcement of the
        reused-content policy is at the channel level, so the cost of a wrong
        export is the whole channel, not one video.
        """
        if self.is_exportable:
            return
        raise RightsBlocked(
            f"job {self.video_id!r} has rights_status={self.rights_status.value!r} "
            f"and no override; refusing to export. Classify the source in "
            f"config/watchlist.yaml, or pass an explicit override."
        )


class JobStore:
    """Reads and writes ``jobs/<video_id>/state.json``.

    The jobs root is injected rather than imported from config so tests can
    point it at a temp directory.
    """

    def __init__(self, root: Path | str = DEFAULT_JOBS_ROOT) -> None:
        self.root = Path(root)

    # -- paths ----------------------------------------------------------

    def job_dir(self, video_id: str) -> Path:
        return self.root / video_id

    def state_path(self, video_id: str) -> Path:
        return self.job_dir(video_id) / "state.json"

    def clip_dir(self, video_id: str, clip_id: str) -> Path:
        return self.job_dir(video_id) / "clips" / clip_id

    def exists(self, video_id: str) -> bool:
        return self.state_path(video_id).is_file()

    def list_jobs(self) -> list[str]:
        """Video ids that have a state file, oldest-modified first."""
        if not self.root.is_dir():
            return []
        found = [d.name for d in self.root.iterdir() if (d / "state.json").is_file()]
        return sorted(found, key=lambda v: self.state_path(v).stat().st_mtime)

    # -- load / save ----------------------------------------------------

    def load(self, video_id: str) -> JobState:
        path = self.state_path(video_id)
        if not path.is_file():
            raise FileNotFoundError(f"no job state at {path}")
        return JobState.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, state: JobState) -> Path:
        """Atomically write the state file, creating the job directory."""
        state.touch()
        path = self.state_path(state.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json(indent=2)

        # Same directory as the target, so os.replace stays on one filesystem
        # and is therefore atomic.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path

    def create(self, video_id: str, **fields: Any) -> JobState:
        state = JobState(video_id=video_id, **fields)
        self.save(state)
        return state

    def load_or_create(self, video_id: str, **fields: Any) -> JobState:
        """Resume an existing job, or start one. The normal entrypoint."""
        if self.exists(video_id):
            return self.load(video_id)
        return self.create(video_id, **fields)

    # -- stage execution ------------------------------------------------

    def should_run(self, stage: StageState, *, force: bool = False) -> bool:
        """Whether a stage needs running.

        A ``running`` stage is treated as needing a rerun: the only way it is
        still ``running`` on load is that the previous process died.
        """
        return force or not stage.is_done

    @contextmanager
    def stage(
        self, state: JobState, name: str, *, clip_id: str | None = None
    ) -> Iterator[StageState]:
        """Run a stage, recording status and timings around it.

        Flushes to disk on entry and on both exit paths, so an interrupted run
        leaves the stage marked ``running`` rather than silently ``pending``.

            with store.stage(state, "transcribe") as st:
                st.detail["backend"] = "gemini"
        """
        owner: JobState | ClipState = state
        if clip_id is not None:
            clip = state.clip(clip_id)
            if clip is None:
                raise KeyError(f"job {state.video_id!r} has no clip {clip_id!r}")
            owner = clip

        st = owner.stage(name)
        st.status = StageStatus.running
        st.started_at = utcnow()
        st.finished_at = None
        st.error = None
        st.attempts += 1
        self.save(state)

        try:
            yield st
        except BaseException as exc:
            st.status = StageStatus.failed
            st.finished_at = utcnow()
            st.error = f"{type(exc).__name__}: {exc}"
            self.save(state)
            raise
        else:
            st.status = StageStatus.done
            st.finished_at = utcnow()
            st.error = None
            self.save(state)


def read_state(path: Path | str) -> JobState:
    """Load a state file by full path, for ad-hoc inspection."""
    return JobState.model_validate_json(Path(path).read_text(encoding="utf-8"))


def summarize(state: JobState) -> str:
    """One-line-per-stage summary, for CLI status output and debugging."""
    lines = [
        f"{state.video_id}  rights={state.rights_status.value}"
        f"{' (OVERRIDDEN)' if state.rights_override else ''}"
    ]
    for name in JOB_STAGES:
        st = state.stage(name)
        when = st.finished_at or st.started_at
        stamp = when.isoformat(timespec="seconds") if when else "-"
        lines.append(f"  {name:<12} {st.status.value:<8} {stamp}")
    for clip in state.clips:
        score = f"{clip.score:.2f}" if clip.score is not None else "-"
        lines.append(f"  {clip.clip_id:<12} score={score}")
    return "\n".join(lines)


__all__ = [
    "CLIP_STAGES",
    "DEFAULT_JOBS_ROOT",
    "EXPORTABLE_RIGHTS",
    "JOB_STAGES",
    "SCHEMA_VERSION",
    "ClipState",
    "JobState",
    "JobStore",
    "RightsBlocked",
    "RightsStatus",
    "StageState",
    "StageStatus",
    "read_state",
    "summarize",
    "utcnow",
]
