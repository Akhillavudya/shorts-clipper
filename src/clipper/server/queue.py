"""The durable job queue (WEB_UI_PLAN.md section 1).

One JSON file, ``jobs/_queue/queue.json``, holding every submission in the
order it was made. It is the scheduler: the supervisor thread reads it to
decide what to start next, and the API writes to it when you paste a link.

Three properties it has to have, and how each is bought:

**It survives a restart.** The queue is the whole point of "paste three links
and walk away" - a queue that lives in memory loses two of them when the
process dies, which is exactly when you are not watching. Writes go through
``mkstemp`` + ``os.replace`` in the target directory, the same atomic dance
:class:`~clipper.state.JobStore` does, so an interrupted write cannot truncate
the file.

**It has exactly one writer.** The supervisor (inside the Flask process) owns
``queue.json``; the worker child owns ``jobs/<video_id>/state.json`` and its
own event log. No file has two writers, which is why there is no cross-process
lock anywhere in this package - see section 3 of the plan for why a lock would
not have worked anyway.

**It is invisible to the rest of the system.** ``JobStore.list_jobs`` only
returns directories containing a ``state.json``, so ``_queue`` never shows up
as a job, exactly as ``.discover`` does not.

``entry_id`` rather than ``video_id`` is the identity, and that is the
load-bearing choice here: at submit time all that exists is a URL. The video id
arrives minutes later, from ``ingest``, and the worker reports it back by
emitting a ``resolved`` event that the supervisor copies onto the entry. Until
then the row still has to be addressable - to show it, to cancel it, to delete
it.

Flask-free and thread-free on purpose. Everything here is a pure function of
the file on disk, so ``tests/test_server_queue.py`` needs no app, no client and
no process.
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clipper.state import DEFAULT_JOBS_ROOT, RightsStatus, utcnow

#: Where the queue lives, relative to the jobs root. Leading underscore so it
#: sorts away from video ids and reads as "not a job".
QUEUE_DIRNAME = "_queue"

QUEUE_FILENAME = "queue.json"


class QueueStatus(StrEnum):
    """Lifecycle of one submission.

    ``blocked`` is deliberately not ``failed``: a rights block is the cheap,
    correct outcome the whole of section 5 is designed to produce, and showing
    it in the same colour as "ffmpeg died" would train you to ignore it.

    ``interrupted`` is deliberately not ``cancelled``: nobody asked for it. It
    means the server and its worker died together, which ``reclaim`` detects on
    the next startup and which a retry resumes from, because every stage is
    idempotent.
    """

    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"
    interrupted = "interrupted"
    blocked = "blocked"


#: Statuses that still own, or want, a worker process.
ACTIVE_STATUSES: frozenset[QueueStatus] = frozenset({QueueStatus.queued, QueueStatus.running})


class RunOptions(BaseModel):
    """The CLI flags a submission carries, as data.

    Deliberately a mirror of ``clipper run`` / ``clipper render`` rather than a
    new vocabulary: the worker unpacks these straight into
    :func:`clipper.pipeline.run_all`, so a flag that needs translating here is
    a flag that will drift.

    ``allow_red`` exists but is **not settable from the submit form** - see
    ``api.py``. Section 5: you cannot meaningfully pre-authorise an override
    for a channel you have not seen yet. It is reachable only from the job
    page, after ingest, with the rights note in front of you.
    """

    model_config = ConfigDict(extra="forbid")

    rights: str | None = None
    music: str | None = None
    no_music: bool = False
    centre_crop: bool = False
    force: bool = False
    #: A render-only entry: an existing job, re-rendered, skipping the first
    #: three stages. Phase 5 submits these from the job page.
    render_only: bool = False
    approved_only: bool = False
    clip_index: int | None = None
    allow_red: bool = False
    #: Why the gate was overridden, in the words typed on the job page.
    #: ``pipeline.render`` records it on the clip, and hard rule 1 is the
    #: reason it is not optional in practice: an override nobody wrote a
    #: reason for is indistinguishable from a default, which is the exact
    #: thing that rule forbids.
    override_reason: str | None = None

    @field_validator("rights")
    @classmethod
    def _known_rights(cls, value: str | None) -> str | None:
        """Same wording as ``rights/registry.py`` uses for ``--rights``."""
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        try:
            return RightsStatus(cleaned).value
        except ValueError as exc:
            allowed = ", ".join(s.value for s in RightsStatus)
            raise ValueError(f"rights={value!r} is not one of {allowed}") from exc

    @field_validator("music")
    @classmethod
    def _blank_music_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class QueueEntry(BaseModel):
    """One submission, from paste to terminal status."""

    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source: str
    status: QueueStatus = QueueStatus.queued
    options: RunOptions = Field(default_factory=RunOptions)
    video_id: str | None = None
    rights_status: str | None = None
    pid: int | None = None
    submitted_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: Stamped by the supervisor each tick from ``proc.poll() is None``. A
    #: liveness claim, which is a different and weaker claim than progress -
    #: "alive, 2m since last event" is honest where a synthetic percentage is
    #: not.
    last_alive: datetime | None = None
    error: str | None = None
    exit_code: int | None = None

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def label(self) -> str:
        """What to call this row before ``ingest`` has produced a title."""
        return self.video_id or self.source


class _QueueFile(BaseModel):
    """The file envelope. Versioned because the SPA is typed against it."""

    version: int = 1
    entries: list[QueueEntry] = Field(default_factory=list)


class QueueBusy(RuntimeError):
    """A queue mutation was refused because the entry is running.

    Carried to the API as a ``409``: it is not a bad request, it is a request
    that is correct but not now.
    """


class QueueStore:
    """Reads and writes ``jobs/_queue/queue.json``.

    The jobs root is injected, never imported from config, for the same reason
    :class:`~clipper.state.JobStore` injects it: tests point it at a temp
    directory and the server points it at whatever ``--jobs-root`` said.

    Every mutator is load-modify-save against the file rather than against
    cached state. That costs a few milliseconds per submit and buys the
    property that two :class:`QueueStore` instances over one directory - the
    API's and the supervisor's, which is the real arrangement - cannot hold
    divergent pictures of the queue.
    """

    def __init__(self, jobs_root: Path | str = DEFAULT_JOBS_ROOT) -> None:
        self.jobs_root = Path(jobs_root)
        self.lock = threading.RLock()

    # -- paths ----------------------------------------------------------

    @property
    def dir(self) -> Path:
        return self.jobs_root / QUEUE_DIRNAME

    @property
    def path(self) -> Path:
        return self.dir / QUEUE_FILENAME

    @property
    def events_dir(self) -> Path:
        return self.dir / "events"

    @property
    def logs_dir(self) -> Path:
        return self.dir / "logs"

    def cancel_path(self, entry_id: str) -> Path:
        """The graceful-cancel sentinel the worker checks between stages."""
        return self.dir / f"{entry_id}.cancel"

    def log_path(self, entry_id: str) -> Path:
        """The worker's raw stdout/stderr. Never stripped - see ``text.py``."""
        return self.logs_dir / f"worker-{entry_id}.log"

    # -- load / save ----------------------------------------------------

    def load(self) -> list[QueueEntry]:
        """Every entry, in submission order. A missing file is an empty queue.

        A *corrupt* file is not: it raises, because silently starting from an
        empty queue would discard work that was submitted and is arguably still
        the most surprising thing this package could do.
        """
        if not self.path.is_file():
            return []
        return _QueueFile.model_validate_json(self.path.read_text(encoding="utf-8")).entries

    def save(self, entries: Iterable[QueueEntry]) -> Path:
        """Atomically replace the queue file, creating ``_queue`` if needed."""
        payload = _QueueFile(entries=list(entries)).model_dump_json(indent=2)
        self.dir.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(dir=self.dir, prefix=".queue.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return self.path

    # -- reads ----------------------------------------------------------

    def get(self, entry_id: str) -> QueueEntry | None:
        for entry in self.load():
            if entry.entry_id == entry_id:
                return entry
        return None

    def next_queued(self) -> QueueEntry | None:
        """The oldest entry still waiting. Strict FIFO, and it stays that way.

        No priorities and no reordering: one desktop runs one job at a time,
        and the order you pasted the links in is the order you meant.
        """
        for entry in self.load():
            if entry.status is QueueStatus.queued:
                return entry
        return None

    def running(self) -> QueueEntry | None:
        """The entry that owns a worker right now, if any.

        Singular by design - section 9 rules out multi-job parallelism, which
        on one desktop is slower than serial *and* makes the state-file race in
        section 3 real again.
        """
        for entry in self.load():
            if entry.status is QueueStatus.running:
                return entry
        return None

    def running_video_id(self) -> str | None:
        """Which job's ``state.json`` currently has a writer.

        The single-writer discipline in section 3 is this one function: the
        review endpoint refuses with ``409`` when the id it was handed matches,
        and the job payload's ``writable`` flag is its negation.
        """
        entry = self.running()
        return entry.video_id if entry is not None else None

    def entry_for_video(self, video_id: str) -> QueueEntry | None:
        """The most recent entry that produced this job, if the queue has one.

        Most recent rather than first: a job can be submitted, rendered, and
        re-rendered, and the row that matters is the latest one.
        """
        found = None
        for entry in self.load():
            if entry.video_id == video_id:
                found = entry
        return found

    # -- writes ---------------------------------------------------------

    def enqueue(self, source: str, options: RunOptions | None = None) -> QueueEntry:
        """Append one submission and persist it. Returns the stored entry."""
        entry = QueueEntry(source=source, options=options or RunOptions())
        entries = self.load()
        entries.append(entry)
        self.save(entries)
        return entry

    def enqueue_many(
        self, sources: Iterable[str], options: RunOptions | None = None
    ) -> list[QueueEntry]:
        """Several submissions in one write.

        "Paste several links" is one user action, so it is one file write and
        one ordering decision, not N racing round-trips.
        """
        options = options or RunOptions()
        entries = self.load()
        fresh = [QueueEntry(source=s, options=options.model_copy()) for s in sources]
        entries.extend(fresh)
        self.save(entries)
        return fresh

    def update(self, entry_id: str, **fields: Any) -> QueueEntry:
        """Set fields on one entry and persist. Raises if it is gone.

        Validated through the model rather than assigned blind, so a typo'd
        status becomes an error here instead of a row the SPA cannot colour.
        """
        entries = self.load()
        for index, entry in enumerate(entries):
            if entry.entry_id == entry_id:
                # Merged into a plain dict and re-validated rather than
                # ``model_copy(update=...)``, which skips validators by design
                # and would happily store a status the SPA cannot colour.
                data = entry.model_dump()
                data.update(fields)
                updated = QueueEntry.model_validate(data)
                entries[index] = updated
                self.save(entries)
                return updated
        raise KeyError(f"no queue entry {entry_id!r}")

    def remove(self, entry_id: str) -> QueueEntry:
        """Drop an entry from the queue. Refuses the running one.

        Removing a running entry would orphan its worker: the supervisor reaps
        by looking the entry up, so a row that vanished mid-run leaves a live
        ffmpeg tree with nothing tracking it. Cancel first, then remove.
        """
        entries = self.load()
        for index, entry in enumerate(entries):
            if entry.entry_id == entry_id:
                if entry.status is QueueStatus.running:
                    raise QueueBusy(f"{entry_id} is running - cancel it before removing it")
                del entries[index]
                self.save(entries)
                self.cancel_path(entry_id).unlink(missing_ok=True)
                return entry
        raise KeyError(f"no queue entry {entry_id!r}")

    def clear_finished(self) -> list[QueueEntry]:
        """Drop every terminal row, keeping ``queued`` and ``running``."""
        entries = self.load()
        kept = [e for e in entries if e.is_active]
        removed = [e for e in entries if not e.is_active]
        if removed:
            self.save(kept)
            for entry in removed:
                self.cancel_path(entry.entry_id).unlink(missing_ok=True)
        return removed

    def reclaim(self, *, is_alive: Callable[[int], bool]) -> list[QueueEntry]:
        """Mark entries whose worker is gone as ``interrupted``. Run at startup.

        An entry left ``running`` with a dead pid means the server and its
        worker died together - a laptop lid, a power cut, a Ctrl-C on the
        wrong terminal. ``interrupted`` says exactly that, and because every
        stage is idempotent (``should_run`` skips completed stages, output
        files are guarded by ``or not X.is_file()``) the retry button can
        simply resume.

        An entry with no pid at all is reclaimed too: it was killed between
        being marked ``running`` and the supervisor recording the process.

        ``is_alive`` is injected so the test suite decides what a live pid is
        without spawning one.
        """
        entries = self.load()
        reclaimed: list[QueueEntry] = []
        for index, entry in enumerate(entries):
            if entry.status is not QueueStatus.running:
                continue
            if entry.pid is not None and is_alive(entry.pid):
                continue
            entries[index] = entry.model_copy(
                update={
                    "status": QueueStatus.interrupted,
                    "finished_at": utcnow(),
                    "error": "the server stopped while this job was running",
                    "pid": None,
                }
            )
            reclaimed.append(entries[index])
        if reclaimed:
            self.save(entries)
        return reclaimed


__all__ = [
    "ACTIVE_STATUSES",
    "QUEUE_DIRNAME",
    "QUEUE_FILENAME",
    "QueueBusy",
    "QueueEntry",
    "QueueStatus",
    "QueueStore",
    "RunOptions",
]
