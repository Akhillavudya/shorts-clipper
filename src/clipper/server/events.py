"""The progress event log (WEB_UI_PLAN.md section 2).

One append-only JSONL file per queue entry at
``jobs/_queue/events/<entry_id>.jsonl``::

    {"seq":12,"ts":"2026-09-17T14:03:11.442+00:00","kind":"progress",
     "stage":"clip_01/reframe","detail":"tracked crop to 1080x1920"}

:func:`EventLog.progress_callback` returns exactly the
``Progress = Callable[[str, str], None]`` shape ``pipeline.py`` already accepts,
so the worker passes it into ``run_all`` and **nothing in the pipeline
changes**.

**Why persisted rather than in-memory.** The producer is a different process,
so anything the worker holds in memory is unreachable from the server. It also
survives a browser refresh and a server restart mid-job - you are going to walk
away, so the history has to outlive the tab. And it is the only durable record
of the minutes in between: ``state.json`` flips ``running -> done`` with nothing
between them, and ``st.detail`` mutations are flushed only when the stage
context manager exits.

**Why polling rather than SSE or websockets, and please leave it that way.**
This repo has zero async code and this adds none. One browser tab on localhost
polling once a second is free. Werkzeug's threaded dev server pins a thread per
open SSE stream and browsers auto-reconnect, which is a thread-leak class of bug
in exchange for nothing visible. Polling asserts cleanly with
``app.test_client()`` - no async, no generators - which is the difference
between the test suite for this being easy and being a project of its own. And
it degrades correctly: shut the laptop for an hour and one ``?since=`` request
catches up.

Two robustness rules, both about the writer being killable:

**A truncated final line is skipped, not fatal.** A hard kill mid-``write``
leaves half a JSON object. A UI that returns 500 because the job crashed has
turned one failure into two, and has hidden the fifty good lines above the bad
one.

**Writes are line-atomic-ish and append-only.** Each event is one ``write`` of
one complete line to a file opened in append mode, which on both platforms is
as close to atomic as this needs to be given there is exactly one writer per
file.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from clipper.server.text import clean_error, strip_ansi
from clipper.state import DEFAULT_JOBS_ROOT, utcnow

#: The kinds an event can have. Not an enum on the wire - these are read by
#: TypeScript and by ``jq`` - but listed here so the set is discoverable and so
#: a typo becomes an assertion in one place.
EVENT_KINDS: frozenset[str] = frozenset(
    {
        "submitted",
        "started",
        "resolved",
        "rights",
        "progress",
        "stage",
        "blocked",
        "error",
        "finished",
        "cancelled",
    }
)

#: Hard cap on events per entry. A long render emits a few dozen; two thousand
#: is a runaway loop, and the log is a progress feed, not an audit trail - the
#: raw worker log keeps everything. Past the cap, appends are dropped rather
#: than rotated: losing the tail of a runaway is better than losing the head,
#: which is where the cause is.
MAX_EVENTS = 2000

#: Default page size for :meth:`EventLog.read`. The SPA polls with a cursor, so
#: this only bounds a cold first read.
DEFAULT_LIMIT = 500

#: Text longer than this in a single event is truncated. Individual progress
#: lines are short; an ffmpeg stderr dump arriving here means something went
#: wrong, and the raw log is where to read it.
MAX_DETAIL_CHARS = 1000


@dataclass(frozen=True)
class Event:
    """One line of the log.

    Frozen, like the presenter's view models, and for the same reason: it is
    read by the API, serialised, and never mutated.
    """

    seq: int
    ts: datetime
    kind: str
    stage: str = ""
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        """The wire form. ``schemas.event_json`` delegates here."""
        return {
            "seq": self.seq,
            "ts": self.ts.isoformat(),
            "kind": self.kind,
            "stage": self.stage,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Event:
        return cls(
            seq=int(data["seq"]),
            ts=datetime.fromisoformat(data["ts"]),
            kind=str(data.get("kind", "progress")),
            stage=str(data.get("stage", "")),
            detail=str(data.get("detail", "")),
        )


class EventLog:
    """Append-only progress events, one file per queue entry.

    Written by the worker child, read by the API. That split is why the
    sequence counter is recovered from the file on first use rather than being
    handed in: the two processes never share memory, and the reader must be
    able to attach to a log it did not start.
    """

    def __init__(self, jobs_root: Path | str = DEFAULT_JOBS_ROOT) -> None:
        self.jobs_root = Path(jobs_root)
        # entry_id -> (last seq written, last (kind, stage, detail) written).
        # Per-process, and only ever a cache: correctness comes from the file.
        self._seq: dict[str, int] = {}
        self._last: dict[str, tuple[str, str, str]] = {}

    # -- paths ----------------------------------------------------------

    @property
    def dir(self) -> Path:
        return self.jobs_root / "_queue" / "events"

    def path(self, entry_id: str) -> Path:
        return self.dir / f"{entry_id}.jsonl"

    def exists(self, entry_id: str) -> bool:
        return self.path(entry_id).is_file()

    # -- writing --------------------------------------------------------

    def append(
        self,
        entry_id: str,
        kind: str,
        stage: str = "",
        detail: str = "",
        *,
        ts: datetime | None = None,
    ) -> Event | None:
        """Append one event. Returns it, or ``None`` if it was dropped.

        Dropped in two cases, both deliberate:

        - **A duplicate of the immediately preceding event.** ``pipeline``
          re-announces a stage on resume and yt-dlp's hooks (phase 7) fire per
          chunk; a feed that repeats one line forty times reads as stuck.
        - **Past :data:`MAX_EVENTS`.** See the constant.
        """
        seq = self._next_seq(entry_id)
        if seq > MAX_EVENTS:
            return None

        kind = strip_ansi(kind).strip() or "progress"
        stage = _clip(strip_ansi(stage).strip())
        detail = _clip(strip_ansi(detail).strip())

        signature = (kind, stage, detail)
        if self._last.get(entry_id) == signature:
            return None

        event = Event(seq=seq, ts=ts or utcnow(), kind=kind, stage=stage, detail=detail)
        line = json.dumps(event.to_json(), ensure_ascii=False) + "\n"

        path = self.path(entry_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A killed writer leaves a final line with no newline on it. Appending
        # straight onto that glues the two together and loses *both* - the
        # broken one and the good one. Close the wound first; ``read`` then
        # skips exactly one line instead of two.
        if _ends_mid_line(path):
            line = "\n" + line
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

        self._seq[entry_id] = seq
        self._last[entry_id] = signature
        return event

    def error(self, entry_id: str, message: str, stage: str = "") -> Event | None:
        """An ``error`` event, cleaned and truncated on the way in.

        The raw log keeps the original bytes - ``text.py`` states that rule.
        This is the *surfaced* copy, so it is the one that gets trimmed.
        """
        return self.append(entry_id, "error", stage, clean_error(message) or "")

    def progress_callback(self, entry_id: str) -> Callable[[str, str], None]:
        """A ``Progress`` callback for ``pipeline.run_all(on_progress=...)``.

        The exact ``Callable[[str, str], None]`` the pipeline already calls, so
        the worker hands this straight over and ``pipeline.py`` is untouched.
        Exceptions are swallowed: a full disk in the event log must not take
        down a render that is otherwise fine, and the raw worker log still has
        the line.
        """

        def note(stage: str, detail: str = "") -> None:
            with contextlib.suppress(OSError):
                self.append(entry_id, "progress", stage, detail)

        return note

    # -- reading --------------------------------------------------------

    def read(self, entry_id: str, *, since: int = 0, limit: int = DEFAULT_LIMIT) -> list[Event]:
        """Events with ``seq > since``, oldest first, at most ``limit`` of them.

        A missing file is an empty list: the entry may simply not have started.
        A malformed or truncated line is skipped - see the module docstring.
        """
        path = self.path(entry_id)
        if not path.is_file():
            return []

        events: list[Event] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = Event.from_json(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    # A half-written final line from a killed writer, or a
                    # line from an older shape. Neither is worth a 500.
                    continue
                if event.seq > since:
                    events.append(event)
                    if len(events) >= limit:
                        break
        return events

    def tail(self, entry_id: str) -> Event | None:
        """The most recent event, or ``None``.

        What the UI shows during a silent two-minute encode, next to the
        elapsed time since it: ``render · clip_02/audio_mix · two-pass loudnorm
        to -14 LUFS · 2m 14s``. That is four true things and no invented ones.
        """
        events = self.read(entry_id, limit=MAX_EVENTS)
        return events[-1] if events else None

    def last_seq(self, entry_id: str) -> int:
        """The highest ``seq`` on disk. ``0`` for a log that does not exist."""
        last = self.tail(entry_id)
        return last.seq if last is not None else 0

    def delete(self, entry_id: str) -> None:
        """Remove one entry's log. Called when its queue row is removed."""
        self.path(entry_id).unlink(missing_ok=True)
        self._seq.pop(entry_id, None)
        self._last.pop(entry_id, None)

    # -- internals ------------------------------------------------------

    def _next_seq(self, entry_id: str) -> int:
        """Next sequence number, recovered from the file the first time.

        Recovered rather than assumed, because the reader and the writer are
        different processes and the writer may be a *restarted* one - phase 2's
        retry appends to the log its first attempt left behind.
        """
        if entry_id not in self._seq:
            self._seq[entry_id] = self.last_seq(entry_id)
        return self._seq[entry_id] + 1


def _ends_mid_line(path: Path) -> bool:
    """True when the file exists, is non-empty, and has no trailing newline."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) != b"\n"


def _clip(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


__all__ = [
    "DEFAULT_LIMIT",
    "EVENT_KINDS",
    "MAX_DETAIL_CHARS",
    "MAX_EVENTS",
    "Event",
    "EventLog",
]
