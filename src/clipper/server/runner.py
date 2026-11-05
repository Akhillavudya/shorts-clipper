"""The job supervisor (WEB_UI_PLAN.md section 1).

One thread inside the Flask process that spawns and reaps worker children and
does no work itself::

    Flask process (clipper ui)
    ├── Werkzeug threaded server        - API, always responsive
    └── supervisor thread (JobRunner)   - tick() loop; spawns and reaps
            └── Popen: python -m clipper.server.worker --entry <id> ...
                    └── pipeline stages -> ffmpeg / yt-dlp children

**``tick()`` is the design.** One call is one unit of supervision: reap a
finished child, or stamp liveness on a running one, or start the next queued
entry. :meth:`JobRunner.start` is a trivial ``while not stopped: tick();
sleep`` around it, and every test drives ``tick()`` directly with a
:class:`FakeClock` and a fake process. No thread, no sleeping, no real
subprocess anywhere in the suite - which is what makes the hardest-to-test
component in the project the one with the most tests.

**One job at a time.** Not a limitation to be lifted later: two renders on one
desktop are slower than two in sequence *and* make the state-file race in
section 3 real again. :meth:`QueueStore.running` is singular for the same
reason.

**The supervisor owns ``queue.json`` and nothing else.** It never writes a
job's ``state.json`` - the worker does - and it never writes an event log
except through the queue routes' one ``submitted`` line. It learns the video id
and the rights status by *reading* the events its child wrote. That is the
whole single-writer discipline, and it is why there is no lock in this package.

### Cancellation is graceful first, then not

``cancel()`` writes ``jobs/_queue/<entry_id>.cancel``. The worker checks for it
between stages and exits ``130`` cleanly, leaving a coherent ``state.json``. A
cancel that arrives three minutes into a ``-preset slow`` encode cannot be
caught there, so :data:`CANCEL_GRACE_S` later the tree is killed.

Killing the *tree* rather than the process: ``proc.terminate()`` takes down the
Python child and orphans ffmpeg and yt-dlp, which keep a handle on
``source.mp4`` and keep burning CPU with nothing tracking them. On Windows that
is ``taskkill /T /F``; on POSIX it is a process group and ``SIGKILL``.

A hard kill leaves the stage marked ``running`` in ``state.json``. That is
correct and needs no change anywhere: ``should_run`` already treats a
``running`` stage as one to rerun, which is exactly how a crash is told apart
from a stage that never started.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from clipper.server.events import EventLog
from clipper.server.queue import QueueBusy, QueueEntry, QueueStatus, QueueStore
from clipper.server.text import clean_error
from clipper.server.worker import (
    EXIT_BLOCKED,
    EXIT_CANCELLED,
    EXIT_FAILED,
    EXIT_OK,
)
from clipper.state import utcnow

log = logging.getLogger(__name__)

#: How long a worker gets to notice the cancel sentinel before its tree is
#: killed. Ten seconds is long enough for a stage boundary and short enough
#: that "cancel" still feels like a button rather than a request.
CANCEL_GRACE_S = 10.0

#: How long the supervisor sleeps between ticks. Fast enough that a job starts
#: the moment you submit it, slow enough to be free.
POLL_INTERVAL_S = 0.25

#: Refuse to start a job with less than this free on the jobs volume. A 1440p
#: source is 0.5-2 GB and nothing deletes it; failing three stages in with a
#: half-written file is a worse way to find that out than not starting.
MIN_FREE_BYTES = 10 * 1024**3

#: Windows exit codes that mean the process died rather than returned. They are
#: reported as huge unsigned values by ``taskkill``-adjacent paths and as
#: negative numbers by ``Popen.returncode``, so both forms are recognised.
_ACCESS_VIOLATION = 0xC0000005


class ProcessHandle(Protocol):
    """What the supervisor needs from a child. A real ``Popen`` or a fake."""

    pid: int

    def poll(self) -> int | None:
        """Exit code, or ``None`` while it is still running."""

    def kill_tree(self, *, timeout_s: float = CANCEL_GRACE_S) -> None:
        """Kill the process and everything it spawned."""


Spawn = Callable[[QueueEntry], ProcessHandle]


# --------------------------------------------------------------------------
# process plumbing - the only part that touches the OS
# --------------------------------------------------------------------------


def _creation_flags() -> int:
    """``CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`` on Windows, else 0.

    The new process group is what makes a tree kill possible; ``CREATE_NO_
    WINDOW`` is what stops a console flashing on the desktop every time a job
    starts or is cancelled. That matters specifically because the whole feature
    is "open a page and walk away" - a window blinking into existence on its
    own is alarming, not informative.
    """
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


def is_alive(pid: int) -> bool:
    """Whether a pid is a live process. Used by ``QueueStore.reclaim``.

    ``os.kill(pid, 0)`` is the POSIX idiom and is **actively dangerous on
    Windows**, where CPython implements ``os.kill`` for any signal other than
    the two console events as ``TerminateProcess(handle, sig)`` - so the
    liveness probe would kill the thing it was asking about. Hence the two
    implementations.
    """
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        return _is_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, and owned by somebody else. Not ours to reclaim either way.
        return True
    return True


def _is_alive_windows(pid: int) -> bool:
    """``OpenProcess`` + ``GetExitCodeProcess``.

    ``STILL_ACTIVE`` (259) is famously ambiguous with a process that exited
    *with* code 259. That only matters to ``reclaim``, where the cost of the
    ambiguity is leaving one row as ``running`` for one extra startup rather
    than marking it ``interrupted``, and the worker's exit codes are 0, 1, 3
    and 130.
    """
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def kill_tree(pid: int, *, timeout_s: float = CANCEL_GRACE_S) -> None:
    """Kill a process and every child it spawned. Never raises.

    ffmpeg and yt-dlp are grandchildren of this process, and they are the ones
    holding gigabyte file handles. Killing only the Python child leaves them
    running with nothing tracking them, which is both a CPU leak and a locked
    ``source.mp4`` that the next attempt cannot overwrite.
    """
    if pid is None or pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=timeout_s,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("kill_tree(%s) failed, assuming already gone: %s", pid, exc)


class WorkerProcess:
    """A real ``Popen``, wearing the :class:`ProcessHandle` shape."""

    def __init__(self, proc: subprocess.Popen, *, log_file: Any = None) -> None:
        self._proc = proc
        self._log_file = log_file
        self.pid = proc.pid

    def poll(self) -> int | None:
        code = self._proc.poll()
        if code is not None and self._log_file is not None:
            # Closed on reap rather than in a ``finally``: the child inherits
            # the handle, and closing it while it is still writing truncates
            # the log the failure is being diagnosed from.
            try:
                self._log_file.close()
            finally:
                self._log_file = None
        return code

    def kill_tree(self, *, timeout_s: float = CANCEL_GRACE_S) -> None:
        kill_tree(self.pid, timeout_s=timeout_s)


def spawn_worker(
    entry: QueueEntry,
    *,
    repo_root: Path,
    config_path: Path,
    jobs_root: Path,
    log_path: Path,
) -> ProcessHandle:
    """Start one worker child for ``entry``.

    Every path handed across the boundary is absolute and every argument is a
    list element. Section 7: ``config.CONFIG_DIR`` is a *relative* ``Path`` and
    this repo lives under a directory with a space in its name, so relying on
    CWD inheritance or on shell quoting are two different ways to fail on this
    exact machine. ``cwd`` is set as well, belt and braces, because ffmpeg's
    filter escaping in ``media.py`` depends on it.

    stdout and stderr both land in ``log_path`` - yt-dlp's and ffmpeg's chatter
    is what makes "it failed" into "I can see why", and it does not belong in
    the structured event log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")

    argv = [
        sys.executable,
        "-m",
        "clipper.server.worker",
        "--entry",
        entry.entry_id,
        "--source",
        entry.source,
        "--config",
        str(Path(config_path).resolve()),
        "--jobs-root",
        str(Path(jobs_root).resolve()),
        "--options",
        entry.options.model_dump_json(),
    ]

    kwargs: dict[str, Any] = {
        "cwd": str(Path(repo_root).resolve()),
        "stdout": handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": os.environ.copy(),
        "creationflags": _creation_flags(),
    }
    if os.name != "nt":
        # A session of its own, so ``os.killpg`` reaches ffmpeg as well.
        kwargs.pop("creationflags")
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(argv, **kwargs)  # noqa: S603 - argv list, no shell
    except BaseException:
        handle.close()
        raise
    return WorkerProcess(proc, log_file=handle)


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------


class JobRunner:
    """Drains the queue, one job at a time.

    ``spawn`` and ``clock`` are injected for the same reason the limiter's are:
    the interesting behaviour is all timing and process lifecycle, and a test
    that had to start a real interpreter to assert "a crashed job does not stop
    the next one" would be too slow and too flaky to write.
    """

    def __init__(
        self,
        *,
        queue: QueueStore,
        events: EventLog,
        repo_root: Path | str | None = None,
        config_path: Path | str | None = None,
        spawn: Spawn | None = None,
        clock: Any = None,
        min_free_bytes: int = MIN_FREE_BYTES,
        poll_interval_s: float = POLL_INTERVAL_S,
        cancel_grace_s: float = CANCEL_GRACE_S,
    ) -> None:
        self.queue = queue
        self.events = events
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.config_path = (
            Path(config_path) if config_path is not None else Path("config/settings.yaml")
        )
        self._spawn = spawn or self._default_spawn
        self._clock = clock or time
        self.min_free_bytes = min_free_bytes
        self.poll_interval_s = poll_interval_s
        self.cancel_grace_s = cancel_grace_s

        self._handle: ProcessHandle | None = None
        self._entry_id: str | None = None
        #: Monotonic time the cancel sentinel was written, if it was.
        self._cancel_at: float | None = None
        #: Highest event seq already copied onto the queue row.
        self._seen_seq = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------

    @property
    def alive(self) -> bool:
        """Whether the supervisor thread is running.

        ``GET /api/queue`` reports this so the page can say "nothing is
        draining the queue" rather than looking idle - which is the difference
        between a bug and a state.
        """
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_entry_id(self) -> str | None:
        return self._entry_id

    def start(self) -> None:
        """Reclaim orphans, then run ``tick()`` on a daemon thread.

        :meth:`QueueStore.reclaim` runs first and exactly here: an entry left
        ``running`` with a dead pid means the server and its worker died
        together, and every stage is idempotent, so the row becomes
        ``interrupted`` with a retry that resumes rather than restarts.
        """
        if self.alive:
            return
        self.reclaim()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="clipper-runner", daemon=True)
        self._thread.start()

    def reclaim(self, *, stop_orphans: bool = False) -> list[QueueEntry]:
        """Turn entries whose worker is gone into ``interrupted``. Idempotent.

        Called by :meth:`start`, and a method of its own so it can be driven
        without a thread. An entry left ``running`` with a dead pid means the
        server and its worker died together - a lid, a power cut, a Ctrl-C on
        the wrong window. ``interrupted`` says exactly that, and because every
        stage is idempotent the retry resumes rather than restarts.
        """
        # Only the server holding RunnerClaim may stop a surviving old worker.
        # A force-killed server cannot run its finally block; leaving its child
        # alive would make the queue permanently running without a supervisor.
        if stop_orphans:
            for entry in self.queue.load():
                if entry.status == QueueStatus.running and entry.pid and is_alive(entry.pid):
                    kill_tree(entry.pid)
                    if is_alive(entry.pid):
                        raise RuntimeError(
                            "An old worker is still running; stop it before restarting."
                        )
        reclaimed = self.queue.reclaim(is_alive=is_alive)
        for entry in reclaimed:
            self.events.append(
                entry.entry_id, "error", detail="the server stopped while this job was running"
            )
        return reclaimed

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad tick must not end the loop
                log.exception("supervisor tick failed")
            self._stop.wait(self.poll_interval_s)

    def tick(self) -> None:
        # Only short server decisions are serialized, never worker execution.
        with self.queue.lock:
            self._tick()

    def _tick(self) -> None:
        """One unit of supervision. Safe to call as often as you like.

        Three mutually exclusive outcomes: reap a finished child, watch a
        running one, or start the next queued entry.
        """
        if self._stop.is_set():
            return
        if self._handle is not None:
            code = self._handle.poll()
            if code is None:
                self._watch()
            else:
                self._reap(code)
            return
        self._maybe_start()

    # -- running ---------------------------------------------------------

    def _watch(self) -> None:
        """Stamp liveness, copy new facts off the event log, kill if overdue.

        ``last_alive`` is a *liveness* claim and deliberately a weaker one than
        progress: it is stamped by the supervisor, which is not blocked inside
        ffmpeg, so the page can honestly say "process alive, 2m 14s since the
        last progress line". A synthetic percentage would be neither true nor
        checkable.
        """
        assert self._entry_id is not None
        fields: dict[str, Any] = {"last_alive": utcnow()}
        fields.update(self._drain_events(self._entry_id))
        try:
            self._update(self._entry_id, **fields)
        except KeyError:
            # The row was deleted out from under a live worker. Refusing to
            # leave an orphan holding source.mp4 matters more than the row.
            self._handle.kill_tree(timeout_s=self.cancel_grace_s)
            self._handle = None
            self._entry_id = None
            self._cancel_at = None
            self._seen_seq = 0
            return

        if self._cancel_at is not None and self._elapsed(self._cancel_at) >= self.cancel_grace_s:
            log.info("cancel grace expired for %s, killing the tree", self._entry_id)
            self._handle.kill_tree(timeout_s=self.cancel_grace_s)
            # Not marked cancelled here: the reap on the next tick does that,
            # so there is one place that decides a terminal status.
            self._cancel_at = None

    def _drain_events(self, entry_id: str) -> dict[str, Any]:
        """Read what the child has learned and hand it back as queue fields.

        ``resolved`` carries the video id, which does not exist until ``ingest``
        has made a network metadata call, and ``rights`` carries the status the
        gate produced. Both are read rather than shared, because the two
        processes share nothing but files.
        """
        fields: dict[str, Any] = {}
        for event in self.events.read(entry_id, since=self._seen_seq):
            self._seen_seq = event.seq
            if event.kind == "resolved" and event.detail:
                fields["video_id"] = event.detail
            elif event.kind in {"rights", "blocked"} and event.stage:
                fields["rights_status"] = event.stage
        return fields

    # -- reaping ---------------------------------------------------------

    def _reap(self, code: int) -> None:
        """Turn an exit code into a terminal queue status."""
        entry_id = self._entry_id
        assert entry_id is not None

        fields: dict[str, Any] = self._drain_events(entry_id)
        fields.update(
            {
                "status": _status_for(code),
                "exit_code": code,
                "finished_at": utcnow(),
                "pid": None,
            }
        )
        fields["error"] = self._error_for(entry_id, code)

        self._handle = None
        self._entry_id = None
        self._cancel_at = None
        self._seen_seq = 0

        self.queue.cancel_path(entry_id).unlink(missing_ok=True)
        self._update(entry_id, **fields)
        log.info("worker %s finished: status=%s exit=%s", entry_id, fields["status"], code)

    def _error_for(self, entry_id: str, code: int) -> str | None:
        """What to show on a failed row.

        A crash - an access violation out of MediaPipe, or a tree kill - leaves
        no Python exception anywhere, so the exit code has to be read out loud.
        Anything else quotes the worker's own ``error`` event, which is already
        stripped and truncated.
        """
        if code in {EXIT_OK, EXIT_BLOCKED, EXIT_CANCELLED}:
            return None

        for event in reversed(self.events.read(entry_id, limit=2000)):
            if event.kind == "error" and event.detail:
                return clean_error(event.detail)

        return _crash_message(code)

    # -- starting --------------------------------------------------------

    def _maybe_start(self) -> None:
        entry = self.queue.next_queued()
        if entry is None:
            return

        free = self._free_bytes()
        if free is not None and free < self.min_free_bytes:
            # Refused rather than attempted. A 1440p source is 0.5-2 GB, and
            # discovering the disk is full three stages in leaves a truncated
            # file and a job that has to be forced to recover.
            detail = (
                f"{free / 1024**3:.1f} GB free on {self.queue.jobs_root}, "
                f"below the {self.min_free_bytes / 1024**3:.0f} GB floor"
            )
            self.events.append(entry.entry_id, "blocked", stage="disk", detail=detail)
            self._update(
                entry.entry_id,
                status=QueueStatus.blocked,
                finished_at=utcnow(),
                error=detail,
            )
            return

        try:
            handle = self._spawn(entry)
        except Exception as exc:  # noqa: BLE001 - a failed spawn is a failed job
            log.exception("could not start a worker for %s", entry.entry_id)
            self.events.error(entry.entry_id, f"could not start a worker: {exc}")
            self._update(
                entry.entry_id,
                status=QueueStatus.failed,
                finished_at=utcnow(),
                error=clean_error(str(exc)),
            )
            return

        self._handle = handle
        self._entry_id = entry.entry_id
        log.info("worker %s started: pid=%s", entry.entry_id, handle.pid)
        self._cancel_at = None
        self._seen_seq = self.events.last_seq(entry.entry_id)

        self._update(
            entry.entry_id,
            status=QueueStatus.running,
            pid=handle.pid,
            started_at=utcnow(),
            last_alive=utcnow(),
            error=None,
            exit_code=None,
        )

    def _default_spawn(self, entry: QueueEntry) -> ProcessHandle:
        return spawn_worker(
            entry,
            repo_root=self.repo_root,
            config_path=self.config_path,
            jobs_root=self.queue.jobs_root,
            log_path=self.queue.log_path(entry.entry_id),
        )

    # -- cancellation ----------------------------------------------------

    def cancel(self, entry_id: str) -> QueueEntry:
        """Stop an entry. Graceful if it is running, immediate if it is queued.

        A queued entry has no process, so it goes straight to ``cancelled``. A
        running one gets the sentinel; the worker exits ``130`` at the next
        stage boundary, and if it is deep inside an encode the next ticks kill
        the tree once :data:`CANCEL_GRACE_S` has passed.
        """
        entry = self.queue.get(entry_id)
        if entry is None:
            raise KeyError(f"no queue entry {entry_id!r}")

        if entry.status is QueueStatus.queued:
            self.events.append(entry_id, "cancelled", detail="cancelled before it started")
            return self._update(
                entry_id, status=QueueStatus.cancelled, finished_at=utcnow(), pid=None
            )

        if entry.status is not QueueStatus.running:
            raise QueueBusy(f"{entry_id} is {entry.status}, not running")

        path = self.queue.cancel_path(entry_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(utcnow().isoformat(), encoding="utf-8")
        self.events.append(entry_id, "cancelled", detail="cancel requested")

        if self._entry_id == entry_id:
            self._cancel_at = self._now()
        else:
            # Running according to the queue but not ours - a worker this
            # supervisor did not spawn, which reclaim will catch. The sentinel
            # is still the right thing to leave behind.
            log.warning("cancel for %s, which this supervisor does not own", entry_id)

        return self.queue.get(entry_id) or entry

    def retry(self, entry_id: str, *, force: bool = False) -> QueueEntry:
        """Put a finished entry back at the end of the queue, in place.

        The same ``entry_id``, deliberately: its event log is the history of
        this submission, and a retry that appended to a fresh file would leave
        the first attempt's failure orphaned from the row that explains it.
        ``events.py`` recovers the sequence counter from the file for exactly
        this.

        Stages are not reset here. Every one of them is idempotent and
        ``should_run`` skips the completed ones, so a retry resumes rather than
        restarts - which is the whole reason a 40-minute job can be interrupted
        without being lost. ``force`` is how you ask for the other thing.
        """
        entry = self.queue.get(entry_id)
        if entry is None:
            raise KeyError(f"no queue entry {entry_id!r}")
        if entry.status in {QueueStatus.running, QueueStatus.queued}:
            raise QueueBusy(f"{entry_id} is already {entry.status}")

        options = entry.options.model_copy(update={"force": force})
        self.queue.cancel_path(entry_id).unlink(missing_ok=True)
        return self._update(
            entry_id,
            status=QueueStatus.queued,
            options=options.model_dump(),
            started_at=None,
            finished_at=None,
            last_alive=None,
            pid=None,
            error=None,
            exit_code=None,
        )

    # -- shutdown --------------------------------------------------------

    def shutdown(self, *, timeout_s: float | None = None) -> None:
        """Stop accepting work and take the current job down with us.

        Ctrl-C must not orphan a twenty-minute render. The sentinel goes down
        first, the child gets ``timeout_s`` to exit cleanly, and then the tree
        is killed.

        The entry is marked **``interrupted``**, not ``cancelled``: nobody
        asked for this one, and the retry button should say so. The stage stays
        ``running`` in ``state.json``, which ``should_run`` already reads as
        "rerun me".

        Killing rather than detaching is deliberate. An orphaned worker still
        writing ``state.json`` after the server is gone is the second-writer
        problem of section 3, with nobody watching.
        """
        timeout_s = self.cancel_grace_s if timeout_s is None else timeout_s
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

        # Wait for any in-flight spawn decision before inspecting ownership.
        with self.queue.lock:
            handle, entry_id = self._handle, self._entry_id
        if handle is None or entry_id is None:
            return

        path = self.queue.cancel_path(entry_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(utcnow().isoformat(), encoding="utf-8")

        deadline = self._now() + timeout_s
        while handle.poll() is None and self._now() < deadline:
            self._clock.sleep(min(self.poll_interval_s, timeout_s))
        if handle.poll() is None:
            handle.kill_tree(timeout_s=timeout_s)
        handle.poll()  # closes the worker log handle after exit

        self._handle = None
        self._entry_id = None
        self._cancel_at = None
        self._seen_seq = 0

        path.unlink(missing_ok=True)
        self.events.append(entry_id, "cancelled", detail="the server shut down")
        self._update(
            entry_id,
            status=QueueStatus.interrupted,
            finished_at=utcnow(),
            pid=None,
            error="the server shut down while this job was running",
        )

    # -- helpers ---------------------------------------------------------

    def _now(self) -> float:
        return self._clock.time()

    def _elapsed(self, since: float) -> float:
        return self._now() - since

    def _free_bytes(self) -> int | None:
        root = self.queue.jobs_root
        try:
            probe = root if root.is_dir() else (root.anchor or ".")
            return shutil.disk_usage(probe).free
        except OSError:
            # A drive letter that went away. Refusing to start on "cannot
            # tell" would stop the queue for the wrong reason.
            return None

    def _update(self, entry_id: str, **fields: Any) -> QueueEntry:
        """Persist queue fields, tolerating a row that was deleted underneath.

        The API can remove a row while the supervisor is mid-tick. That is not
        an error worth taking the thread down for - the next tick finds no
        handle and moves on.
        """
        try:
            return self.queue.update(entry_id, **fields)
        except KeyError:
            log.warning("queue entry %s vanished while updating it", entry_id)
            raise


def _status_for(code: int) -> QueueStatus:
    """Exit code to terminal status. The contract from section 1."""
    if code == EXIT_OK:
        return QueueStatus.done
    if code == EXIT_BLOCKED:
        return QueueStatus.blocked
    if code == EXIT_CANCELLED:
        return QueueStatus.cancelled
    return QueueStatus.failed


def _crash_message(code: int) -> str:
    """A readable line for an exit code with no exception behind it.

    ``0xC0000005`` is the access violation a MediaPipe or OpenCV fault
    produces, and it is the single most likely way this pipeline dies without
    saying anything. Naming it is the difference between "exit code
    -1073741819" and knowing which stage to suspect.
    """
    unsigned = code & 0xFFFFFFFF if code < 0 else code
    if unsigned == _ACCESS_VIOLATION:
        return (
            "the worker crashed with an access violation (0xC0000005) - a native "
            "fault, most likely in face tracking. Retry with --centre-crop if it "
            "repeats. See the raw log."
        )
    if code < 0:
        return f"the worker was killed by signal {-code}. See the raw log."
    return f"the worker exited with code {code} (expected 0). See the raw log."


__all__ = [
    "CANCEL_GRACE_S",
    "EXIT_BLOCKED",
    "EXIT_CANCELLED",
    "EXIT_FAILED",
    "EXIT_OK",
    "MIN_FREE_BYTES",
    "POLL_INTERVAL_S",
    "JobRunner",
    "ProcessHandle",
    "WorkerProcess",
    "is_alive",
    "kill_tree",
    "spawn_worker",
]
