"""The job supervisor (WEB_UI_PLAN.md sections 1, 8 and 9).

The critical file. Everything hard about phase 2 - process lifecycle,
cancellation, crash isolation, restart recovery - lives in ``runner.py``, and
all of it is reachable without starting a process or sleeping, because
``tick()`` is one unit of supervision and both ``spawn`` and ``clock`` are
injected.

``FakeProcess`` implements :class:`~clipper.server.runner.ProcessHandle` and
nothing else. That it is a complete stand-in for a ``Popen`` is the evidence
the protocol is the right size.
"""

from __future__ import annotations

import pytest

from clipper.server.events import EventLog
from clipper.server.queue import QueueBusy, QueueStatus, QueueStore, RunOptions
from clipper.server.runner import JobRunner, is_alive, kill_tree
from clipper.server.worker import EXIT_BLOCKED, EXIT_CANCELLED, EXIT_FAILED, EXIT_OK

#: The exit code a native fault in MediaPipe or OpenCV produces on Windows,
#: as ``Popen.returncode`` reports it.
ACCESS_VIOLATION = -1073741819


class FakeProcess:
    """A ``ProcessHandle`` that exits when a test says so."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.code: int | None = None
        self.killed = False
        self.kill_count = 0

    def poll(self) -> int | None:
        return self.code

    def kill_tree(self, *, timeout_s: float = 10.0) -> None:
        self.killed = True
        self.kill_count += 1
        # A real tree kill is what finally produces an exit code, so the fake
        # produces one too. Without it the reap would never happen and the
        # cancel tests would pass for the wrong reason.
        if self.code is None:
            self.code = EXIT_CANCELLED

    def exit(self, code: int) -> None:
        self.code = code


@pytest.fixture
def jobs_root(tmp_path):
    return tmp_path / "jobs"


@pytest.fixture
def queue(jobs_root) -> QueueStore:
    return QueueStore(jobs_root)


@pytest.fixture
def events(jobs_root) -> EventLog:
    return EventLog(jobs_root)


@pytest.fixture
def spawned() -> list[FakeProcess]:
    return []


@pytest.fixture
def runner(queue, events, spawned, clock, tmp_path):
    """A supervisor whose children are fakes and whose clock does not tick."""

    def spawn(entry):
        proc = FakeProcess(pid=4000 + len(spawned))
        proc.entry = entry
        spawned.append(proc)
        return proc

    return JobRunner(
        queue=queue,
        events=events,
        spawn=spawn,
        clock=clock,
        repo_root=tmp_path,
        config_path=tmp_path / "config" / "settings.yaml",
        # Zero rather than the 10 GB default: the temp directory's volume has
        # whatever it has, and a disk floor is tested explicitly below.
        min_free_bytes=0,
    )


def status(queue: QueueStore, entry_id: str) -> QueueStatus:
    entry = queue.get(entry_id)
    assert entry is not None
    return entry.status


# -- starting ---------------------------------------------------------------


def test_an_empty_queue_is_a_no_op(runner, spawned):
    runner.tick()
    runner.tick()
    assert spawned == []


def test_a_tick_starts_the_oldest_queued_entry(runner, queue, spawned):
    first = queue.enqueue("https://youtu.be/one")
    second = queue.enqueue("https://youtu.be/two")

    runner.tick()

    assert [p.entry.entry_id for p in spawned] == [first.entry_id]
    assert status(queue, first.entry_id) is QueueStatus.running
    assert status(queue, second.entry_id) is QueueStatus.queued


def test_starting_records_the_pid_and_the_times(runner, queue):
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()

    stored = queue.get(entry.entry_id)
    assert stored.pid == 4000
    assert stored.started_at is not None
    assert stored.last_alive is not None
    assert stored.error is None


def test_only_one_job_runs_at_a_time(runner, queue, spawned):
    """Not a limit to be lifted: two renders on one desktop are slower than two
    in sequence, and concurrent workers make the state-file race real again."""
    queue.enqueue("https://youtu.be/one")
    queue.enqueue("https://youtu.be/two")

    for _ in range(5):
        runner.tick()

    assert len(spawned) == 1


def test_the_options_reach_the_child(runner, queue, spawned):
    queue.enqueue("https://youtu.be/one", RunOptions(centre_crop=True, no_music=True))
    runner.tick()
    assert spawned[0].entry.options.centre_crop is True
    assert spawned[0].entry.options.no_music is True


def test_a_spawn_that_raises_fails_the_entry_rather_than_the_supervisor(
    queue, events, clock, tmp_path
):
    def spawn(entry):
        raise OSError("python is not on PATH")

    runner = JobRunner(queue=queue, events=events, spawn=spawn, clock=clock, min_free_bytes=0)
    entry = queue.enqueue("https://youtu.be/one")

    runner.tick()

    stored = queue.get(entry.entry_id)
    assert stored.status is QueueStatus.failed
    assert "python is not on PATH" in stored.error
    # And the supervisor is still usable.
    runner.tick()


# -- watching ---------------------------------------------------------------


def test_liveness_is_stamped_while_the_child_is_silent(runner, queue, clock):
    """The honest claim during a two-minute encode.

    "Alive, 2m 14s since the last progress line" is true and checkable, which
    is the thing a synthetic percentage is not. The supervisor can make it
    because it is not the process blocked inside ffmpeg.
    """
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    first = queue.get(entry.entry_id).last_alive

    runner.tick()
    assert queue.get(entry.entry_id).last_alive >= first


def test_the_video_id_and_rights_status_are_copied_off_the_event_log(runner, queue, events):
    """The only channel between the two processes is files.

    The row is addressable by ``entry_id`` from the moment you paste a link;
    the video id arrives minutes later when ``ingest`` resolves it, and the
    supervisor learns it by reading what the child wrote.
    """
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()

    events.append(entry.entry_id, "resolved", detail="vid123")
    events.append(entry.entry_id, "rights", stage="amber", detail="permission by email")
    runner.tick()

    stored = queue.get(entry.entry_id)
    assert stored.video_id == "vid123"
    assert stored.rights_status == "amber"


def test_running_video_id_is_what_the_review_endpoint_reads(runner, queue, events):
    """Section 3's single-writer discipline, end to end through the runner."""
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    assert queue.running_video_id() is None

    events.append(entry.entry_id, "resolved", detail="vid123")
    runner.tick()
    assert queue.running_video_id() == "vid123"


# -- reaping ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (EXIT_OK, QueueStatus.done),
        (EXIT_FAILED, QueueStatus.failed),
        (EXIT_BLOCKED, QueueStatus.blocked),
        (EXIT_CANCELLED, QueueStatus.cancelled),
    ],
)
def test_exit_codes_are_the_contract(runner, queue, spawned, code, expected):
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    spawned[0].exit(code)
    runner.tick()

    stored = queue.get(entry.entry_id)
    assert stored.status is expected
    assert stored.exit_code == code
    assert stored.pid is None
    assert stored.finished_at is not None


def test_a_blocked_job_carries_no_error(runner, queue, spawned):
    """``blocked`` is the correct outcome, not a failure, and is coloured as one.

    Hanging an error string off it would put the cheap, right answer in the
    same visual bucket as "ffmpeg died", which trains you to ignore it.
    """
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    spawned[0].exit(EXIT_BLOCKED)
    runner.tick()

    assert queue.get(entry.entry_id).error is None


def test_a_failure_quotes_the_workers_own_error_event(runner, queue, events, spawned):
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    events.error(entry.entry_id, "\x1b[31mRuntimeError: ffmpeg: Invalid data\x1b[0m")
    spawned[0].exit(EXIT_FAILED)
    runner.tick()

    error = queue.get(entry.entry_id).error
    assert error == "RuntimeError: ffmpeg: Invalid data"
    assert "\x1b" not in error


def test_the_next_entry_starts_after_one_finishes(runner, queue, spawned):
    queue.enqueue("https://youtu.be/one")
    second = queue.enqueue("https://youtu.be/two")

    runner.tick()
    spawned[0].exit(EXIT_OK)
    runner.tick()
    runner.tick()

    assert len(spawned) == 2
    assert spawned[1].entry.entry_id == second.entry_id


# -- crash isolation, the whole justification for a subprocess ---------------


def test_a_native_crash_is_named_and_does_not_stop_the_queue(runner, queue, spawned):
    """``0xC0000005`` is the access violation a MediaPipe fault produces.

    Two claims in one test, and they are the entire argument for spawning a
    process instead of running a thread: the crash is reported as a crash
    rather than as a Python exception that never existed, and the job after it
    still starts. In-process, this exit code would have taken the interpreter
    - and the page being watched - down with it.
    """
    crashed = queue.enqueue("https://youtu.be/one")
    following = queue.enqueue("https://youtu.be/two")

    runner.tick()
    spawned[0].exit(ACCESS_VIOLATION)
    runner.tick()

    stored = queue.get(crashed.entry_id)
    assert stored.status is QueueStatus.failed
    assert "0xC0000005" in stored.error
    assert "access violation" in stored.error

    runner.tick()
    assert spawned[1].entry.entry_id == following.entry_id
    assert status(queue, following.entry_id) is QueueStatus.running


# -- cancellation -----------------------------------------------------------


def test_cancelling_a_queued_entry_needs_no_process(runner, queue, spawned):
    entry = queue.enqueue("https://youtu.be/one")
    runner.cancel(entry.entry_id)

    assert status(queue, entry.entry_id) is QueueStatus.cancelled
    runner.tick()
    assert spawned == []


def test_cancelling_a_running_entry_writes_the_sentinel_first(runner, queue, spawned):
    """Graceful first: the worker exits ``130`` at its next stage boundary and
    leaves a coherent ``state.json`` behind."""
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()

    runner.cancel(entry.entry_id)

    assert queue.cancel_path(entry.entry_id).is_file()
    assert spawned[0].killed is False
    assert status(queue, entry.entry_id) is QueueStatus.running


def test_a_worker_that_ignores_the_sentinel_has_its_tree_killed(runner, queue, spawned, clock):
    """A cancel three minutes into a ``-preset slow`` encode cannot be caught
    by the worker at all, so the grace period expires and the tree goes."""
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    runner.cancel(entry.entry_id)

    runner.tick()
    assert spawned[0].killed is False

    clock.advance(11.0)
    runner.tick()
    assert spawned[0].killed is True

    runner.tick()
    assert status(queue, entry.entry_id) is QueueStatus.cancelled


def test_the_sentinel_is_cleaned_up_on_reap(runner, queue, spawned):
    """Left behind, it would cancel the retry of the same entry instantly."""
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    runner.cancel(entry.entry_id)
    spawned[0].exit(EXIT_CANCELLED)
    runner.tick()

    assert not queue.cancel_path(entry.entry_id).is_file()


def test_cancelling_something_finished_is_refused(runner, queue, spawned):
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    spawned[0].exit(EXIT_OK)
    runner.tick()

    with pytest.raises(QueueBusy):
        runner.cancel(entry.entry_id)


def test_cancelling_an_unknown_entry_raises(runner):
    with pytest.raises(KeyError):
        runner.cancel("nope")


# -- retry ------------------------------------------------------------------


def test_retry_requeues_in_place_keeping_the_entry_id(runner, queue, spawned, events):
    """The same id, deliberately: the event log *is* this submission's history,
    and a retry writing to a fresh file orphans the failure from the row that
    explains it."""
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    events.error(entry.entry_id, "boom")
    spawned[0].exit(EXIT_FAILED)
    runner.tick()

    retried = runner.retry(entry.entry_id)

    assert retried.entry_id == entry.entry_id
    assert retried.status is QueueStatus.queued
    assert retried.error is None
    assert retried.exit_code is None
    assert retried.finished_at is None
    assert len(queue.load()) == 1


def test_retry_resumes_rather_than_restarts_unless_forced(runner, queue, spawned):
    """``should_run`` skips completed stages, so a retry of an interrupted job
    does not re-download two gigabytes to find nothing was wrong with them."""
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    spawned[0].exit(EXIT_FAILED)
    runner.tick()

    assert runner.retry(entry.entry_id).options.force is False
    runner.tick()
    spawned[1].exit(EXIT_FAILED)
    runner.tick()
    assert runner.retry(entry.entry_id, force=True).options.force is True


def test_retrying_something_already_queued_is_refused(runner, queue):
    entry = queue.enqueue("https://youtu.be/one")
    with pytest.raises(QueueBusy):
        runner.retry(entry.entry_id)


def test_a_retried_entry_runs_again(runner, queue, spawned):
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()
    spawned[0].exit(EXIT_FAILED)
    runner.tick()

    runner.retry(entry.entry_id)
    runner.tick()

    assert len(spawned) == 2
    assert status(queue, entry.entry_id) is QueueStatus.running


# -- disk preflight ---------------------------------------------------------


def test_a_full_disk_blocks_the_entry_instead_of_starting_it(queue, events, clock, spawned):
    """A 1440p source is 0.5-2 GB and nothing deletes it. Finding that out three
    stages in leaves a truncated file and a job that needs forcing to recover."""

    def spawn(entry):  # pragma: no cover - the point is that it is not called
        proc = FakeProcess()
        spawned.append(proc)
        return proc

    runner = JobRunner(
        queue=queue,
        events=events,
        spawn=spawn,
        clock=clock,
        min_free_bytes=10**18,
    )
    entry = queue.enqueue("https://youtu.be/one")

    runner.tick()

    stored = queue.get(entry.entry_id)
    assert stored.status is QueueStatus.blocked
    assert "GB free" in stored.error
    assert spawned == []

    blocked = next(e for e in events.read(entry.entry_id) if e.kind == "blocked")
    assert blocked.stage == "disk"


# -- restart recovery -------------------------------------------------------


def test_reclaim_turns_a_dead_worker_into_interrupted(runner, queue, events):
    """The server and its worker died together - a lid, a power cut, a Ctrl-C.

    ``interrupted`` is deliberately not ``cancelled``: nobody asked for it, and
    the retry button should say so.
    """
    entry = queue.enqueue("https://youtu.be/one")
    queue.update(entry.entry_id, status=QueueStatus.running, pid=999_999_999)

    reclaimed = runner.reclaim()

    assert [e.entry_id for e in reclaimed] == [entry.entry_id]
    assert status(queue, entry.entry_id) is QueueStatus.interrupted
    assert any(e.kind == "error" for e in events.read(entry.entry_id))


def test_reclaim_leaves_a_queued_entry_alone(runner, queue):
    entry = queue.enqueue("https://youtu.be/one")
    assert runner.reclaim() == []
    assert status(queue, entry.entry_id) is QueueStatus.queued


# -- shutdown ---------------------------------------------------------------


def test_shutdown_kills_the_child_and_marks_it_interrupted(runner, queue, spawned, clock):
    """Ctrl-C must not orphan a twenty-minute render.

    Killing rather than detaching: a worker still writing ``state.json`` after
    the server is gone is section 3's second-writer problem with nobody
    watching.
    """
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()

    runner.shutdown(timeout_s=5.0)

    assert spawned[0].killed is True
    stored = queue.get(entry.entry_id)
    assert stored.status is QueueStatus.interrupted
    assert stored.pid is None
    assert not queue.cancel_path(entry.entry_id).is_file()


def test_shutdown_lets_a_cooperative_worker_exit_on_its_own(
    runner, queue, spawned, clock, monkeypatch
):
    """The sentinel goes down first and the kill is the fallback, not the plan."""
    entry = queue.enqueue("https://youtu.be/one")
    runner.tick()

    proc = spawned[0]
    real_sleep = clock.sleep

    def sleep_and_exit(seconds):
        real_sleep(seconds)
        proc.exit(EXIT_CANCELLED)

    monkeypatch.setattr(clock, "sleep", sleep_and_exit)

    runner.shutdown(timeout_s=5.0)

    assert proc.killed is False
    assert status(queue, entry.entry_id) is QueueStatus.interrupted


def test_shutdown_with_nothing_running_is_a_no_op(runner, queue):
    runner.shutdown()
    assert queue.load() == []


def test_exclusive_restart_stops_surviving_worker(runner, queue, monkeypatch):
    entry = queue.enqueue("source")
    queue.update(entry.entry_id, status="running", pid=12345)
    live = {12345}
    monkeypatch.setattr("clipper.server.runner.is_alive", lambda pid: pid in live)
    monkeypatch.setattr("clipper.server.runner.kill_tree", lambda pid: live.remove(pid))
    runner.reclaim(stop_orphans=True)
    assert not live
    assert queue.get(entry.entry_id).status == QueueStatus.interrupted
    runner.retry(entry.entry_id)
    assert queue.get(entry.entry_id).status == QueueStatus.queued


def test_alive_is_false_until_the_thread_starts(runner):
    """``GET /api/queue`` reports this so the page can say the queue is not
    being drained, rather than sitting there looking idle."""
    assert runner.alive is False
    runner.start()
    try:
        assert runner.alive is True
    finally:
        runner.shutdown()
    assert runner.alive is False


# -- the OS helpers ---------------------------------------------------------


def test_is_alive_says_no_to_a_pid_that_cannot_exist():
    assert is_alive(999_999_999) is False
    assert is_alive(0) is False
    assert is_alive(-1) is False


def test_is_alive_says_yes_to_this_process():
    """And, on Windows, does not kill it.

    ``os.kill(pid, 0)`` is the POSIX idiom and on Windows is implemented as
    ``TerminateProcess``, so the liveness probe would kill what it asked about.
    This test passing at all is the assertion.
    """
    import os

    assert is_alive(os.getpid()) is True


def test_kill_tree_on_a_dead_pid_does_not_raise():
    kill_tree(999_999_999, timeout_s=1.0)
    kill_tree(0)
