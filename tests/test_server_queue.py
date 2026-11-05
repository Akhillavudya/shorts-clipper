"""The durable queue (WEB_UI_PLAN.md sections 1 and 8).

No app, no client, no process: :class:`~clipper.server.queue.QueueStore` is a
pure function of one file on disk, and every test here reads that file back
through a *fresh* store. Reading back through the same object would only prove
that a Python attribute kept its value, which is not what "the queue survives a
restart" means.
"""

from __future__ import annotations

import json

import pytest

from clipper.server.queue import (
    QueueBusy,
    QueueEntry,
    QueueStatus,
    QueueStore,
    RunOptions,
)


@pytest.fixture
def queue(tmp_path) -> QueueStore:
    return QueueStore(tmp_path / "jobs")


def reopen(queue: QueueStore) -> QueueStore:
    """A second store over the same directory - a restarted server."""
    return QueueStore(queue.jobs_root)


# -- submission ---------------------------------------------------------


def test_enqueue_persists_and_survives_a_restart(queue):
    """Phase 1's milestone: submit three, restart, all three still queued."""
    sources = [
        "https://www.youtube.com/watch?v=aaa",
        "https://www.youtube.com/watch?v=bbb",
        "C:/videos/local episode.mp4",
    ]
    queue.enqueue_many(sources)

    entries = reopen(queue).load()

    assert [e.source for e in entries] == sources
    assert {e.status for e in entries} == {QueueStatus.queued}
    assert len({e.entry_id for e in entries}) == 3


def test_entry_id_exists_before_the_video_id_does(queue):
    """Identity at submit time is the entry, because all we have is a URL."""
    entry = queue.enqueue("https://www.youtube.com/watch?v=aaa")

    assert entry.entry_id
    assert entry.video_id is None
    assert entry.label == entry.source

    queue.update(entry.entry_id, video_id="aaa")
    assert reopen(queue).get(entry.entry_id).label == "aaa"


def test_queue_file_is_created_lazily(tmp_path):
    """Constructing a store touches nothing - see ``review/app.py``'s comment."""
    queue = QueueStore(tmp_path / "jobs")

    assert queue.load() == []
    assert not queue.dir.exists()

    queue.enqueue("https://example.test/a")
    assert queue.path.is_file()


def test_next_queued_is_strict_fifo(queue):
    first, second = queue.enqueue_many(["a", "b"])
    assert queue.next_queued().entry_id == first.entry_id

    queue.update(first.entry_id, status=QueueStatus.running)
    assert queue.next_queued().entry_id == second.entry_id

    queue.update(second.entry_id, status=QueueStatus.done)
    assert queue.next_queued() is None


def test_options_round_trip_through_the_file(queue):
    entry = queue.enqueue(
        "https://example.test/a",
        RunOptions(rights="GREEN", music="lofi.wav", centre_crop=True, clip_index=3),
    )

    stored = reopen(queue).get(entry.entry_id)

    assert stored.options.rights == "green"  # normalised by the validator
    assert stored.options.music == "lofi.wav"
    assert stored.options.centre_crop is True
    assert stored.options.clip_index == 3
    assert stored.options.allow_red is False


def test_unknown_rights_is_refused():
    with pytest.raises(ValueError, match="is not one of"):
        RunOptions(rights="probably fine")


def test_unknown_option_is_refused():
    """``extra='forbid'``: a typo'd flag is an error, not a silently ignored one."""
    with pytest.raises(ValueError):
        RunOptions(alow_red=True)


def test_enqueue_many_gives_each_entry_its_own_options(queue):
    first, second = queue.enqueue_many(["a", "b"], RunOptions(force=True))
    queue.update(first.entry_id, options=RunOptions(force=False))

    assert reopen(queue).get(second.entry_id).options.force is True


# -- persistence --------------------------------------------------------


def test_save_is_atomic_and_leaves_no_temp_files(queue):
    queue.enqueue_many(["a", "b", "c"])

    leftovers = [p.name for p in queue.dir.iterdir() if p.name.startswith(".queue.")]
    assert leftovers == []


def test_the_file_is_readable_json_with_a_version(queue):
    """``cat queue.json`` is the debugging story, exactly as for state.json."""
    queue.enqueue("https://example.test/a")

    data = json.loads(queue.path.read_text(encoding="utf-8"))

    assert data["version"] == 1
    assert data["entries"][0]["source"] == "https://example.test/a"
    assert data["entries"][0]["status"] == "queued"


def test_a_corrupt_queue_file_raises_rather_than_reading_as_empty(queue):
    queue.enqueue("https://example.test/a")
    queue.path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):
        reopen(queue).load()


def test_update_validates_what_it_is_given(queue):
    entry = queue.enqueue("https://example.test/a")

    with pytest.raises(ValueError):
        queue.update(entry.entry_id, status="nearly done")

    assert reopen(queue).get(entry.entry_id).status is QueueStatus.queued


def test_update_of_a_missing_entry_raises(queue):
    with pytest.raises(KeyError):
        queue.update("nope", status=QueueStatus.done)


# -- the running entry --------------------------------------------------


def test_running_video_id_is_the_single_writer_check(queue):
    entry = queue.enqueue("https://example.test/a")
    assert queue.running_video_id() is None

    queue.update(entry.entry_id, status=QueueStatus.running, video_id="vid123", pid=4242)
    assert queue.running_video_id() == "vid123"

    queue.update(entry.entry_id, status=QueueStatus.done, pid=None)
    assert queue.running_video_id() is None


def test_running_video_id_is_none_while_the_video_is_unresolved(queue):
    """A running entry before ingest owns no state file, so nothing is locked."""
    entry = queue.enqueue("https://example.test/a")
    queue.update(entry.entry_id, status=QueueStatus.running, pid=1)

    assert queue.running() is not None
    assert queue.running_video_id() is None


def test_entry_for_video_returns_the_most_recent(queue):
    first = queue.enqueue("https://example.test/a")
    queue.update(first.entry_id, video_id="vid123", status=QueueStatus.done)
    second = queue.enqueue("https://example.test/a")
    queue.update(second.entry_id, video_id="vid123", status=QueueStatus.queued)

    assert queue.entry_for_video("vid123").entry_id == second.entry_id
    assert queue.entry_for_video("other") is None


# -- removal ------------------------------------------------------------


def test_remove_drops_a_queued_entry_and_its_cancel_sentinel(queue):
    first, second = queue.enqueue_many(["a", "b"])
    queue.cancel_path(first.entry_id).parent.mkdir(parents=True, exist_ok=True)
    queue.cancel_path(first.entry_id).write_text("", encoding="utf-8")

    queue.remove(first.entry_id)

    assert [e.entry_id for e in reopen(queue).load()] == [second.entry_id]
    assert not queue.cancel_path(first.entry_id).exists()


def test_remove_refuses_the_running_entry(queue):
    entry = queue.enqueue("https://example.test/a")
    queue.update(entry.entry_id, status=QueueStatus.running, pid=4242)

    with pytest.raises(QueueBusy):
        queue.remove(entry.entry_id)

    assert reopen(queue).get(entry.entry_id) is not None


def test_remove_of_a_missing_entry_raises(queue):
    with pytest.raises(KeyError):
        queue.remove("nope")


def test_clear_finished_keeps_queued_and_running(queue):
    done, running, waiting = queue.enqueue_many(["a", "b", "c"])
    queue.update(done.entry_id, status=QueueStatus.failed)
    queue.update(running.entry_id, status=QueueStatus.running, pid=1)

    removed = queue.clear_finished()

    assert [e.entry_id for e in removed] == [done.entry_id]
    assert [e.entry_id for e in reopen(queue).load()] == [running.entry_id, waiting.entry_id]


# -- restart recovery ---------------------------------------------------


def test_reclaim_turns_running_with_a_dead_pid_into_interrupted(queue):
    alive, dead, finished = queue.enqueue_many(["a", "b", "c"])
    queue.update(alive.entry_id, status=QueueStatus.running, pid=111)
    queue.update(dead.entry_id, status=QueueStatus.running, pid=222)
    queue.update(finished.entry_id, status=QueueStatus.done)

    reclaimed = queue.reclaim(is_alive=lambda pid: pid == 111)

    assert [e.entry_id for e in reclaimed] == [dead.entry_id]

    after = {e.entry_id: e for e in reopen(queue).load()}
    assert after[alive.entry_id].status is QueueStatus.running
    assert after[dead.entry_id].status is QueueStatus.interrupted
    assert after[dead.entry_id].pid is None
    assert after[dead.entry_id].finished_at is not None
    assert "server stopped" in after[dead.entry_id].error
    assert after[finished.entry_id].status is QueueStatus.done


def test_reclaim_takes_a_running_entry_that_never_recorded_a_pid(queue):
    """Killed between being marked running and the process being recorded."""
    entry = queue.enqueue("https://example.test/a")
    queue.update(entry.entry_id, status=QueueStatus.running)

    reclaimed = queue.reclaim(is_alive=lambda pid: True)

    assert [e.entry_id for e in reclaimed] == [entry.entry_id]


def test_reclaim_writes_nothing_when_there_is_nothing_to_reclaim(queue):
    queue.enqueue("https://example.test/a")
    before = queue.path.read_bytes()

    assert queue.reclaim(is_alive=lambda pid: True) == []
    assert queue.path.read_bytes() == before


def test_interrupted_is_not_active_but_queued_and_running_are():
    def entry(status):
        return QueueEntry(source="a", status=status)

    assert entry(QueueStatus.queued).is_active
    assert entry(QueueStatus.running).is_active
    assert not entry(QueueStatus.interrupted).is_active
    assert not entry(QueueStatus.blocked).is_active
