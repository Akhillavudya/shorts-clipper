"""The progress event log (WEB_UI_PLAN.md sections 2 and 8).

The two tests that earn their keep here are the ones about the writer being
killable: a truncated final line must be skipped rather than 500 the page, and
a reader attaching to a log it did not start must pick the sequence up where
the file left it. Everything else in this file is the cursor contract the SPA
polls against.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from clipper.server.events import MAX_DETAIL_CHARS, MAX_EVENTS, Event, EventLog


@pytest.fixture
def log(tmp_path) -> EventLog:
    return EventLog(tmp_path / "jobs")


def reopen(log: EventLog) -> EventLog:
    """A second log over the same directory - the API reading the worker's file."""
    return EventLog(log.jobs_root)


ENTRY = "b4c0ffee"


# -- appending ----------------------------------------------------------


def test_append_and_read_round_trip(log):
    log.append(ENTRY, "started", "ingest", "downloading")
    log.append(ENTRY, "progress", "transcribe", "1071 words")

    events = reopen(log).read(ENTRY)

    assert [(e.seq, e.kind, e.stage, e.detail) for e in events] == [
        (1, "started", "ingest", "downloading"),
        (2, "progress", "transcribe", "1071 words"),
    ]
    assert all(e.ts.tzinfo is not None for e in events)


def test_seq_is_monotonic_across_processes(log):
    log.append(ENTRY, "started", "ingest")
    log.append(ENTRY, "progress", "ingest", "downloaded")

    # A restarted worker - phase 2's retry - appends to a log it did not open.
    fresh = reopen(log)
    fresh.append(ENTRY, "progress", "transcribe")

    assert [e.seq for e in reopen(log).read(ENTRY)] == [1, 2, 3]


def test_each_entry_has_its_own_file_and_its_own_sequence(log):
    log.append("one", "started", "ingest")
    log.append("two", "started", "ingest")

    assert [e.seq for e in log.read("one")] == [1]
    assert [e.seq for e in log.read("two")] == [1]
    assert log.path("one") != log.path("two")


def test_reading_a_log_that_does_not_exist_is_empty_not_an_error(log):
    assert log.read("never-started") == []
    assert log.tail("never-started") is None
    assert log.last_seq("never-started") == 0


def test_duplicate_events_are_throttled(log):
    log.append(ENTRY, "progress", "render", "encoding clip_02")
    log.append(ENTRY, "progress", "render", "encoding clip_02")
    log.append(ENTRY, "progress", "render", "encoding clip_03")

    assert [e.detail for e in log.read(ENTRY)] == ["encoding clip_02", "encoding clip_03"]
    assert [e.seq for e in log.read(ENTRY)] == [1, 2]


def test_a_repeat_after_something_else_is_kept(log):
    """Only the immediately preceding event is deduped - a return is real news."""
    log.append(ENTRY, "progress", "render", "clip_01")
    log.append(ENTRY, "progress", "render", "clip_02")
    log.append(ENTRY, "progress", "render", "clip_01")

    assert [e.detail for e in log.read(ENTRY)] == ["clip_01", "clip_02", "clip_01"]


def test_ansi_is_stripped_on_the_way_in(log):
    log.append(ENTRY, "progress", "render", "\x1b[31mffmpeg\x1b[0m started")

    assert log.read(ENTRY)[0].detail == "ffmpeg started"
    assert "\x1b" not in log.path(ENTRY).read_text(encoding="utf-8")


def test_error_events_are_cleaned_and_truncated(log):
    event = log.error(ENTRY, "\x1b[31m" + "x" * 5000, "render")

    assert event.kind == "error"
    assert event.stage == "render"
    assert "\x1b" not in event.detail
    assert len(event.detail) <= MAX_DETAIL_CHARS + 1  # the ellipsis


def test_long_details_are_clipped(log):
    log.append(ENTRY, "progress", "render", "y" * (MAX_DETAIL_CHARS + 500))

    assert len(log.read(ENTRY)[0].detail) == MAX_DETAIL_CHARS + 1


def test_unicode_round_trips(log):
    log.append(ENTRY, "progress", "moments", "found “quote” — 日本語")

    assert reopen(log).read(ENTRY)[0].detail == "found “quote” — 日本語"


def test_the_cap_drops_further_events_rather_than_rotating(log, monkeypatch):
    monkeypatch.setattr("clipper.server.events.MAX_EVENTS", 3)
    for index in range(6):
        log.append(ENTRY, "progress", "render", f"clip_{index}")

    events = log.read(ENTRY)

    # The head survives, because that is where the cause is.
    assert [e.detail for e in events] == ["clip_0", "clip_1", "clip_2"]


# -- the cursor ---------------------------------------------------------


def test_since_returns_only_newer_events(log):
    for index in range(5):
        log.append(ENTRY, "progress", "render", f"clip_{index}")

    events = reopen(log).read(ENTRY, since=3)

    assert [e.seq for e in events] == [4, 5]


def test_limit_caps_the_page_without_skipping(log):
    for index in range(10):
        log.append(ENTRY, "progress", "render", f"clip_{index}")

    first = log.read(ENTRY, since=0, limit=4)
    second = log.read(ENTRY, since=first[-1].seq, limit=4)

    assert [e.seq for e in first] == [1, 2, 3, 4]
    assert [e.seq for e in second] == [5, 6, 7, 8]


def test_tail_is_the_most_recent_event(log):
    log.append(ENTRY, "started", "ingest")
    log.append(ENTRY, "progress", "render", "clip_02/audio_mix")

    assert log.tail(ENTRY).detail == "clip_02/audio_mix"
    assert log.last_seq(ENTRY) == 2


# -- the writer is killable ---------------------------------------------


def test_a_truncated_final_line_is_skipped_not_fatal(log):
    """A hard kill mid-write leaves half a JSON object. Read the rest anyway."""
    log.append(ENTRY, "started", "ingest")
    log.append(ENTRY, "progress", "transcribe", "done")
    with log.path(ENTRY).open("a", encoding="utf-8") as fh:
        fh.write('{"seq":3,"ts":"2026-09-17T14:03:11')

    events = reopen(log).read(ENTRY)

    assert [e.seq for e in events] == [1, 2]


def test_a_new_writer_resumes_after_a_truncated_line(log):
    log.append(ENTRY, "started", "ingest")
    with log.path(ENTRY).open("a", encoding="utf-8") as fh:
        fh.write('{"seq":2,"ts":"nope')

    fresh = reopen(log)
    event = fresh.append(ENTRY, "progress", "transcribe")

    # seq 2 is reused, because nothing readable ever claimed it.
    assert event.seq == 2
    assert [e.seq for e in fresh.read(ENTRY)] == [1, 2]


def test_blank_lines_are_ignored(log):
    log.append(ENTRY, "started", "ingest")
    with log.path(ENTRY).open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    log.append(ENTRY, "progress", "transcribe")

    assert [e.seq for e in reopen(log).read(ENTRY)] == [1, 2]


def test_delete_removes_the_file_and_the_cached_sequence(log):
    log.append(ENTRY, "started", "ingest")
    log.delete(ENTRY)

    assert not log.exists(ENTRY)
    assert log.append(ENTRY, "started", "ingest").seq == 1


# -- the pipeline contract ----------------------------------------------


def test_progress_callback_is_the_shape_pipeline_already_calls(log):
    """``Progress = Callable[[str, str], None]`` - nothing in pipeline.py changes."""
    note = log.progress_callback(ENTRY)

    note("ingest", "downloading")
    note("transcribe")

    events = reopen(log).read(ENTRY)

    assert [(e.kind, e.stage, e.detail) for e in events] == [
        ("progress", "ingest", "downloading"),
        ("progress", "transcribe", ""),
    ]
    assert note("moments", "scored 20 candidates") is None


def test_progress_callback_swallows_disk_errors(log, monkeypatch):
    """A full disk in the event log must not take down a render."""

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(EventLog, "append", boom)

    log.progress_callback(ENTRY)("render", "clip_01")  # does not raise


# -- the wire form ------------------------------------------------------


def test_the_file_is_one_json_object_per_line(log):
    log.append(ENTRY, "started", "ingest", "downloading")

    line = log.path(ENTRY).read_text(encoding="utf-8").splitlines()[0]
    data = json.loads(line)

    assert set(data) == {"seq", "ts", "kind", "stage", "detail"}
    assert data["kind"] == "started"


def test_event_json_survives_json_dumps():
    event = Event(seq=1, ts=datetime(2026, 9, 17, 14, 3, tzinfo=UTC), kind="started")

    assert json.dumps(event.to_json())
    assert Event.from_json(event.to_json()) == event


def test_max_events_is_a_sane_default():
    assert MAX_EVENTS >= 500
