"""The worker child (WEB_UI_PLAN.md sections 1, 5 and 8).

``pipeline`` is faked wholesale. That is not a shortcut: the worker's whole job
is to decide *which* stages run, in what order, and what exit code comes out
the other end, and every one of those decisions is visible without running a
single real stage. Running them would need a network, ffmpeg and an LLM key,
which is exactly what ``tests/`` does not have.

No subprocess is started here either - ``run_entry`` is called in-process. The
process boundary is ``runner.py``'s problem and is tested there.
"""

from __future__ import annotations

import json

import pytest

from clipper.server.events import EventLog
from clipper.server.queue import RunOptions
from clipper.server.worker import (
    EXIT_BLOCKED,
    EXIT_CANCELLED,
    EXIT_FAILED,
    EXIT_OK,
    main,
    run_entry,
)
from clipper.state import JobStore, RightsBlocked, RightsStatus

ENTRY = "e" * 32


@pytest.fixture
def jobs_root(tmp_path):
    return tmp_path / "jobs"


@pytest.fixture
def log(jobs_root) -> EventLog:
    return EventLog(jobs_root)


@pytest.fixture
def cancel_path(jobs_root):
    return jobs_root / "_queue" / f"{ENTRY}.cancel"


@pytest.fixture
def calls() -> list[tuple[str, tuple, dict]]:
    return []


@pytest.fixture
def fake_pipeline(monkeypatch, jobs_root, calls):
    """Replace every stage with a recorder. Returns a knob-bag for the tests.

    The fakes fire ``on_progress`` exactly as the real stages do, because the
    event log adapting to ``Progress = Callable[[str, str], None]`` is the
    contract that lets ``pipeline.py`` stay untouched - and a fake that did not
    call it would not be testing that.
    """
    from clipper import pipeline

    store = JobStore(jobs_root)

    class Knobs:
        video_id = "vid123"
        rights = RightsStatus.green
        rights_note = "CC BY on the channel."
        ingest_hook = None
        moments_hook = None
        render_hook = None
        rendered = ["clip_01", "clip_02"]

    knobs = Knobs()

    def record(name):
        def fn(*args, on_progress=None, **kwargs):
            calls.append((name, args[1:], kwargs))
            if on_progress is not None:
                on_progress(name, f"{name} did something")
            hook = getattr(knobs, f"{name}_hook", None)
            if hook is not None:
                hook()
            return None

        return fn

    def fake_ingest(ctx, source, *, rights=None, force=False, on_progress=None, **kwargs):
        calls.append(("ingest", (source,), {"rights": rights, "force": force}))
        if on_progress is not None:
            on_progress("download", source)
        state = store.load_or_create(knobs.video_id, source_url=source, rights_status=knobs.rights)
        state.rights_status = knobs.rights
        state.rights_note = knobs.rights_note
        store.save(state)
        if knobs.ingest_hook is not None:
            knobs.ingest_hook()
        return state

    def fake_render(ctx, video_id, **kwargs):
        calls.append(("render", (video_id,), kwargs))
        progress = kwargs.get("on_progress")
        if progress is not None:
            progress("clip_01/reframe", "tracked crop to 1080x1920")
        if knobs.render_hook is not None:
            knobs.render_hook()
        return list(knobs.rendered)

    monkeypatch.setattr(pipeline.Context, "build", classmethod(lambda cls, **kw: object()))
    monkeypatch.setattr(pipeline, "ingest", fake_ingest)
    monkeypatch.setattr(pipeline, "transcribe", record("transcribe"))
    monkeypatch.setattr(pipeline, "moments", record("moments"))
    monkeypatch.setattr(pipeline, "render", fake_render)
    return knobs


def run(log, cancel_path, jobs_root, tmp_path, **option_overrides) -> int:
    return run_entry(
        ENTRY,
        source=option_overrides.pop("source", "https://youtu.be/vid123"),
        options=RunOptions(**option_overrides),
        config_path=tmp_path / "config" / "settings.yaml",
        jobs_root=jobs_root,
        log=log,
        cancel_path=cancel_path,
    )


def kinds(log: EventLog) -> list[str]:
    return [e.kind for e in log.read(ENTRY, limit=2000)]


def stages(log: EventLog) -> list[str]:
    return [e.stage for e in log.read(ENTRY, limit=2000) if e.kind == "stage"]


def names(calls) -> list[str]:
    return [name for name, _, _ in calls]


# -- the happy path ---------------------------------------------------------


def test_a_green_job_runs_every_stage_and_exits_zero(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_OK
    assert names(calls) == ["ingest", "transcribe", "moments", "render"]
    assert stages(log) == ["ingest", "transcribe", "moments", "render"]


def test_render_retry_restores_deleted_source_without_rescoring(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    store = JobStore(jobs_root)
    state = store.create(
        "vid123", source_url="https://youtu.be/vid123", rights_status=RightsStatus.green
    )
    state.stage("ingest").detail["source_deleted"] = True
    store.save(state)
    assert run(log, cancel_path, jobs_root, tmp_path, source="vid123", render_only=True) == EXIT_OK
    assert names(calls) == ["ingest", "render"]
    assert calls[0][1] == ("https://youtu.be/vid123",)
    assert "source_deleted" not in store.load("vid123").stage("ingest").detail


def test_the_video_id_travels_back_as_a_resolved_event(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path
):
    """The supervisor has no other way to learn it - the processes share files.

    At submit time all that exists is a URL; the id arrives minutes later, out
    of ``ingest``'s metadata call.
    """
    run(log, cancel_path, jobs_root, tmp_path)
    resolved = [e for e in log.read(ENTRY, limit=2000) if e.kind == "resolved"]
    assert [e.detail for e in resolved] == ["vid123"]


def test_the_rights_status_and_its_note_are_reported_even_when_green(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path
):
    """Green never interrupts a job, but the row still says why it is green."""
    run(log, cancel_path, jobs_root, tmp_path)
    rights = next(e for e in log.read(ENTRY, limit=2000) if e.kind == "rights")
    assert rights.stage == "green"
    assert rights.detail == "CC BY on the channel."


def test_progress_from_inside_a_stage_lands_in_the_log(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path
):
    """``pipeline.py`` is untouched: its existing callback shape is the seam."""
    run(log, cancel_path, jobs_root, tmp_path)
    progress = {(e.stage, e.detail) for e in log.read(ENTRY, limit=2000) if e.kind == "progress"}
    assert ("clip_01/reframe", "tracked crop to 1080x1920") in progress
    assert ("download", "https://youtu.be/vid123") in progress


def test_it_finishes_with_a_count_of_what_it_rendered(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path
):
    run(log, cancel_path, jobs_root, tmp_path)
    finished = next(e for e in log.read(ENTRY, limit=2000) if e.kind == "finished")
    assert finished.detail == "2 clips rendered"
    assert finished.stage == "vid123"


# -- the rights gate, section 5 ---------------------------------------------


def test_an_unknown_source_stops_before_transcribe_and_exits_three(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    """The cheap failure the whole design exists for.

    No LLM quota, no CPU, and - crucially - the failure arrives a minute or two
    after pasting rather than forty minutes later at ``render``.
    """
    fake_pipeline.rights = RightsStatus.unknown
    fake_pipeline.rights_note = "no watchlist entry, and the licence is not CC BY."

    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_BLOCKED
    assert names(calls) == ["ingest"]

    blocked = next(e for e in log.read(ENTRY, limit=2000) if e.kind == "blocked")
    assert blocked.stage == "unknown"
    assert blocked.detail == "no watchlist entry, and the licence is not CC BY."
    assert "finished" not in kinds(log)


def test_a_red_source_is_blocked_exactly_like_an_unknown_one(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path
):
    fake_pipeline.rights = RightsStatus.red
    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_BLOCKED


def test_amber_is_exportable_and_runs_through(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    fake_pipeline.rights = RightsStatus.amber
    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_OK
    assert "render" in names(calls)


def test_allow_red_carries_a_blocked_source_past_the_gate(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    """The override is real, and it is only ever set from the job page.

    ``api.parse_options`` refuses it at submit time; this is the path a
    deliberate "render anyway", with a reason typed, takes afterwards.
    """
    fake_pipeline.rights = RightsStatus.red
    code = run(
        log,
        cancel_path,
        jobs_root,
        tmp_path,
        allow_red=True,
        override_reason="written permission by email, 2026-09-14",
    )
    assert code == EXIT_OK
    assert names(calls) == ["ingest", "transcribe", "moments", "render"]

    _, _, kwargs = next(c for c in calls if c[0] == "render")
    assert kwargs["allow_red"] is True
    assert kwargs["override_reason"] == "written permission by email, 2026-09-14"


def test_a_rights_block_raised_by_render_is_still_exit_three(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path
):
    """``render`` gates per clip, so the gate can fire after the job-level one."""

    def boom():
        raise RightsBlocked("clip_01: rights_status is red")

    fake_pipeline.render_hook = boom
    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_BLOCKED
    assert "blocked" in kinds(log)
    assert "error" not in kinds(log)


# -- cancellation -----------------------------------------------------------


def test_a_sentinel_written_before_the_start_stops_everything(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    cancel_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_path.write_text("now", encoding="utf-8")

    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_CANCELLED
    assert calls == []
    assert "cancelled" in kinds(log)


def test_a_sentinel_written_mid_ingest_is_caught_at_the_next_boundary(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    """Graceful cancellation is checked between stages and only there.

    Inside a three-minute encode there is nowhere to check, which is exactly
    why the supervisor also kills the tree.
    """

    def cancel_now():
        cancel_path.parent.mkdir(parents=True, exist_ok=True)
        cancel_path.write_text("now", encoding="utf-8")

    fake_pipeline.ingest_hook = cancel_now

    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_CANCELLED
    assert names(calls) == ["ingest"]
    assert "transcribe" not in stages(log)


def test_a_cancel_during_moments_still_skips_the_expensive_stage(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    def cancel_now():
        cancel_path.parent.mkdir(parents=True, exist_ok=True)
        cancel_path.write_text("now", encoding="utf-8")

    fake_pipeline.moments_hook = cancel_now

    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_CANCELLED
    assert names(calls) == ["ingest", "transcribe", "moments"]


# -- failure ----------------------------------------------------------------


def test_a_stage_exception_becomes_exit_one_and_a_stripped_error_event(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path
):
    def boom():
        raise RuntimeError("\x1b[31mffmpeg: Invalid data found\x1b[0m")

    fake_pipeline.moments_hook = boom

    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_FAILED
    error = next(e for e in log.read(ENTRY, limit=2000) if e.kind == "error")
    assert error.detail == "RuntimeError: ffmpeg: Invalid data found"
    assert "\x1b" not in error.detail


def test_run_entry_never_raises(fake_pipeline, log, cancel_path, jobs_root, tmp_path):
    """The process boundary is here: everything has to become an exit code.

    An exception escaping ``run_entry`` would exit with Python's own ``1``,
    which is indistinguishable from a stage failing - and would leave no event
    behind to say which.
    """

    def boom():
        raise KeyboardInterrupt

    fake_pipeline.moments_hook = boom
    assert run(log, cancel_path, jobs_root, tmp_path) == EXIT_CANCELLED


# -- render-only ------------------------------------------------------------


def test_render_only_skips_the_first_three_stages(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    """Phase 5's "render from the job page": the job already exists.

    ``source`` is the video id in this mode, so nothing is resolved and no
    network call happens.
    """
    code = run(
        log,
        cancel_path,
        jobs_root,
        tmp_path,
        source="vid123",
        render_only=True,
        approved_only=True,
        clip_index=2,
    )
    assert code == EXIT_OK
    assert names(calls) == ["render"]

    _, args, kwargs = calls[0]
    assert args == ("vid123",)
    assert kwargs["approved_only"] is True
    assert kwargs["clip_index"] == 2

    resolved = next(e for e in log.read(ENTRY, limit=2000) if e.kind == "resolved")
    assert resolved.detail == "vid123"


# -- the command line -------------------------------------------------------


def test_main_round_trips_options_through_json(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    """Options cross the process boundary as one JSON argument, not as flags.

    A flag per option would need translating in two places and would drift;
    ``RunOptions`` validates on the way in and on the way out.
    """
    code = main(
        [
            "--entry",
            ENTRY,
            "--source",
            "https://youtu.be/vid123",
            "--config",
            str(tmp_path / "config" / "settings.yaml"),
            "--jobs-root",
            str(jobs_root),
            "--options",
            json.dumps({"centre_crop": True, "no_music": True}),
        ]
    )
    assert code == EXIT_OK

    _, _, kwargs = next(c for c in calls if c[0] == "render")
    assert kwargs["centre_crop"] is True
    assert kwargs["no_music"] is True


def test_main_finds_the_cancel_sentinel_the_supervisor_would_write(
    fake_pipeline, jobs_root, tmp_path, calls
):
    """Both sides derive the path from ``QueueStore``, so they cannot disagree."""
    from clipper.server.queue import QueueStore

    path = QueueStore(jobs_root).cancel_path(ENTRY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("now", encoding="utf-8")

    code = main(
        [
            "--entry",
            ENTRY,
            "--source",
            "https://youtu.be/vid123",
            "--config",
            str(tmp_path / "config" / "settings.yaml"),
            "--jobs-root",
            str(jobs_root),
        ]
    )
    assert code == EXIT_CANCELLED
    assert calls == []


def test_retry_after_classification_skips_ingest_resolution(
    fake_pipeline, log, cancel_path, jobs_root, tmp_path, calls
):
    from clipper.server.queue import QueueStore
    from clipper.state import StageStatus

    store = JobStore(jobs_root)
    state = store.create("vid123")
    state.rights_status = RightsStatus.green
    state.rights_override = True
    state.rights_override_reason = "Permission email saved"
    state.stage("ingest").status = StageStatus.done
    store.save(state)
    queue = QueueStore(jobs_root)
    entry = queue.enqueue("https://youtu.be/vid123")
    queue.update(entry.entry_id, video_id="vid123")
    code = run_entry(
        entry.entry_id,
        source=entry.source,
        options=RunOptions(),
        config_path=tmp_path / "settings.yaml",
        jobs_root=jobs_root,
        log=log,
        cancel_path=cancel_path,
    )
    assert code == EXIT_OK
    assert names(calls) == ["transcribe", "moments", "render"]
    assert store.load("vid123").rights_override_reason == "Permission email saved"
