"""The wire contract (WEB_UI_PLAN.md sections 4 and 8).

No app, no client, no request context: these are pure functions over presenter
dataclasses, so the format they produce is asserted directly.

Three classes of bug are worth a test here, and they are the three that would
otherwise be found in a browser:

- **A ``@property`` that never made it onto the wire.** ``dataclasses.asdict``
  drops every one of them, so ``verdict``, ``span_label`` and ``percent`` are
  written out by hand - and hand-written things get forgotten. The exact-keys
  tests pin the format, and the reflection test catches a *new* field added to
  a presenter dataclass whose serializer was not updated.
- **A raw ``datetime`` reaching ``jsonify``.** One ``json.dumps`` per payload
  covers the whole class.
- **A backslash or an unescaped space in a media URL.** The presenter's paths
  are relative and forward-slashed; this is where they become routable.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

import pytest

from clipper.review import presenter
from clipper.server import schemas


@pytest.fixture
def axis() -> presenter.Axis:
    return presenter.Axis(name="hook", label="hook", score=8.0, weight=0.3)


@pytest.fixture
def preview() -> presenter.Preview:
    return presenter.Preview(path="clips/clip_01/final.mp4", kind="final", start=0.0, end=39.5)


@pytest.fixture
def clip(axis, preview) -> presenter.ClipCard:
    return presenter.ClipCard(
        clip_id="clip_01",
        index=1,
        score=7.1,
        type="insight",
        reason="States a counterintuitive point and resolves it.",
        suggested_title="the boring fund wins",
        hook_rewrite="Nobody tells you this.",
        emphasis_words=["boring"],
        start=12.5,
        end=52.0,
        duration_s=39.5,
        rubric=[axis],
        transcript="the thing nobody tells you",
        approved=True,
        review_note="good hook",
        reviewed_at=datetime(2026, 9, 17, 14, 3, 11, tzinfo=UTC),
        preview=preview,
        rendered=True,
        render_note="tracked crop",
        youtube_video_id="abc123",
        source_link="https://youtu.be/vid123?t=12s",
    )


@pytest.fixture
def view(clip) -> presenter.JobView:
    return presenter.JobView(
        video_id="vid123",
        title="An Episode",
        channel_title="A Channel",
        source_url="https://www.youtube.com/watch?v=vid123",
        duration_s=3600.0,
        rights_status="green",
        rights_note="CC BY.",
        rights_override=False,
        exportable=True,
        counts=presenter.Counts(total=2, approved=1, rejected=0),
        clips=[clip],
        warnings=["No transcript on disk."],
    )


@pytest.fixture
def summary() -> presenter.JobSummary:
    return presenter.JobSummary(
        video_id="vid123",
        title="An Episode",
        channel_title="A Channel",
        rights_status="green",
        rights_override=False,
        exportable=True,
        stage="render",
        stage_failed=False,
        counts=presenter.Counts(total=2, approved=1, rejected=0),
        updated_at=datetime(2026, 9, 17, 14, 3, 11, tzinfo=UTC),
    )


# -- iso --------------------------------------------------------------------


def test_iso_emits_an_offset_and_passes_none_through():
    assert schemas.iso(None) is None
    assert schemas.iso(datetime(2026, 9, 17, 14, 3, 11, tzinfo=UTC)) == "2026-09-17T14:03:11+00:00"


# -- the exact key sets -----------------------------------------------------


def test_axis_json_carries_the_bar_width(axis):
    assert schemas.axis_json(axis) == {
        "name": "hook",
        "label": "hook",
        "score": 8.0,
        "weight": 0.3,
        "percent": 80.0,
    }


def test_counts_json_carries_the_derived_pending_and_done():
    counts = presenter.Counts(total=3, approved=1, rejected=1)
    assert schemas.counts_json(counts) == {
        "total": 3,
        "approved": 1,
        "rejected": 1,
        "pending": 1,
        "done": False,
    }


def test_preview_json_is_none_when_there_is_nothing_to_play():
    assert schemas.preview_json(None, job_id="vid123") is None


def test_preview_json_builds_the_media_url(preview):
    payload = schemas.preview_json(preview, job_id="vid123")
    assert payload["url"] == "/media/vid123/clips/clip_01/final.mp4"
    assert payload["label"] == "final render"
    assert payload["is_source"] is False


def test_preview_json_marks_the_source_fallback():
    source = presenter.Preview(path="source.mp4", kind="source", start=12.5, end=52.0)
    payload = schemas.preview_json(source, job_id="vid123")
    assert payload["is_source"] is True
    assert payload["start"] == 12.5 and payload["end"] == 52.0


def test_clip_card_json_key_set_is_the_contract(clip):
    assert set(schemas.clip_card_json(clip, job_id="vid123")) == {
        "clip_id",
        "index",
        "score",
        "score_label",
        "type",
        "reason",
        "suggested_title",
        "hook_rewrite",
        "emphasis_words",
        "start",
        "end",
        "span_label",
        "duration_s",
        "duration_label",
        "rubric",
        "transcript",
        "approved",
        "verdict",
        "review_note",
        "reviewed_at",
        "preview",
        "rendered",
        "render_note",
        "youtube_video_id",
        "source_link",
    }


def test_clip_card_json_carries_the_formatted_properties(clip):
    payload = schemas.clip_card_json(clip, job_id="vid123")
    assert payload["verdict"] == "approved"
    assert payload["score_label"] == "7.10"
    assert payload["span_label"] == "0:12 - 0:52"
    assert payload["duration_label"] == "40s"
    assert payload["reviewed_at"] == "2026-09-17T14:03:11+00:00"


def test_job_view_json_key_set_is_the_contract(view):
    assert set(schemas.job_view_json(view)) == {
        "video_id",
        "title",
        "channel_title",
        "source_url",
        "duration_s",
        "duration_label",
        "rights_status",
        "rights_note",
        "rights_override",
        "exportable",
        "counts",
        "clips",
        "warnings",
        "writable",
        "entry",
    }


def test_job_view_json_defaults_to_writable_with_no_owning_entry(view):
    payload = schemas.job_view_json(view)
    assert payload["writable"] is True
    assert payload["entry"] is None


def test_job_view_json_carries_the_single_writer_flag(view):
    payload = schemas.job_view_json(view, writable=False, entry={"entry_id": "abc"})
    assert payload["writable"] is False
    assert payload["entry"] == {"entry_id": "abc"}


def test_job_summary_json_key_set_is_the_contract(summary):
    assert set(schemas.job_summary_json(summary)) == {
        "video_id",
        "title",
        "channel_title",
        "rights_status",
        "rights_override",
        "exportable",
        "stage",
        "stage_failed",
        "counts",
        "updated_at",
    }


# -- the datetime class of bugs, in one line each ---------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda f: schemas.clip_card_json(f["clip"], job_id="vid123"),
        lambda f: schemas.job_view_json(f["view"]),
        lambda f: schemas.job_summary_json(f["summary"]),
    ],
    ids=["clip_card", "job_view", "job_summary"],
)
def test_every_payload_is_json_serialisable(build, clip, view, summary):
    json.dumps(build({"clip": clip, "view": view, "summary": summary}))


# -- the forgotten-field regression -----------------------------------------


@pytest.mark.parametrize(
    "dataclass_, build",
    [
        (presenter.Axis, lambda f: schemas.axis_json(f["axis"])),
        (presenter.Preview, lambda f: schemas.preview_json(f["preview"], job_id="vid123")),
        (presenter.Counts, lambda f: schemas.counts_json(f["view"].counts)),
        (presenter.ClipCard, lambda f: schemas.clip_card_json(f["clip"], job_id="vid123")),
        (presenter.JobSummary, lambda f: schemas.job_summary_json(f["summary"])),
        (presenter.JobView, lambda f: schemas.job_view_json(f["view"])),
    ],
    ids=["Axis", "Preview", "Counts", "ClipCard", "JobSummary", "JobView"],
)
def test_every_dataclass_field_reaches_the_wire(
    dataclass_, build, axis, preview, clip, view, summary
):
    """Adding a field to a presenter dataclass must not silently skip the API.

    The serializers are hand-written on purpose (the wire format is a contract,
    not a reflection of the internals), and the cost of that choice is exactly
    this failure mode. This is the test that pays it back.
    """
    payload = build(
        {"axis": axis, "preview": preview, "clip": clip, "view": view, "summary": summary}
    )
    missing = [f.name for f in dataclasses.fields(dataclass_) if f.name not in payload]
    assert not missing, f"{dataclass_.__name__} fields absent from its JSON: {missing}"


# -- URLs -------------------------------------------------------------------


def test_media_url_escapes_each_segment_and_keeps_the_separators():
    url = schemas.media_url("vid 123", "clips/clip_01/final cut.mp4")
    assert url == "/media/vid%20123/clips/clip_01/final%20cut.mp4"


def test_media_url_never_emits_a_backslash():
    """``str(Path)`` gives backslashes on Windows; a URL must never carry one."""
    assert "\\" not in schemas.media_url("vid123", "clips/clip_01/final.mp4")


# -- errors -----------------------------------------------------------------


def test_stage_json_strips_ansi_off_a_surfaced_error():
    from clipper.state import StageState, StageStatus

    stage = StageState(
        status=StageStatus.failed,
        error="\x1b[31mffmpeg: Invalid data found\x1b[0m",
        attempts=2,
        detail={"clips": 7},
    )
    payload = schemas.stage_json("render", stage)
    assert payload["error"] == "ffmpeg: Invalid data found"
    assert payload["status"] == "failed"
    assert payload["attempts"] == 2
    assert payload["detail"] == {"clips": 7}
    json.dumps(payload)


def test_stage_json_reports_a_stage_that_never_ran():
    from clipper.state import StageState

    payload = schemas.stage_json("ingest", StageState())
    assert payload["status"] == "pending"
    assert payload["started_at"] is None
    assert payload["error"] is None
