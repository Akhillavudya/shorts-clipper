"""Turning presenter view models into JSON (WEB_UI_PLAN.md section 4).

``review/presenter.py`` builds frozen dataclasses whose most useful parts are
``@property`` accessors - ``verdict``, ``span_label``, ``percent``, ``pending``.
``dataclasses.asdict`` drops every one of them and leaves raw ``datetime``
objects that ``jsonify`` then refuses. This module is the layer that exists
because of that, and it keeps four rules:

1. **Every field is written out by hand, never by reflection.** The wire format
   is a contract the SPA's ``lib/types.ts`` is typed against, so it is worth
   being able to read it in one place and worth a test asserting the exact key
   set. Reflection would make adding a field to a presenter dataclass silently
   change the API.
2. **Every datetime goes through :func:`iso`.** One function, so "why is this
   one a string and that one an object" cannot happen.
3. **This is where relative paths become URLs.** ``presenter.py`` states that it
   builds paths and never URLs, so routing knowledge lives here instead.
4. **Every surfaced error goes through :func:`~clipper.server.text.clean_error`.**

Flask-free on purpose: these are pure functions over dataclasses, so
``test_server_schemas.py`` asserts the wire format without an app, a request
context or a client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import quote

from clipper.review import presenter
from clipper.server.text import clean_error


def iso(value: datetime | None) -> str | None:
    """A datetime as ISO-8601, or ``None``. The only datetime path out of here.

    State-file timestamps are timezone-aware UTC (``state.utcnow``), so this
    emits an offset and the browser's ``new Date()`` reads it correctly. A
    naive datetime is passed through as-is rather than being guessed at - a
    wrong timezone is worse than an ambiguous one.
    """
    return value.isoformat() if value is not None else None


def media_url(job_id: str, relpath: str) -> str:
    """``clips/clip_01/final.mp4`` -> ``/media/<job_id>/clips/clip_01/final.mp4``.

    ``quote`` with the default ``safe="/"`` escapes each segment and leaves the
    separators, which is exactly right for the ``<path:relpath>`` converter.
    The presenter's paths are always forward-slashed, so no ``Path`` ever
    reaches here to contribute a backslash.
    """
    return f"/media/{quote(job_id)}/{quote(relpath)}"


def axis_json(axis: presenter.Axis) -> dict[str, Any]:
    return {
        "name": axis.name,
        "label": axis.label,
        "score": axis.score,
        "weight": axis.weight,
        "percent": axis.percent,
    }


def preview_json(preview: presenter.Preview | None, *, job_id: str) -> dict[str, Any] | None:
    """What to play, where to get it, and which span of it matters."""
    if preview is None:
        return None
    return {
        "path": preview.path,
        "kind": preview.kind,
        "start": preview.start,
        "end": preview.end,
        "label": preview.label,
        "is_source": preview.is_source,
        "url": media_url(job_id, preview.path),
    }


def counts_json(counts: presenter.Counts) -> dict[str, Any]:
    return {
        "total": counts.total,
        "approved": counts.approved,
        "rejected": counts.rejected,
        "pending": counts.pending,
        "done": counts.done,
    }


def clip_card_json(clip: presenter.ClipCard, *, job_id: str) -> dict[str, Any]:
    return {
        "clip_id": clip.clip_id,
        "index": clip.index,
        "score": clip.score,
        "score_label": clip.score_label,
        "type": clip.type,
        "reason": clip.reason,
        "suggested_title": clip.suggested_title,
        "hook_rewrite": clip.hook_rewrite,
        "emphasis_words": list(clip.emphasis_words),
        "start": clip.start,
        "end": clip.end,
        "span_label": clip.span_label,
        "duration_s": clip.duration_s,
        "duration_label": clip.duration_label,
        "rubric": [axis_json(a) for a in clip.rubric],
        "transcript": clip.transcript,
        "approved": clip.approved,
        "verdict": clip.verdict,
        "review_note": clip.review_note,
        "reviewed_at": iso(clip.reviewed_at),
        "preview": preview_json(clip.preview, job_id=job_id),
        "rendered": clip.rendered,
        "render_note": clip.render_note,
        "youtube_video_id": clip.youtube_video_id,
        "source_link": clip.source_link,
    }


def job_summary_json(summary: presenter.JobSummary) -> dict[str, Any]:
    return {
        "video_id": summary.video_id,
        "title": summary.title,
        "channel_title": summary.channel_title,
        "rights_status": summary.rights_status,
        "rights_override": summary.rights_override,
        "exportable": summary.exportable,
        "stage": summary.stage,
        "stage_failed": summary.stage_failed,
        "counts": counts_json(summary.counts),
        "updated_at": iso(summary.updated_at),
    }


def job_view_json(
    view: presenter.JobView,
    *,
    writable: bool = True,
    entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One job's page.

    ``writable`` is the section 3 single-writer discipline made visible: while
    a worker owns this job's ``state.json``, the SPA disables the verdict
    buttons rather than letting the user discover the 409. ``entry`` is the
    queue row that owns it, already serialised, or ``None`` when nothing does.
    Phase 0 has no queue, so every job is writable and no entry is attached.
    """
    return {
        "video_id": view.video_id,
        "title": view.title,
        "channel_title": view.channel_title,
        "source_url": view.source_url,
        "duration_s": view.duration_s,
        "duration_label": view.duration_label,
        "rights_status": view.rights_status,
        "rights_note": view.rights_note,
        "rights_override": view.rights_override,
        "exportable": view.exportable,
        "counts": counts_json(view.counts),
        "clips": [clip_card_json(c, job_id=view.video_id) for c in view.clips],
        "warnings": list(view.warnings),
        "writable": writable,
        "entry": entry,
    }


def stage_json(name: str, stage: Any) -> dict[str, Any]:
    """One :class:`~clipper.state.StageState`, for the job page's stage strip.

    ``error`` is cleaned here rather than at the raise site: the state file
    keeps whatever ffmpeg actually said, and only the surfaced copy is trimmed.
    """
    return {
        "name": name,
        "status": str(stage.status),
        "started_at": iso(stage.started_at),
        "finished_at": iso(stage.finished_at),
        "duration_s": stage.duration_s,
        "attempts": stage.attempts,
        "error": clean_error(stage.error),
        "detail": dict(stage.detail),
    }


def run_options_json(options: Any) -> dict[str, Any]:
    """The flags a queue entry was submitted with (``queue.RunOptions``)."""
    return {
        "rights": options.rights,
        "music": options.music,
        "no_music": options.no_music,
        "centre_crop": options.centre_crop,
        "force": options.force,
        "render_only": options.render_only,
        "approved_only": options.approved_only,
        "clip_index": options.clip_index,
        "allow_red": options.allow_red,
        "override_reason": options.override_reason,
    }


def queue_entry_json(entry: Any) -> dict[str, Any]:
    """One :class:`~clipper.server.queue.QueueEntry`.

    ``label`` is emitted because the row has to say *something* for the minutes
    between pasting a URL and ``ingest`` producing a title, and the fallback
    ("show the URL") is a display decision that belongs on this side of the
    wire rather than being reimplemented in TypeScript.

    ``pid`` is included deliberately: when a render has been silent for two
    minutes, "process 24180 is alive" is the one thing that distinguishes
    working from hung, and it is what you type into Task Manager if it is not.
    """
    return {
        "entry_id": entry.entry_id,
        "source": entry.source,
        "status": str(entry.status),
        "label": entry.label,
        "options": run_options_json(entry.options),
        "video_id": entry.video_id,
        "rights_status": entry.rights_status,
        "pid": entry.pid,
        "submitted_at": iso(entry.submitted_at),
        "started_at": iso(entry.started_at),
        "finished_at": iso(entry.finished_at),
        "last_alive": iso(entry.last_alive),
        "error": clean_error(entry.error),
        "exit_code": entry.exit_code,
    }


def event_json(event: Any) -> dict[str, Any]:
    """One :class:`~clipper.server.events.Event`.

    The event already knows its own wire form - it is written to JSONL in
    exactly this shape - so this delegates rather than restating it. Keeping
    the function here anyway means every route still serialises through
    ``schemas``, and the day the stored shape and the wire shape need to differ
    there is already a seam.
    """
    return event.to_json()


def check_json(check: Any) -> dict[str, Any]:
    """One :class:`~clipper.server.doctor.Check`."""
    return {
        "name": check.name,
        "status": check.status,
        "detail": check.detail,
        "hint": check.hint,
    }


__all__ = [
    "axis_json",
    "check_json",
    "clip_card_json",
    "counts_json",
    "event_json",
    "iso",
    "job_summary_json",
    "job_view_json",
    "media_url",
    "preview_json",
    "queue_entry_json",
    "run_options_json",
    "stage_json",
]
