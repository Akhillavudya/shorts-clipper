"""The JSON API (WEB_UI_PLAN.md section 4).

A Flask blueprint mounted at ``/api`` on the same app that serves the review
pages, so there is one process, one port and one job store. It is additive on
purpose: the Jinja routes, the ``/media/`` route and all 324 lines of
``tests/test_review.py`` keep working while the SPA is built beside them.

The read model is ``review/presenter.py`` - not a second one. This layer adds
routing and serialisation and nothing else, which is why every route here is a
few lines: the interesting code is in the presenter (what a reviewer sees) and
in ``schemas.py`` (what the wire looks like).

Phase 2 attaches the supervisor. ``cancel`` and ``retry`` are routes rather
than queue methods because both are lifecycle decisions about a *process*: a
queued entry is cancelled by editing the file, a running one by writing a
sentinel and then killing a tree, and only ``runner.py`` knows which it is
looking at. Without a runner attached the queue routes still work and the page
still reports ``runner.alive: false`` rather than looking idle.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    request,
    send_file,
    stream_with_context,
)
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from clipper.review import presenter
from clipper.rights.registry import RightsError
from clipper.rights.watchlist_edit import append_channel
from clipper.server import disk, schemas
from clipper.server import doctor as doctor_module
from clipper.server.events import DEFAULT_LIMIT, MAX_EVENTS, EventLog
from clipper.server.operations import log_tail
from clipper.server.queue import ACTIVE_STATUSES, QueueBusy, QueueStore, RunOptions
from clipper.state import CLIP_STAGES, JOB_STAGES, JobStore, RightsStatus, StageStatus, utcnow

#: How many sources one paste may carry. Not a security boundary - there is no
#: attacker on loopback - but a fat-fingered paste of a whole subscriptions
#: page should be a 400, not fifty multi-gigabyte downloads.
MAX_SOURCES_PER_SUBMIT = 25


def parse_sources(payload: dict[str, Any]) -> list[str]:
    """``{"sources": [...]}`` or ``{"source": "..."}`` -> a list of sources.

    An array because "paste several links" is one user action and should be
    one request; the singular form is accepted because a one-link submit
    written by hand, or by ``curl``, should not have to know that.
    """
    raw = payload.get("sources", payload.get("source"))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        abort(400, description="sources must be a list of URLs or file paths")

    sources = []
    for item in raw:
        if not isinstance(item, str):
            abort(400, description=f"source {item!r} is not a string")
        cleaned = item.strip()
        if cleaned:
            sources.append(cleaned)

    if not sources:
        abort(400, description="no sources given")
    if len(sources) > MAX_SOURCES_PER_SUBMIT:
        abort(400, description=f"at most {MAX_SOURCES_PER_SUBMIT} sources per submit")
    return sources


def parse_options(payload: dict[str, Any]) -> RunOptions:
    """The submit form's options, validated.

    ``allow_red`` is refused here rather than ignored, and that refusal is the
    point of WEB_UI_PLAN.md section 5: you cannot meaningfully pre-authorise a
    rights override for a channel you have not seen. The override is reachable
    only from the job page, after ingest, with the rights note in front of you
    and a reason typed. Silently dropping the flag would leave a caller
    believing it had been honoured.
    """
    raw = payload.get("options") or {}
    if not isinstance(raw, dict):
        abort(400, description="options must be an object")
    if raw.get("allow_red"):
        abort(
            400,
            description=(
                "allow_red cannot be set at submit time - override the rights gate "
                "from the job page once ingest has reported what the source is"
            ),
        )
    if raw.get("render_only") or raw.get("override_reason"):
        abort(400, description="Use the job page for render-only and rights overrides.")
    try:
        return RunOptions.model_validate(raw)
    except ValidationError as exc:
        abort(400, description=_first_error(exc))


def _first_error(exc: ValidationError) -> str:
    """One readable line out of a pydantic error bundle."""
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "options"
        message = error["msg"].removeprefix("Value error, ")
        return f"{location}: {message}"
    return "invalid options"


def free_bytes(path: Path) -> int | None:
    """Free space on the volume holding the jobs root, or ``None``.

    ``None`` rather than zero when it cannot be read: a UI that says "0 bytes
    free" because a drive letter went away is worse than one that says nothing.
    """
    try:
        return shutil.disk_usage(path if path.is_dir() else path.anchor or ".").free
    except OSError:
        return None


def job_dir_or_404(store: JobStore, job_id: str) -> Path:
    """Resolve a job directory, refusing anything outside the jobs root.

    The single implementation of that guard: ``review/app.py``'s media route
    and every ``/api/jobs/<id>`` route go through here, so a path-traversal
    test written against one of them is testing all of them. Both halves
    matter - ``directory.parent != root`` rejects ``..`` and absolute-looking
    ids after resolution, and ``store.exists`` rejects a directory that is not
    actually a job.
    """
    root = store.root.resolve()
    directory = (root / job_id).resolve()
    if directory.parent != root or not store.exists(job_id):
        abort(404, description=f"no job {job_id!r} under {store.root}")
    return directory


def _reset_stages_from(store: JobStore, video_id: str, from_stage: str) -> None:
    """Mark ``from_stage`` and everything after it ``pending``, in order.

    The caller holds the server's queue lock and has checked every active
    entry for this job before reaching here. Reset clip stages too, otherwise
    restarting render would silently reuse the old finished clips.
    """
    state = store.load(video_id)
    index = JOB_STAGES.index(from_stage)
    for name in JOB_STAGES[index:]:
        stage = state.stage(name)
        stage.status = StageStatus.pending
        stage.started_at = None
        stage.finished_at = None
        stage.error = None
    for clip in state.clips:
        for name in CLIP_STAGES:
            stage = clip.stage(name)
            stage.status = StageStatus.pending
            stage.started_at = stage.finished_at = stage.error = None
    store.save(state)


def create_blueprint() -> Blueprint:
    """The ``/api`` blueprint. Reads its store and weights off ``app.config``.

    Built by a factory rather than declared at module level so that two apps in
    one test session (the review app and, later, the SPA app) each get their
    own, and so a blueprint can never outlive the config it was registered
    against.
    """
    bp = Blueprint("api", __name__, url_prefix="/api")

    @bp.errorhandler(HTTPException)
    def _as_json(exc: HTTPException):
        """Every ``abort()`` raised inside this blueprint answers in JSON.

        The app's own handlers cover 400 and 404 because the Jinja pages raise
        those too. This covers the rest - a 503 from a missing runner, a 405
        from the wrong verb - so no API route can ever hand the SPA an HTML
        error page to parse as JSON.
        """
        return jsonify({"error": exc.description}), exc.code

    def _store() -> JobStore:
        return current_app.config["JOB_STORE"]

    def _queue() -> QueueStore:
        """The queue, built from the store's root if the app did not supply one.

        Lazy so that ``create_app`` starting up never creates ``_queue/`` on
        disk: the directory appears on the first submit, not on the first page
        view of a machine that has never queued anything.
        """
        queue = current_app.config.get("JOB_QUEUE")
        if queue is None:
            queue = QueueStore(_store().root)
            current_app.config["JOB_QUEUE"] = queue
        return queue

    def _runner():
        """The supervisor, if one is attached. ``None`` under ``clipper review``.

        Read rather than built: a runner spawns processes, and a blueprint that
        could conjure one into existence on the first request would mean a
        read-only review page could start a 2 GB download by being visited.
        ``clipper ui`` sets it, nothing else does.
        """
        return current_app.config.get("JOB_RUNNER")

    def _require_runner():
        runner = _runner()
        if runner is None:
            abort(
                503,
                description=(
                    "no job runner is attached to this server, so nothing is draining "
                    "the queue. Start it with `clipper ui`."
                ),
            )
        return runner

    def _events() -> EventLog:
        log = current_app.config.get("EVENT_LOG")
        if log is None:
            log = EventLog(_store().root)
            current_app.config["EVENT_LOG"] = log
        return log

    def serialized(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with _queue().lock:
                return fn(*args, **kwargs)

        return wrapped

    def _idle(job_id: str) -> None:
        if any(e.video_id == job_id and e.status in ACTIVE_STATUSES for e in _queue().load()):
            abort(409, description="This job is queued or running. Wait for it to finish.")

    @bp.get("/settings")
    def settings():
        from clipper.config import Settings

        path = current_app.config.get("SETTINGS_PATH", "config/settings.yaml")
        cfg = Settings.load(path)
        # Deliberately allowlisted: arbitrary config and environment values can contain keys.
        return jsonify(
            {
                "config_path": str(Path(path).resolve()),
                "jobs_root": str(_store().root.resolve()),
                "values": {
                    key: cfg.get(key, default)
                    for key, default in (
                        ("server.min_free_gb", 10),
                        ("edit.reframe", "tracked"),
                        ("edit.music_enabled", False),
                    )
                },
            }
        )

    @bp.get("/disk")
    def storage():
        return jsonify({"free_bytes": free_bytes(_store().root), "jobs": disk.job_sizes(_store())})

    @bp.get("/jobs/<job_id>/log")
    def raw_log(job_id: str):
        job_dir_or_404(_store(), job_id)
        entry = _queue().entry_for_video(job_id)
        if entry is None or not _queue().log_path(entry.entry_id).is_file():
            abort(404, description="No worker log is available for this job.")
        lines = request.args.get("tail", default=200, type=int)
        return Response(
            log_tail(_queue().log_path(entry.entry_id), max(1, min(lines, 2000))),
            mimetype="text/plain",
            headers={"Cache-Control": "no-store"},
        )

    @bp.delete("/jobs/<job_id>/source")
    @serialized
    def delete_source(job_id: str):
        directory = job_dir_or_404(_store(), job_id)
        _idle(job_id)
        source = directory / "source.mp4"
        if source.is_symlink() or not source.resolve().is_relative_to(directory):
            abort(409, description="Source is linked outside this job and cannot be deleted.")
        if not source.is_file():
            abort(404, description="No downloaded source to delete. Local originals are kept.")
        state = _store().load(job_id)
        size = source.stat().st_size
        # Persist the restart requirement first, so a failed save cannot leave stale state.
        state.stage("ingest").status = StageStatus.pending
        state.stage("ingest").detail["source_deleted"] = True
        _store().save(state)
        source.unlink()
        return jsonify({"deleted_bytes": size})

    def _body() -> dict:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="body must be a JSON object")
        return payload

    def _reason(payload: dict) -> str:
        value = payload.get("reason", payload.get("override_reason"))
        if not isinstance(value, str) or len(value.strip()) < 10:
            abort(400, description="A reason of at least 10 characters is required.")
        return value.strip()

    @bp.post("/jobs/<job_id>/rights")
    @serialized
    def rights(job_id: str):
        job_dir_or_404(_store(), job_id)
        _idle(job_id)
        payload = _body()
        reason = _reason(payload)
        try:
            status = RightsStatus(payload.get("status"))
        except (ValueError, TypeError):
            abort(400, description="status must be green, amber, red or unknown")
        remember = payload.get("remember", False)
        if type(remember) is not bool:
            abort(400, description="remember must be a boolean")
        state = _store().load(job_id)
        entry = _queue().entry_for_video(job_id)
        resume = (
            entry
            and entry.status == "blocked"
            and status in {RightsStatus.green, RightsStatus.amber}
        )
        if resume:
            _require_runner()
        if remember:
            if not state.channel_id:
                abort(400, description="This source has no channel ID to remember.")
            runner = _runner()
            path = current_app.config.get("WATCHLIST_PATH") or (
                Path(runner.repo_root) / "config/watchlist.yaml"
                if runner
                else Path("config/watchlist.yaml")
            )
            try:
                append_channel(
                    Path(path),
                    {
                        "id": state.channel_id,
                        "name": state.channel_title or state.channel_id,
                        "rights_status": status.value,
                        "checked": utcnow().date().isoformat(),
                        "note": reason,
                    },
                )
            except (RightsError, OSError, ValueError) as exc:
                abort(409, description=str(exc))
        state.rights_status = status
        state.rights_note = reason
        state.rights_override = status in {RightsStatus.green, RightsStatus.amber}
        state.rights_override_reason = reason if state.rights_override else None
        _store().save(state)
        if entry:
            options = entry.options.model_copy(
                update={
                    "rights": status.value,
                    "force": False,
                    "allow_red": False,
                    "override_reason": None,
                }
            )
            _queue().update(
                entry.entry_id, rights_status=status.value, options=options.model_dump()
            )
            if resume:
                entry = _runner().retry(entry.entry_id)
        return jsonify({"status": status.value, "resumed": bool(resume)})

    @bp.post("/jobs/<job_id>/render")
    @serialized
    def render_job(job_id: str):
        directory = job_dir_or_404(_store(), job_id)
        _require_runner()
        _idle(job_id)
        payload = _body()
        allowed = {
            "approved_only",
            "clip_index",
            "allow_red",
            "override_reason",
            "force",
            "music",
            "no_music",
            "centre_crop",
        }
        if payload.keys() - allowed:
            abort(400, description="Unknown render options")
        for key in ("approved_only", "allow_red", "force", "no_music", "centre_crop"):
            if key in payload and type(payload[key]) is not bool:
                abort(400, description=f"{key} must be a boolean")
        index = payload.get("clip_index")
        if index is not None and (type(index) is not int or index < 1):
            abort(400, description="clip_index must be a positive integer")
        state = _store().load(job_id)
        if payload.get("allow_red"):
            payload["override_reason"] = _reason(payload)
        elif not state.is_exportable:
            abort(409, description="Classify the source rights before rendering.")
        if not state.clips or not (directory / "transcript.json").is_file():
            entry = _queue().entry_for_video(job_id)
            if (
                payload.get("allow_red")
                and entry
                and entry.status == "blocked"
                and not entry.options.render_only
            ):
                state.rights_override = True
                state.rights_override_reason = payload["override_reason"]
                _store().save(state)
                options = entry.options.model_copy(
                    update={
                        "allow_red": True,
                        "override_reason": payload["override_reason"],
                        "force": False,
                    }
                )
                _queue().update(entry.entry_id, options=options.model_dump())
                entry = _runner().retry(entry.entry_id)
                return jsonify({"entry": schemas.queue_entry_json(entry)}), 202
            abort(409, description="Resume the pipeline to finish transcription and moments first.")
        from clipper.ingest.download import IngestError, source_video_path

        try:
            source_video_path(_store(), state)
        except IngestError:
            if not state.stage("ingest").detail.get("source_deleted"):
                abort(409, description="Source is missing. Retry the pipeline to restore it first.")
        selected = (
            [c for c in state.clips if c.index == index]
            if index
            else [
                c
                for c in state.clips
                if c.approved is True
                or (not payload.get("approved_only") and c.approved is not False)
            ]
        )
        if not selected:
            abort(400, description="No clips match this render selection.")
        try:
            options = RunOptions.model_validate({**payload, "render_only": True})
        except ValidationError as exc:
            abort(400, description=str(exc))
        entry = _queue().enqueue(job_id, options)
        entry = _queue().update(
            entry.entry_id, video_id=job_id, rights_status=state.rights_status.value
        )
        _events().append(entry.entry_id, "submitted", detail="render selected clips")
        return jsonify({"entry": schemas.queue_entry_json(entry)}), 202

    @bp.get("/jobs")
    def jobs():
        summaries = presenter.build_index(_store())
        return jsonify(
            {
                "jobs": [schemas.job_summary_json(s) for s in summaries],
                "jobs_root": str(_store().root),
            }
        )

    @bp.get("/jobs/<job_id>")
    def job(job_id: str):
        store = _store()
        job_dir_or_404(store, job_id)
        view = presenter.build_job(store, job_id, weights=current_app.config["SCORING_WEIGHTS"])
        state = store.load(job_id)

        # Section 3, the single-writer discipline, made visible: while a worker
        # owns this job's state.json nobody else writes it, and the SPA greys
        # the verdict buttons rather than letting you discover that as a 409.
        queue = _queue()
        entry = queue.entry_for_video(job_id)
        writable = queue.running_video_id() != job_id
        payload = schemas.job_view_json(
            view,
            writable=writable,
            entry=schemas.queue_entry_json(entry) if entry else None,
        )
        # The stage strip is state-file data rather than presenter data: the
        # review page never needed it, and the queue page needs it to say
        # "stage 3 of 4" honestly rather than inferring progress from counts.
        payload["stages"] = [schemas.stage_json(name, state.stage(name)) for name in state.stages]
        payload["next_stage"] = state.next_pending_stage()
        payload["channel_id"] = state.channel_id
        payload["licence"] = state.licence
        payload["rights_override_reason"] = state.rights_override_reason
        return jsonify(payload)

    # -- the queue -------------------------------------------------------

    @bp.post("/jobs/<job_id>/clips/<clip_id>/review")
    @serialized
    def review_clip(job_id: str, clip_id: str):
        store = _store()
        job_dir_or_404(store, job_id)
        if _queue().running_video_id() == job_id:
            abort(409, description="Verdicts unlock when this job finishes. Nothing was changed.")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "approved" not in payload:
            abort(400, description="body must contain approved: true, false or null")
        approved = payload["approved"]
        if approved is not None and type(approved) is not bool:
            abort(400, description="approved must be true, false or null")
        note = payload.get("note")
        if note is not None and not isinstance(note, str):
            abort(400, description="note must be a string")
        try:
            presenter.apply_review(store, job_id, clip_id, approved=approved, note=note)
        except presenter.ClipNotFound as exc:
            abort(404, description=str(exc))
        view = presenter.build_job(store, job_id, weights=current_app.config["SCORING_WEIGHTS"])
        clip = next(c for c in view.clips if c.clip_id == clip_id)
        return jsonify(
            {
                "clip": schemas.clip_card_json(clip, job_id=job_id),
                "counts": schemas.counts_json(view.counts),
            }
        )

    def _downloads(job_id: str):
        directory = job_dir_or_404(_store(), job_id)
        if _queue().running_video_id() == job_id:
            abort(409, description="Downloads unlock when this job finishes.")
        state = _store().load(job_id)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", state.title or job_id).strip("-")[:100] or "clips"
        return directory, state, slug

    def _final(directory: Path, clip_id: str) -> Path:
        target = (directory / "clips" / clip_id / "final.mp4").resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            abort(404, description="This clip has no finished render.")
        return target

    @bp.get("/jobs/<job_id>/clips/<clip_id>/download")
    def download_clip(job_id: str, clip_id: str):
        directory, state, slug = _downloads(job_id)
        if not any(c.clip_id == clip_id for c in state.clips):
            abort(404, description="No such clip.")
        return send_file(
            _final(directory, clip_id),
            as_attachment=True,
            download_name=f"{slug}-{clip_id}.mp4",
            conditional=True,
            max_age=0,
        )

    @bp.get("/jobs/<job_id>/download")
    def download_job(job_id: str):
        directory, state, slug = _downloads(job_id)
        files = []
        for clip in state.clips:
            if (directory / "clips" / clip.clip_id / "final.mp4").is_file():
                files.append((_final(directory, clip.clip_id), f"{slug}-{clip.clip_id}.mp4"))
        if not files:
            abort(404, description="This job has no finished renders.")

        def chunks():
            # A non-seekable sink lets ZipFile emit descriptors as it goes.
            # Drain each block: neither full MP4s nor the archive live in RAM.
            class Sink:
                def __init__(self):
                    self.buffer = bytearray()

                def write(self, data):
                    self.buffer.extend(data)
                    return len(data)

                def flush(self):
                    pass

                def drain(self):
                    data = bytes(self.buffer)
                    self.buffer.clear()
                    return data

            sink = Sink()
            with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED) as archive:
                for path, name in files:
                    with (
                        path.open("rb") as source,
                        archive.open(name, "w", force_zip64=True) as dest,
                    ):
                        while block := source.read(64 * 1024):
                            dest.write(block)
                            yield sink.drain()
                    yield sink.drain()
            yield sink.drain()

        return Response(
            stream_with_context(chunks()),
            mimetype="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{slug}.zip"',
                "Cache-Control": "no-store",
            },
        )

    @bp.post("/queue")
    @serialized
    def submit():
        """Paste one or more sources. ``202``, because nothing has run yet.

        The response is the created entries, not a redirect: the SPA appends
        them to the table it is already polling, and the ``entry_id`` in each
        is what a cancel or a delete addresses before a ``video_id`` exists.
        """
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            abort(400, description="body must be a JSON object")

        sources = parse_sources(payload)
        options = parse_options(payload)

        entries = _queue().enqueue_many(sources, options)

        # The one place the server writes to an event log. Ownership passes to
        # the worker when the supervisor spawns it (phase 2), and the worker
        # appends after this line rather than beside it, so the file still has
        # exactly one writer at a time.
        log = _events()
        for entry in entries:
            log.append(entry.entry_id, "submitted", detail=entry.source)

        return jsonify({"entries": [schemas.queue_entry_json(e) for e in entries]}), 202

    @bp.get("/queue")
    def queue():
        """The whole queue, plus a page of events for one entry.

        One endpoint rather than two because the SPA polls this on a timer -
        1000 ms while anything runs, 5000 ms idle, stopped entirely when the
        tab is hidden - and two requests per tick would double that for no
        gain.

        ``since`` is a cursor into *one* entry's log, and ``events_entry_id``
        says which. A client whose ``events_entry_id`` changed between polls
        must reset ``since`` to 0: sequence numbers restart per entry, so
        carrying the old cursor over would skip the new job's first events.
        """
        store = _queue()
        log = _events()
        entries = store.load()
        running = next((e for e in entries if str(e.status) == "running"), None)

        requested = request.args.get("entry")
        if requested:
            focus = next((e for e in entries if e.entry_id == requested), None)
            if focus is None:
                abort(404, description=f"no queue entry {requested!r}")
        else:
            # The running entry is what you are watching; failing that, the
            # most recent submission, so the tail of the job that just
            # finished stays on screen instead of blanking.
            focus = running or (entries[-1] if entries else None)

        since = request.args.get("since", default=0, type=int) or 0
        limit = request.args.get("limit", default=DEFAULT_LIMIT, type=int) or DEFAULT_LIMIT
        limit = max(1, min(limit, MAX_EVENTS))

        events = log.read(focus.entry_id, since=since, limit=limit) if focus else []

        return jsonify(
            {
                "entries": [schemas.queue_entry_json(e) for e in entries],
                "running": schemas.queue_entry_json(running) if running else None,
                "events": [schemas.event_json(e) for e in events],
                "events_entry_id": focus.entry_id if focus else None,
                "seq": events[-1].seq if events else since,
                "runner": {
                    # Reported rather than omitted so the SPA can say "the
                    # queue is not being drained" instead of silently looking
                    # idle. False under ``clipper review``, which attaches no
                    # supervisor, and false if the thread died.
                    "alive": bool(getattr(_runner(), "alive", False)),
                    "free_bytes": free_bytes(_store().root),
                },
            }
        )

    @bp.delete("/queue/<entry_id>")
    @serialized
    def dequeue(entry_id: str):
        """Drop an entry. ``409`` while it is running - cancel it first.

        Deleting the row of a live worker would orphan the process: the
        supervisor reaps by looking the entry up, so a row that vanished
        mid-run leaves an ffmpeg tree with nothing tracking it.
        """
        try:
            entry = _queue().remove(entry_id)
        except QueueBusy as exc:
            return jsonify({"error": str(exc)}), 409
        except KeyError:
            abort(404, description=f"no queue entry {entry_id!r}")
        _events().delete(entry_id)
        return jsonify({"entry": schemas.queue_entry_json(entry)})

    @bp.post("/queue/<entry_id>/cancel")
    @serialized
    def cancel(entry_id: str):
        """Stop an entry. ``202``, because stopping is not instant.

        A queued entry is cancelled by the time this returns. A running one has
        only been *asked*: the sentinel is down, the worker exits at its next
        stage boundary, and if it is three minutes into an encode the
        supervisor kills the tree once the grace period expires. Returning
        ``200`` would be claiming the job has stopped, which is not yet true.
        """
        runner = _require_runner()
        try:
            entry = runner.cancel(entry_id)
        except QueueBusy as exc:
            return jsonify({"error": str(exc)}), 409
        except KeyError:
            abort(404, description=f"no queue entry {entry_id!r}")
        return jsonify({"entry": schemas.queue_entry_json(entry)}), 202

    @bp.post("/queue/<entry_id>/retry")
    @serialized
    def retry(entry_id: str):
        """Put a finished entry back in the queue, keeping its id and history.

        ``{"force": true}`` reruns every stage from the start; without it the
        retry resumes, because ``should_run`` skips completed stages and the
        expensive ones are additionally guarded on their output file. That is
        the difference between retrying an interrupted job and re-downloading
        two gigabytes to find out nothing was wrong with them.

        ``{"from_stage": "moments"}`` resets that stage and the ones after it
        to ``pending`` before requeueing - the narrow middle ground between the
        two, for when the transcript is fine and the candidates are not.
        """
        runner = _require_runner()

        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            abort(400, description="body must be a JSON object")
        force = payload.get("force", False)
        if type(force) is not bool:
            abort(400, description="force must be a boolean")
        from_stage = payload.get("from_stage")

        entry = _queue().get(entry_id)
        if entry is None:
            abort(404, description=f"no queue entry {entry_id!r}")

        if entry.status in ACTIVE_STATUSES:
            abort(409, description="This entry is already queued or running.")
        if entry.video_id:
            _idle(entry.video_id)
        if from_stage:
            if from_stage not in JOB_STAGES:
                abort(400, description=f"from_stage must be one of {list(JOB_STAGES)}")
            if entry.options.render_only and from_stage != "render":
                abort(400, description="A render-only entry can only restart from render.")
            if not entry.video_id:
                abort(400, description="this entry never produced a job, so it has no stages")
            _reset_stages_from(_store(), entry.video_id, from_stage)

        try:
            entry = runner.retry(entry_id, force=force)
        except QueueBusy as exc:
            return jsonify({"error": str(exc)}), 409
        except KeyError:
            abort(404, description=f"no queue entry {entry_id!r}")

        _events().append(entry_id, "submitted", detail=f"retry of {entry.label}")
        return jsonify({"entry": schemas.queue_entry_json(entry)}), 202

    @bp.get("/doctor")
    def doctor():
        """The same checks ``clipper doctor`` prints, as data.

        Key checks report presence only - see ``server/doctor.py``. No value of
        any environment variable reaches this response.
        """
        checks = doctor_module.diagnose(
            jobs_root=_store().root,
            config_path=current_app.config.get("SETTINGS_PATH", "config/settings.yaml"),
        )
        return jsonify(
            {
                "checks": [schemas.check_json(c) for c in checks],
                "status": doctor_module.overall_status(checks),
            }
        )

    return bp


__all__ = [
    "MAX_SOURCES_PER_SUBMIT",
    "create_blueprint",
    "free_bytes",
    "job_dir_or_404",
    "parse_options",
    "parse_sources",
]
