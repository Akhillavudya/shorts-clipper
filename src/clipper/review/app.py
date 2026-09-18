"""The local review page (FULL_PLAN.md section 12, Phase 4).

A Flask app with four routes and no database: ``jobs/<id>/state.json`` is the
database, exactly as it is for every other stage. The page lists the scored
candidates for a job, plays each one inline, and writes ``approved`` and
``review_note`` back through :class:`~clipper.state.JobStore`.

Two decisions worth stating, because both are load-bearing:

**It binds to localhost and it is not hardened.** There is no authentication,
no CSRF token and no rate limit, because the only client is the person running
it. The media route serves files out of the jobs root, so exposing this to a
network exposes every source video on the machine. ``serve()`` defaults to
``127.0.0.1`` and warns when it is asked for anything else.

**Review happens before render, so the preview is usually the source video.**
``presenter._preview`` picks the finished Short when one exists and otherwise
hands back ``source.mp4`` with the cut points; the page seeks to the start and
stops at the end. That is the whole reason this stage pays for itself - the
expensive encode only runs on clips someone has already watched and kept.

The stage buttons deliberately do nothing but write state. Rendering from a
web request would put a multi-minute ffmpeg run behind a browser tab that can
be closed halfway; ``clipper render <job>`` already skips rejected clips, and
``--approved-only`` narrows it to what was actually picked.
"""

from __future__ import annotations

import ipaddress
import logging
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from clipper import config as config_module
from clipper.review import presenter
from clipper.state import DEFAULT_JOBS_ROOT, JobStore

#: Accepted values of the ``approved`` form field, and what each one means.
VERDICTS: dict[str, bool | None] = {
    "approve": True,
    "reject": False,
    "reset": None,
}

log = logging.getLogger(__name__)


def _load_weights() -> dict[str, Any]:
    """Scoring weights, for showing each axis's weight next to its bar.

    Optional: the page is still readable without them, and a review UI that
    refuses to start because a config file moved is a worse failure than a
    missing number.
    """
    try:
        return config_module.load_scoring_weights()
    except Exception:  # noqa: BLE001 - reported on the page, not raised
        return {}


def create_app(
    *,
    store: JobStore | None = None,
    jobs_root: Path | str | None = None,
    weights: dict[str, Any] | None = None,
) -> Flask:
    """Build the app. ``store`` wins over ``jobs_root``; tests pass a temp one."""
    if store is None:
        store = JobStore(jobs_root if jobs_root is not None else DEFAULT_JOBS_ROOT)

    app = Flask(__name__)
    app.config["JOB_STORE"] = store
    app.config["SCORING_WEIGHTS"] = weights if weights is not None else _load_weights()
    # Everything on these pages is derived from a state file that another
    # process is expected to be writing. A cached page is a wrong page.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    def _store() -> JobStore:
        return app.config["JOB_STORE"]

    def _job_dir(job_id: str) -> Path:
        """Resolve a job directory, refusing anything outside the jobs root."""
        root = _store().root.resolve()
        directory = (root / job_id).resolve()
        if directory.parent != root or not _store().exists(job_id):
            abort(404, description=f"no job {job_id!r} under {_store().root}")
        return directory

    # -- pages ----------------------------------------------------------

    @app.get("/")
    def index() -> str:
        jobs = presenter.build_index(_store())
        return render_template("index.html", jobs=jobs, jobs_root=_store().root)

    @app.get("/jobs/<job_id>")
    def job(job_id: str) -> str:
        _job_dir(job_id)
        view = presenter.build_job(_store(), job_id, weights=app.config["SCORING_WEIGHTS"])
        return render_template("job.html", job=view)

    # -- the one write ---------------------------------------------------

    @app.post("/jobs/<job_id>/clips/<clip_id>/review")
    def review(job_id: str, clip_id: str):
        _job_dir(job_id)
        payload = request.get_json(silent=True) or {}
        wants_json = bool(payload) or request.accept_mimetypes.best == "application/json"

        if "approved" in payload:
            approved = payload["approved"]
            if approved not in (True, False, None):
                abort(400, description="approved must be true, false or null")
        else:
            verdict = request.form.get("verdict", "")
            if verdict not in VERDICTS:
                abort(400, description=f"verdict must be one of {sorted(VERDICTS)}")
            approved = VERDICTS[verdict]

        note = payload.get("note") if wants_json else request.form.get("note")

        try:
            clip = presenter.apply_review(_store(), job_id, clip_id, approved=approved, note=note)
        except presenter.ClipNotFound as exc:
            abort(404, description=str(exc))

        if not wants_json:
            return redirect(url_for("job", job_id=job_id) + f"#{clip_id}")

        view = presenter.build_job(_store(), job_id, weights=app.config["SCORING_WEIGHTS"])
        return jsonify(
            {
                "clip_id": clip.clip_id,
                "approved": clip.approved,
                "note": clip.review_note or "",
                "reviewed_at": clip.reviewed_at.isoformat() if clip.reviewed_at else None,
                "verdict": (
                    "approved"
                    if clip.approved is True
                    else "rejected"
                    if clip.approved is False
                    else "pending"
                ),
                "counts": {
                    "total": view.counts.total,
                    "approved": view.counts.approved,
                    "rejected": view.counts.rejected,
                    "pending": view.counts.pending,
                },
            }
        )

    # -- media -----------------------------------------------------------

    @app.get("/media/<job_id>/<path:relpath>")
    def media(job_id: str, relpath: str):
        """Stream a file out of one job directory.

        ``conditional=True`` is what makes the source fallback usable: without
        range requests the browser would pull a 1 GB episode from byte zero to
        play forty seconds fourteen minutes in.
        """
        directory = _job_dir(job_id)
        target = (directory / relpath).resolve()

        if not target.is_relative_to(directory):
            abort(404)
        if target.suffix.lower() not in presenter.MEDIA_SUFFIXES:
            abort(404, description=f"{target.suffix} is not a previewable media type")
        if not target.is_file():
            abort(404, description=f"{relpath} has not been produced yet")

        return send_file(target, conditional=True, max_age=0)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "jobs_root": str(_store().root)})

    # -- errors ----------------------------------------------------------

    @app.errorhandler(404)
    def not_found(exc):
        message = getattr(exc, "description", "not found")
        if request.path.startswith(("/media/", "/api/")):
            return jsonify({"error": message}), 404
        return render_template("error.html", code=404, message=message), 404

    @app.errorhandler(400)
    def bad_request(exc):
        message = getattr(exc, "description", "bad request")
        return jsonify({"error": message}), 400

    return app


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost", ""}


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 5151,
    jobs_root: Path | str | None = None,
    store: JobStore | None = None,
    debug: bool = False,
) -> None:
    """Run the development server. Called by ``clipper review``.

    Werkzeug's server is the right one here: single user, single machine, and
    it reloads nothing the pipeline depends on. Anything more is a deployment,
    and this is not deployed.
    """
    app = create_app(store=store, jobs_root=jobs_root)
    if not _is_loopback(host):
        log.warning(
            "serving the review UI on %s: the media route reads from %s, so every "
            "source video on this machine is reachable from that address.",
            host,
            app.config["JOB_STORE"].root,
        )
    app.run(host=host, port=port, debug=debug, threaded=True)


__all__ = ["VERDICTS", "create_app", "serve"]
