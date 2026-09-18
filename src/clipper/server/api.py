"""The JSON API (WEB_UI_PLAN.md section 4).

A Flask blueprint mounted at ``/api`` on the same app that serves the review
pages, so there is one process, one port and one job store. It is additive on
purpose: the Jinja routes, the ``/media/`` route and all 324 lines of
``tests/test_review.py`` keep working while the SPA is built beside them.

The read model is ``review/presenter.py`` - not a second one. This layer adds
routing and serialisation and nothing else, which is why every route here is a
few lines: the interesting code is in the presenter (what a reviewer sees) and
in ``schemas.py`` (what the wire looks like).

Phase 0 is read-only. There is no queue yet, so every job reports
``writable: true`` and carries no queue entry; phase 2 overlays both.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify

from clipper.review import presenter
from clipper.server import doctor as doctor_module
from clipper.server import schemas
from clipper.state import JobStore


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


def create_blueprint() -> Blueprint:
    """The ``/api`` blueprint. Reads its store and weights off ``app.config``.

    Built by a factory rather than declared at module level so that two apps in
    one test session (the review app and, later, the SPA app) each get their
    own, and so a blueprint can never outlive the config it was registered
    against.
    """
    bp = Blueprint("api", __name__, url_prefix="/api")

    def _store() -> JobStore:
        return current_app.config["JOB_STORE"]

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
        payload = schemas.job_view_json(view, writable=True, entry=None)
        # The stage strip is state-file data rather than presenter data: the
        # review page never needed it, and the queue page needs it to say
        # "stage 3 of 4" honestly rather than inferring progress from counts.
        payload["stages"] = [schemas.stage_json(name, state.stage(name)) for name in state.stages]
        payload["next_stage"] = state.next_pending_stage()
        return jsonify(payload)

    @bp.get("/doctor")
    def doctor():
        """The same checks ``clipper doctor`` prints, as data.

        Key checks report presence only - see ``server/doctor.py``. No value of
        any environment variable reaches this response.
        """
        checks = doctor_module.diagnose(jobs_root=_store().root)
        return jsonify(
            {
                "checks": [schemas.check_json(c) for c in checks],
                "status": doctor_module.overall_status(checks),
            }
        )

    return bp


__all__ = ["create_blueprint", "job_dir_or_404"]
