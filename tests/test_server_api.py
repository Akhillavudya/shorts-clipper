"""The JSON API (WEB_UI_PLAN.md sections 4 and 8).

Shaped exactly like ``tests/test_review.py``: no server is started,
``create_app()`` plus ``app.test_client()`` drives the same app object
``clipper review`` serves, and the store points at a temp directory.

Nothing here resolves a URL, shells out to ffmpeg, or sleeps. The API layer
does none of those things - resolution lives in the worker and ffmpeg lives in
``edit/`` - so a test that needed one would be evidence of a layering mistake.
"""

from __future__ import annotations

import pytest

from clipper.review.app import create_app
from clipper.state import JobStore, RightsStatus, StageStatus


@pytest.fixture
def store(tmp_path) -> JobStore:
    """Two jobs: one green with two scored clips, one unknown with none."""
    store = JobStore(tmp_path / "jobs")

    state = store.create(
        "vid123",
        source_url="https://www.youtube.com/watch?v=vid123",
        title="An Episode",
        channel_title="A Channel",
        duration_s=3600.0,
        rights_status=RightsStatus.green,
        rights_note="CC BY.",
    )
    state.add_clip(
        start_word_index=0,
        end_word_index=9,
        start=12.5,
        end=52.0,
        type="insight",
        rubric={"hook": 8.0, "payoff": 6.0},
        score=7.1,
        suggested_title="the boring fund wins",
    )
    state.add_clip(start_word_index=10, end_word_index=20, start=61.0, end=95.0, score=5.9)
    state.stage("ingest").status = StageStatus.done
    state.stage("render").status = StageStatus.failed
    state.stage("render").error = "\x1b[31mffmpeg: Invalid data found\x1b[0m"
    store.save(state)
    (store.job_dir("vid123") / "source.mp4").write_bytes(b"\x00fake source mp4")

    blocked = store.create("vid456", title="Another Episode")
    store.save(blocked)

    return store


@pytest.fixture
def client(store):
    app = create_app(store=store, weights={"weights": {"hook": 0.3, "payoff": 0.25}})
    app.config.update(TESTING=True)
    return app.test_client()


# -- GET /api/jobs ----------------------------------------------------------


def test_jobs_lists_every_job_with_its_counts(client):
    body = client.get("/api/jobs").get_json()
    by_id = {j["video_id"]: j for j in body["jobs"]}

    assert set(by_id) == {"vid123", "vid456"}
    assert by_id["vid123"]["title"] == "An Episode"
    assert by_id["vid123"]["counts"] == {
        "total": 2,
        "approved": 0,
        "rejected": 0,
        "pending": 2,
        "done": False,
    }
    assert by_id["vid456"]["rights_status"] == "unknown"
    assert by_id["vid456"]["exportable"] is False


def test_jobs_emits_datetimes_as_strings(client):
    """The bug class ``jsonify`` would otherwise raise on, asserted positively."""
    for job in client.get("/api/jobs").get_json()["jobs"]:
        assert isinstance(job["updated_at"], str)


def test_jobs_reports_the_root_it_is_reading(client, store):
    assert client.get("/api/jobs").get_json()["jobs_root"] == str(store.root)


def test_jobs_is_empty_rather_than_404_on_a_fresh_install(tmp_path):
    app = create_app(jobs_root=tmp_path / "empty", weights={})
    assert app.test_client().get("/api/jobs").get_json()["jobs"] == []


# -- GET /api/jobs/<id> -----------------------------------------------------


def test_job_carries_the_presenter_properties_the_page_needs(client):
    body = client.get("/api/jobs/vid123").get_json()

    assert body["duration_label"] == "1:00:00"
    clip = body["clips"][0]
    assert clip["verdict"] == "pending"
    assert clip["span_label"] == "0:12 - 0:52"
    assert clip["score_label"] == "7.10"
    assert clip["rubric"][0] == {
        "name": "hook",
        "label": "hook",
        "score": 8.0,
        "weight": 0.3,
        "percent": 80.0,
    }


def test_job_preview_falls_back_to_the_source_with_the_cut_points(client):
    clip = client.get("/api/jobs/vid123").get_json()["clips"][0]
    assert clip["preview"]["url"] == "/media/vid123/source.mp4"
    assert clip["preview"]["is_source"] is True
    assert (clip["preview"]["start"], clip["preview"]["end"]) == (12.5, 52.0)


def test_the_preview_url_is_actually_fetchable(client):
    """The one route that proves ``media_url`` and the media route agree."""
    url = client.get("/api/jobs/vid123").get_json()["clips"][0]["preview"]["url"]
    assert client.get(url).status_code == 200


def test_job_reports_stages_and_the_next_pending_one(client):
    body = client.get("/api/jobs/vid123").get_json()
    stages = {s["name"]: s for s in body["stages"]}

    assert list(stages) == ["ingest", "transcribe", "moments", "render"]
    assert stages["ingest"]["status"] == "done"
    assert stages["transcribe"]["status"] == "pending"
    assert body["next_stage"] == "transcribe"


def test_a_surfaced_stage_error_has_its_ansi_stripped(client, store):
    stages = {s["name"]: s for s in client.get("/api/jobs/vid123").get_json()["stages"]}
    assert stages["render"]["error"] == "ffmpeg: Invalid data found"

    # ...and the state file keeps what ffmpeg actually said.
    assert "\x1b[31m" in store.load("vid123").stage("render").error


def test_job_is_writable_with_no_owning_entry_before_the_queue_exists(client):
    body = client.get("/api/jobs/vid123").get_json()
    assert body["writable"] is True
    assert body["entry"] is None


def test_a_job_the_gate_would_refuse_says_so_in_its_warnings(client):
    body = client.get("/api/jobs/vid456").get_json()
    assert body["exportable"] is False
    assert any("rights_status is 'unknown'" in w for w in body["warnings"])


# -- the 404 surface --------------------------------------------------------


def test_an_unknown_job_is_a_json_404(client):
    response = client.get("/api/jobs/nope")
    assert response.status_code == 404
    assert response.is_json
    assert "nope" in response.get_json()["error"]


@pytest.mark.parametrize(
    "job_id",
    ["../etc", "..%2f..", "vid123/../../secrets", "a/b"],
)
def test_the_api_cannot_be_talked_out_of_the_jobs_root(client, job_id):
    """The same guard the media route uses, reached through a different door."""
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_an_unknown_api_route_is_json_not_html(client):
    """Guards the route precedence the SPA catch-all will depend on in phase 3."""
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.is_json


# -- GET /api/doctor --------------------------------------------------------


def test_doctor_returns_checks_and_an_overall_status(client):
    body = client.get("/api/doctor").get_json()

    assert body["status"] in {"ok", "warn", "fail"}
    assert body["checks"]
    for check in body["checks"]:
        assert set(check) == {"name", "status", "detail", "hint"}
        assert check["status"] in {"ok", "warn", "fail"}


def test_doctor_reports_key_presence_and_never_a_key_value(client, monkeypatch):
    """Hard requirement: the doctor must not become a secret-exfiltration route."""
    secret = "sk-not-a-real-key-0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("GROQ_API_KEY", secret)

    response = client.get("/api/doctor")
    assert secret not in response.get_data(as_text=True)


def test_doctor_measures_the_store_the_api_is_serving(client, store):
    body = client.get("/api/doctor").get_json()
    jobs_root = next(c for c in body["checks"] if c["name"] == "jobs root")
    assert str(store.root) in jobs_root["detail"]


# -- the api switch ---------------------------------------------------------


def test_the_jinja_ui_stands_up_without_the_api(store):
    app = create_app(store=store, weights={}, api=False)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/jobs").status_code == 404
