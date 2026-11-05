"""The JSON API (WEB_UI_PLAN.md sections 4 and 8).

Shaped exactly like ``tests/test_review.py``: no server is started,
``create_app()`` plus ``app.test_client()`` drives the same app object
``clipper review`` serves, and the store points at a temp directory.

Nothing here resolves a URL, shells out to ffmpeg, or sleeps. The API layer
does none of those things - resolution lives in the worker and ffmpeg lives in
``edit/`` - so a test that needed one would be evidence of a layering mistake.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from clipper.review.app import create_app
from clipper.server.events import EventLog
from clipper.server.queue import QueueStore
from clipper.state import JobStore, RightsStatus, StageStatus, utcnow


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


def test_source_delete_preserves_clips_and_resets_ingest(client, store):
    directory = store.job_dir("vid123")
    (directory / "scored.json").write_text("[]")
    before = store.load("vid123").clips
    assert client.delete("/api/jobs/vid123/source").status_code == 200
    assert not (directory / "source.mp4").exists()
    state = store.load("vid123")
    assert state.stage("ingest").status == StageStatus.pending
    assert state.clips == before
    assert (directory / "scored.json").is_file()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_source_delete_active_job_changes_nothing(client, store, status):
    queue = QueueStore(store.root)
    entry = queue.enqueue("source")
    queue.update(entry.entry_id, status=status, video_id="vid123")
    before = store.state_path("vid123").read_bytes()
    assert client.delete("/api/jobs/vid123/source").status_code == 409
    assert store.state_path("vid123").read_bytes() == before
    assert (store.job_dir("vid123") / "source.mp4").is_file()


def test_source_delete_never_deletes_local_original(client, store, tmp_path):
    original = tmp_path / "original.mp4"
    original.write_bytes(b"original")
    state = store.load("vid456")
    state.source_path = str(original)
    store.save(state)
    assert client.delete("/api/jobs/vid456/source").status_code == 404
    assert original.read_bytes() == b"original"


def test_raw_log_tail_and_missing_log(client, store):
    assert client.get("/api/jobs/vid123/log").status_code == 404
    queue = QueueStore(store.root)
    entry = queue.enqueue("source")
    queue.update(entry.entry_id, video_id="vid123")
    path = queue.log_path(entry.entry_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"first\nsecond\nlast\n")
    response = client.get("/api/jobs/vid123/log?tail=2")
    assert response.mimetype == "text/plain"
    assert response.data == b"second\nlast"
    assert client.get("/api/jobs/vid123/log?tail=nope").status_code == 200


def test_disk_and_settings_do_not_expose_secrets(client, store, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-should-not-appear")
    disk = client.get("/api/disk").get_json()
    first = next(j for j in disk["jobs"] if j["video_id"] == "vid123")
    assert first["bytes"] >= first["source_bytes"] > 0
    settings = client.get("/api/settings")
    assert settings.status_code == 200
    assert b"secret-should-not-appear" not in settings.data
    assert set(settings.get_json()["values"]) == {
        "server.min_free_gb",
        "edit.reframe",
        "edit.music_enabled",
    }


@pytest.mark.parametrize("approved", [True, False, None])
def test_spa_review_persists_and_returns_presenter(client, store, approved):
    response = client.post(
        "/api/jobs/vid123/clips/clip_01/review",
        json={"approved": approved, "note": "  Useful hook  "},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["clip"]["approved"] is approved
    assert body["clip"]["review_note"] == "Useful hook"
    assert body["clip"]["preview"]["url"].startswith("/media/")
    assert body["counts"]["total"] == 2
    clip = store.load("vid123").clip("clip_01")
    assert clip.approved is approved
    assert clip.reviewed_at is not None
    assert clip.review_note == "Useful hook"


@pytest.mark.parametrize(
    "payload",
    [[], "yes", {}, {"approved": 1}, {"approved": "true"}, {"approved": False, "note": []}],
)
def test_spa_review_rejects_invalid_body_without_writing(client, store, payload):
    path = store.job_dir("vid123") / "state.json"
    before = path.read_bytes()
    assert client.post("/api/jobs/vid123/clips/clip_01/review", json=payload).status_code == 400
    assert path.read_bytes() == before


def test_spa_review_missing_clip(client):
    assert (
        client.post("/api/jobs/vid123/clips/missing/review", json={"approved": True}).status_code
        == 404
    )


def test_spa_review_preserves_note_when_omitted(client, store):
    client.post("/api/jobs/vid123/clips/clip_01/review", json={"approved": True, "note": "Keep"})
    client.post("/api/jobs/vid123/clips/clip_01/review", json={"approved": None})
    assert store.load("vid123").clip("clip_01").review_note == "Keep"


def test_spa_review_and_downloads_refuse_running_job(client, store):
    queue = QueueStore(store.root)
    entry = queue.enqueue("source.mp4")
    queue.update(entry.entry_id, status="running", video_id="vid123")
    path = store.job_dir("vid123") / "state.json"
    before = path.read_bytes()
    assert (
        client.post("/api/jobs/vid123/clips/clip_01/review", json={"approved": True}).status_code
        == 409
    )
    assert path.read_bytes() == before
    assert client.get("/api/jobs/vid123/clips/clip_01/download").status_code == 409
    assert client.get("/api/jobs/vid123/download").status_code == 409


def test_downloads_only_final_files_and_streams_valid_zip(client, store):
    directory = store.job_dir("vid123")
    expected = {}
    for clip_id in ("clip_01", "clip_02"):
        folder = directory / "clips" / clip_id
        folder.mkdir(parents=True, exist_ok=True)
        payload = (clip_id.encode() + b"\x00") * 20000
        (folder / "final.mp4").write_bytes(payload)
        (folder / "cut.mp4").write_bytes(b"not a final render")
        expected[f"An-Episode-{clip_id}.mp4"] = payload
    response = client.get("/api/jobs/vid123/clips/clip_01/download")
    assert response.status_code == 200
    assert response.data == expected["An-Episode-clip_01.mp4"]
    assert "attachment;" in response.headers["Content-Disposition"]
    assert "An-Episode-clip_01.mp4" in response.headers["Content-Disposition"]
    response.close()
    response = client.get("/api/jobs/vid123/download", buffered=False)
    assert response.is_streamed
    blocks = list(response.response)
    assert max(map(len, blocks)) < 70000
    with zipfile.ZipFile(io.BytesIO(b"".join(blocks))) as archive:
        assert set(archive.namelist()) == set(expected)
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert archive.read(info) == expected[info.filename]


@pytest.mark.parametrize(
    "path",
    [
        "/api/jobs/missing/download",
        "/api/jobs/vid123/download",
        "/api/jobs/vid123/clips/clip_01/download",
        "/api/jobs/vid123/clips/missing/download",
    ],
)
def test_download_missing_files_are_json_404(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_download_does_not_export_unregistered_clip(client, store):
    folder = store.job_dir("vid123") / "clips" / "orphan"
    folder.mkdir(parents=True)
    (folder / "final.mp4").write_bytes(b"orphan")
    assert client.get("/api/jobs/vid123/clips/orphan/download").status_code == 404
    assert client.get("/api/jobs/vid123/download").status_code == 404


def test_download_rejects_resolved_path_outside_job(client, store, monkeypatch, tmp_path):
    from pathlib import Path

    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"private")
    resolve = Path.resolve

    def escape_final(path, *args, **kwargs):
        return outside if path.name == "final.mp4" else resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", escape_final)
    assert client.get("/api/jobs/vid123/clips/clip_01/download").status_code == 404


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


# -- the queue --------------------------------------------------------------


def submit(client, *sources, **options):
    payload = {"sources": list(sources)}
    if options:
        payload["options"] = options
    return client.post("/api/queue", json=payload)


def test_submit_accepts_several_sources_in_one_request(client, store):
    response = submit(
        client,
        "https://www.youtube.com/watch?v=aaa",
        "https://www.youtube.com/watch?v=bbb",
    )

    assert response.status_code == 202
    entries = response.get_json()["entries"]
    assert [e["source"] for e in entries] == [
        "https://www.youtube.com/watch?v=aaa",
        "https://www.youtube.com/watch?v=bbb",
    ]
    assert {e["status"] for e in entries} == {"queued"}

    # And it is on disk, which is the whole point of the phase.
    stored = QueueStore(store.root).load()
    assert [e.entry_id for e in stored] == [e["entry_id"] for e in entries]


def test_a_submitted_entry_survives_a_restarted_app(client, store):
    submit(client, "a", "b", "c")

    # A second app over the same jobs root - the server was restarted.
    restarted = create_app(store=store, weights={}).test_client()
    body = restarted.get("/api/queue").get_json()

    assert [e["source"] for e in body["entries"]] == ["a", "b", "c"]
    assert [e["status"] for e in body["entries"]] == ["queued"] * 3


@pytest.mark.parametrize("key", ["source", "sources"])
def test_submit_accepts_a_bare_string(client, key):
    """One link pasted by hand, or by curl, should not have to be an array."""
    response = client.post("/api/queue", json={key: "https://example.test/a"})

    assert response.status_code == 202
    assert len(response.get_json()["entries"]) == 1


def test_submit_carries_the_run_options(client):
    response = submit(client, "a", rights="green", no_music=True, clip_index=2)

    options = response.get_json()["entries"][0]["options"]
    assert options["rights"] == "green"
    assert options["no_music"] is True
    assert options["clip_index"] == 2
    assert options["allow_red"] is False


def test_submit_refuses_allow_red(client, store):
    """Section 5: a rights override is not something you can pre-authorise."""
    response = submit(client, "a", allow_red=True)

    assert response.status_code == 400
    assert "allow_red" in response.get_json()["error"]
    assert QueueStore(store.root).load() == []


def test_submit_refuses_an_unknown_rights_value(client):
    response = submit(client, "a", rights="probably fine")

    assert response.status_code == 400
    assert "rights" in response.get_json()["error"]


def test_submit_refuses_an_unknown_option(client):
    response = submit(client, "a", alow_red=True)

    assert response.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"sources": []},
        {"sources": ["   "]},
        {"sources": [42]},
        {"sources": {"url": "a"}},
        {"sources": ["a"], "options": "not an object"},
    ],
)
def test_submit_refuses_a_malformed_body(client, payload):
    assert client.post("/api/queue", json=payload).status_code == 400


def test_submit_refuses_a_paste_of_the_whole_internet(client):
    response = submit(client, *[f"https://example.test/{i}" for i in range(50)])

    assert response.status_code == 400


def test_submit_writes_a_submitted_event(client, store):
    entry_id = submit(client, "https://example.test/a").get_json()["entries"][0]["entry_id"]

    events = EventLog(store.root).read(entry_id)

    assert [(e.seq, e.kind, e.detail) for e in events] == [
        (1, "submitted", "https://example.test/a")
    ]


def test_queue_is_empty_before_anything_is_submitted(client):
    body = client.get("/api/queue").get_json()

    assert body["entries"] == []
    assert body["running"] is None
    assert body["events"] == []
    assert body["events_entry_id"] is None
    assert body["seq"] == 0


def test_queue_reports_no_runner_when_none_is_attached(client):
    """``clipper review`` attaches no supervisor, and the page must say so.

    Reported rather than omitted: "nothing is draining the queue" and "the
    queue is empty" look identical otherwise, and only one of them is a bug.
    """
    runner = client.get("/api/queue").get_json()["runner"]

    assert runner["alive"] is False
    assert runner["free_bytes"] is None or runner["free_bytes"] > 0


def test_queue_surfaces_the_running_entry(client, store):
    entry_id = submit(client, "a", "b").get_json()["entries"][0]["entry_id"]
    QueueStore(store.root).update(
        entry_id, status="running", video_id="vid123", pid=4242, started_at=utcnow()
    )

    body = client.get("/api/queue").get_json()

    assert body["running"]["entry_id"] == entry_id
    assert body["running"]["video_id"] == "vid123"
    assert body["running"]["pid"] == 4242
    assert isinstance(body["running"]["started_at"], str)


def test_queue_events_follow_the_since_cursor(client, store):
    entry_id = submit(client, "a").get_json()["entries"][0]["entry_id"]
    log = EventLog(store.root)
    log.append(entry_id, "started", "ingest")
    log.append(entry_id, "progress", "ingest", "downloading")

    body = client.get("/api/queue").get_json()
    assert [e["seq"] for e in body["events"]] == [1, 2, 3]
    assert body["events_entry_id"] == entry_id
    assert body["seq"] == 3

    log.append(entry_id, "progress", "transcribe")
    body = client.get("/api/queue?since=" + str(body["seq"])).get_json()

    assert [e["kind"] for e in body["events"]] == ["progress"]
    assert body["seq"] == 4


def test_queue_events_follow_the_running_entry_not_the_newest(client, store):
    first, _second = submit(client, "a", "b").get_json()["entries"]
    QueueStore(store.root).update(first["entry_id"], status="running")

    body = client.get("/api/queue").get_json()

    assert body["events_entry_id"] == first["entry_id"]
    assert [e["detail"] for e in body["events"]] == ["a"]


def test_queue_events_can_be_asked_for_by_entry(client, store):
    _first, second = submit(client, "a", "b").get_json()["entries"]

    body = client.get("/api/queue?entry=" + second["entry_id"]).get_json()

    assert body["events_entry_id"] == second["entry_id"]
    assert [e["detail"] for e in body["events"]] == ["b"]


def test_queue_404s_on_an_unknown_entry(client):
    assert client.get("/api/queue?entry=nope").status_code == 404


def test_queue_entries_are_json_safe(client, store):
    """The datetime class of bugs, asserted where jsonify would have raised."""
    entry_id = submit(client, "a").get_json()["entries"][0]["entry_id"]
    QueueStore(store.root).update(
        entry_id,
        status="failed",
        started_at=utcnow(),
        finished_at=utcnow(),
        last_alive=utcnow(),
        error="\x1b[31mffmpeg died\x1b[0m",
        exit_code=1,
    )

    entry = client.get("/api/queue").get_json()["entries"][0]

    assert set(entry) == {
        "entry_id",
        "source",
        "status",
        "label",
        "options",
        "video_id",
        "rights_status",
        "pid",
        "submitted_at",
        "started_at",
        "finished_at",
        "last_alive",
        "error",
        "exit_code",
    }
    assert entry["error"] == "ffmpeg died"


def test_delete_removes_a_queued_entry_and_its_events(client, store):
    first, second = submit(client, "a", "b").get_json()["entries"]

    response = client.delete("/api/queue/" + first["entry_id"])

    assert response.status_code == 200
    assert [e.source for e in QueueStore(store.root).load()] == ["b"]
    assert not EventLog(store.root).exists(first["entry_id"])
    assert EventLog(store.root).exists(second["entry_id"])


def test_delete_refuses_the_running_entry_with_409(client, store):
    entry_id = submit(client, "a").get_json()["entries"][0]["entry_id"]
    QueueStore(store.root).update(entry_id, status="running", pid=4242)

    response = client.delete("/api/queue/" + entry_id)

    assert response.status_code == 409
    assert "cancel" in response.get_json()["error"]
    assert len(QueueStore(store.root).load()) == 1


def test_delete_404s_on_an_unknown_entry(client):
    assert client.delete("/api/queue/nope").status_code == 404


def test_the_queue_directory_is_not_a_job(client, store):
    """``_queue`` must never show up in the job list."""
    submit(client, "a")

    assert "_queue" not in {j["video_id"] for j in client.get("/api/jobs").get_json()["jobs"]}


# -- the single-writer discipline (section 3) -------------------------------


def test_a_job_is_writable_when_nothing_owns_it(client):
    body = client.get("/api/jobs/vid123").get_json()

    assert body["writable"] is True
    assert body["entry"] is None


def test_a_running_job_reports_itself_unwritable_and_names_its_entry(client, store):
    submitted = submit(client, "https://www.youtube.com/watch?v=vid123").get_json()
    entry_id = submitted["entries"][0]["entry_id"]
    QueueStore(store.root).update(entry_id, status="running", video_id="vid123", pid=4242)

    body = client.get("/api/jobs/vid123").get_json()

    assert body["writable"] is False
    assert body["entry"]["entry_id"] == entry_id
    # The other job is unaffected: one worker locks one state file.
    assert client.get("/api/jobs/vid456").get_json()["writable"] is True


def test_a_finished_entry_still_rides_along_with_its_job(client, store):
    submitted = submit(client, "https://www.youtube.com/watch?v=vid123").get_json()
    entry_id = submitted["entries"][0]["entry_id"]
    QueueStore(store.root).update(entry_id, status="done", video_id="vid123")

    body = client.get("/api/jobs/vid123").get_json()

    assert body["writable"] is True
    assert body["entry"]["status"] == "done"


# -- the api switch ---------------------------------------------------------


def test_the_jinja_ui_stands_up_without_the_api(store):
    app = create_app(store=store, weights={}, api=False)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/jobs").status_code == 404


# -- the runner (phase 2) ---------------------------------------------------


class FakeProcess:
    """A ``ProcessHandle`` with no process behind it. See test_server_runner."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.code: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        return self.code

    def kill_tree(self, *, timeout_s: float = 10.0) -> None:
        self.killed = True
        if self.code is None:
            self.code = 130


@pytest.fixture
def runner_client(store, clock):
    """The same app with a supervisor attached, as ``clipper ui`` will run it.

    A real :class:`JobRunner` over fake processes rather than a mock runner:
    the routes' job is to hand the supervisor a decision and serialise what
    comes back, and a mock would assert the handing rather than the outcome.
    """
    from clipper.server.runner import JobRunner

    spawned: list[FakeProcess] = []

    def spawn(entry):
        proc = FakeProcess(pid=4000 + len(spawned))
        proc.entry = entry
        spawned.append(proc)
        return proc

    runner = JobRunner(
        queue=QueueStore(store.root),
        events=EventLog(store.root),
        spawn=spawn,
        clock=clock,
        min_free_bytes=0,
    )
    app = create_app(store=store, weights={}, runner=runner)
    app.config.update(TESTING=True)
    client = app.test_client()
    client.runner = runner
    client.spawned = spawned
    return client


def test_the_api_and_the_supervisor_share_one_queue(runner_client):
    """Two ``QueueStore`` instances over one file would be two pictures of it."""
    submit(runner_client, "https://youtu.be/one")
    runner_client.runner.tick()

    body = runner_client.get("/api/queue").get_json()
    # The supervisor thread is never started in tests - tick() is driven by
    # hand - so ``alive`` is honestly false even though a job is running.
    assert body["runner"]["alive"] is False
    assert body["running"]["status"] == "running"
    assert body["running"]["pid"] == 4000


def test_cancel_a_queued_entry_is_immediate(runner_client):
    entry_id = submit(runner_client, "https://youtu.be/one").get_json()["entries"][0]["entry_id"]

    response = runner_client.post(f"/api/queue/{entry_id}/cancel")

    assert response.status_code == 202
    assert response.get_json()["entry"]["status"] == "cancelled"
    runner_client.runner.tick()
    assert runner_client.spawned == []


def test_cancel_a_running_entry_is_a_request_not_a_fact(runner_client, store):
    """``202``: the sentinel is down, the worker has not stopped yet.

    ``200`` would be claiming the job had already stopped, which is exactly the
    thing that is not true for three more minutes of encode.
    """
    entry_id = submit(runner_client, "https://youtu.be/one").get_json()["entries"][0]["entry_id"]
    runner_client.runner.tick()

    response = runner_client.post(f"/api/queue/{entry_id}/cancel")

    assert response.status_code == 202
    assert response.get_json()["entry"]["status"] == "running"
    assert QueueStore(store.root).cancel_path(entry_id).is_file()


def test_cancel_404s_and_409s(runner_client):
    assert runner_client.post("/api/queue/nope/cancel").status_code == 404

    entry_id = submit(runner_client, "https://youtu.be/one").get_json()["entries"][0]["entry_id"]
    runner_client.runner.tick()
    runner_client.spawned[0].code = 0
    runner_client.runner.tick()

    conflict = runner_client.post(f"/api/queue/{entry_id}/cancel")
    assert conflict.status_code == 409
    assert "not running" in conflict.get_json()["error"]


def test_cancel_without_a_runner_is_a_503_not_a_lie(client):
    """Without a supervisor there is nothing to cancel *with*.

    Accepting the request and doing nothing would leave the row saying
    ``cancelled`` while its worker - if there were one - carried on.
    """
    entry_id = submit(client, "https://youtu.be/one").get_json()["entries"][0]["entry_id"]

    response = client.post(f"/api/queue/{entry_id}/cancel")

    assert response.status_code == 503
    assert "clipper ui" in response.get_json()["error"]


def test_retry_puts_a_failed_entry_back_in_the_queue(runner_client):
    entry_id = submit(runner_client, "https://youtu.be/one").get_json()["entries"][0]["entry_id"]
    runner_client.runner.tick()
    runner_client.spawned[0].code = 1
    runner_client.runner.tick()

    response = runner_client.post(f"/api/queue/{entry_id}/retry", json={})

    assert response.status_code == 202
    entry = response.get_json()["entry"]
    assert entry["entry_id"] == entry_id
    assert entry["status"] == "queued"
    assert entry["error"] is None


def test_retry_with_force_reruns_every_stage(runner_client):
    entry_id = submit(runner_client, "https://youtu.be/one").get_json()["entries"][0]["entry_id"]
    runner_client.runner.tick()
    runner_client.spawned[0].code = 1
    runner_client.runner.tick()

    body = runner_client.post(f"/api/queue/{entry_id}/retry", json={"force": True}).get_json()

    assert body["entry"]["options"]["force"] is True


def test_retry_from_a_stage_resets_that_stage_and_the_ones_after_it(runner_client, store):
    """The narrow middle ground: the transcript is fine, the candidates are not.

    Allowed to write ``state.json`` from the server process only because the
    entry is provably not running - ``retry`` refuses a queued or running one -
    and the row goes back in the queue afterwards.
    """
    state = store.load("vid123")
    for name in ("transcribe", "moments", "render"):
        state.stage(name).status = StageStatus.done
    store.save(state)

    entry_id = submit(runner_client, "https://youtu.be/vid123").get_json()["entries"][0]["entry_id"]
    QueueStore(store.root).update(entry_id, status="failed", video_id="vid123")

    response = runner_client.post(f"/api/queue/{entry_id}/retry", json={"from_stage": "moments"})

    assert response.status_code == 202
    after = store.load("vid123")
    assert str(after.stage("ingest").status) == "done"
    assert str(after.stage("transcribe").status) == "done"
    assert str(after.stage("moments").status) == "pending"
    assert str(after.stage("render").status) == "pending"


def test_retry_refuses_an_unknown_stage(runner_client, store):
    entry_id = submit(runner_client, "https://youtu.be/vid123").get_json()["entries"][0]["entry_id"]
    QueueStore(store.root).update(entry_id, status="failed", video_id="vid123")

    response = runner_client.post(f"/api/queue/{entry_id}/retry", json={"from_stage": "publish"})

    assert response.status_code == 400


def test_retry_refuses_an_entry_that_is_already_queued(runner_client):
    entry_id = submit(runner_client, "https://youtu.be/one").get_json()["entries"][0]["entry_id"]

    response = runner_client.post(f"/api/queue/{entry_id}/retry", json={})

    assert response.status_code == 409


def test_retry_404s_on_an_unknown_entry(runner_client):
    assert runner_client.post("/api/queue/nope/retry", json={}).status_code == 404


# -- phase 5: rights, render-only, retry ownership ---------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": "green", "reason": "short"},
        {"status": "blue", "reason": "Permission on file"},
        {"status": "green", "reason": "Permission on file", "remember": "yes"},
    ],
)
def test_rights_invalid_body_preserves_state(client, store, payload):
    path = store.job_dir("vid456") / "state.json"
    before = path.read_bytes()
    assert client.post("/api/jobs/vid456/rights", json=payload).status_code == 400
    assert path.read_bytes() == before


def test_classify_remember_and_resume(runner_client, store, tmp_path):
    import yaml

    state = store.load("vid456")
    state.channel_id = "UCpermission"
    state.channel_title = "Permissioned creator"
    state.stage("ingest").status = StageStatus.done
    store.save(state)
    watchlist = tmp_path / "watchlist.yaml"
    watchlist.write_text("# preserve me\nchannels: []\n", encoding="utf-8")
    runner_client.application.config["WATCHLIST_PATH"] = watchlist
    queue = runner_client.runner.queue
    entry = queue.enqueue("https://youtu.be/vid456")
    queue.update(entry.entry_id, status="blocked", video_id="vid456")
    response = runner_client.post(
        "/api/jobs/vid456/rights",
        json={"status": "green", "reason": "Written permission on file", "remember": True},
    )
    assert response.status_code == 200
    assert response.json["resumed"] is True
    saved = store.load("vid456")
    assert saved.is_exportable and saved.rights_override
    assert saved.rights_override_reason == "Written permission on file"
    assert saved.is_done("ingest")
    assert yaml.safe_load(watchlist.read_text())["channels"][0]["id"] == "UCpermission"
    assert queue.get(entry.entry_id).status == "queued"
    assert not queue.get(entry.entry_id).options.force


@pytest.mark.parametrize("status", ["queued", "running"])
@pytest.mark.parametrize(
    "route,payload",
    [("rights", {"status": "green", "reason": "Permission recorded"}), ("render", {})],
)
def test_job_mutations_refuse_active_owner(runner_client, store, status, route, payload):
    queue = runner_client.runner.queue
    entry = queue.enqueue("source")
    queue.update(entry.entry_id, status=status, video_id="vid123")
    path = store.job_dir("vid123") / "state.json"
    before = path.read_bytes()
    assert runner_client.post(f"/api/jobs/vid123/{route}", json=payload).status_code == 409
    assert path.read_bytes() == before


@pytest.mark.parametrize("status", ["queued", "running"])
def test_retry_from_stage_cannot_reset_active_job(runner_client, store, status):
    queue = runner_client.runner.queue
    entry = queue.enqueue("source")
    queue.update(entry.entry_id, status=status, video_id="vid123")
    path = store.job_dir("vid123") / "state.json"
    before = path.read_bytes()
    assert (
        runner_client.post(
            f"/api/queue/{entry.entry_id}/retry", json={"from_stage": "ingest"}
        ).status_code
        == 409
    )
    assert path.read_bytes() == before


def test_render_only_enqueues_identity_and_refuses_duplicate(runner_client, store):
    (store.job_dir("vid123") / "transcript.json").write_text("{}")
    response = runner_client.post("/api/jobs/vid123/render", json={"clip_index": 1})
    assert response.status_code == 202
    entry = response.json["entry"]
    assert entry["video_id"] == entry["source"] == "vid123"
    assert entry["options"]["render_only"] is True
    assert entry["options"]["clip_index"] == 1
    assert runner_client.post("/api/jobs/vid123/render", json={}).status_code == 409
    runner_client.runner.tick()
    assert runner_client.spawned[0].entry.options.render_only


@pytest.mark.parametrize(
    "payload",
    [
        {"allow_red": True},
        {"clip_index": True},
        {"clip_index": -1},
        {"force": "yes"},
        {"clip_index": 999},
        {"approved_only": True},
    ],
)
def test_render_invalid_selection_or_override(runner_client, store, payload):
    (store.job_dir("vid123") / "transcript.json").write_text("{}")
    assert runner_client.post("/api/jobs/vid123/render", json=payload).status_code == 400
    assert runner_client.runner.queue.load() == []


def test_render_refuses_unknown_rights(runner_client, store):
    state = store.load("vid123")
    state.rights_status = RightsStatus.unknown
    store.save(state)
    assert runner_client.post("/api/jobs/vid123/render", json={}).status_code == 409
    assert runner_client.runner.queue.load() == []


def test_override_blocked_pipeline_records_reason_and_resumes(runner_client, store):
    queue = runner_client.runner.queue
    entry = queue.enqueue("source")
    queue.update(entry.entry_id, status="blocked", video_id="vid456")
    response = runner_client.post(
        "/api/jobs/vid456/render",
        json={"allow_red": True, "override_reason": "Reviewed source and accept risk"},
    )
    assert response.status_code == 202
    assert response.json["entry"]["options"]["allow_red"] is True
    assert response.json["entry"]["options"]["render_only"] is False
    assert store.load("vid456").rights_override_reason == "Reviewed source and accept risk"
    body = runner_client.get("/api/jobs/vid456").json
    assert body["rights_override"] is True
    assert body["rights_override_reason"] == "Reviewed source and accept risk"


def test_red_reclassification_revokes_prior_override(client, store):
    state = store.load("vid123")
    state.rights_override = True
    state.rights_override_reason = "Old permission"
    store.save(state)
    response = client.post(
        "/api/jobs/vid123/rights", json={"status": "red", "reason": "Permission was revoked"}
    )
    assert response.status_code == 200
    assert not store.load("vid123").is_exportable


def test_remember_local_source_does_not_change_state(client, store):
    before = (store.job_dir("vid456") / "state.json").read_bytes()
    assert (
        client.post(
            "/api/jobs/vid456/rights",
            json={"status": "green", "reason": "This is my own recording", "remember": True},
        ).status_code
        == 400
    )
    assert (store.job_dir("vid456") / "state.json").read_bytes() == before


def test_retry_render_resets_clip_stages_and_drops_previous_force(runner_client, store):
    from clipper.server.queue import RunOptions
    from clipper.state import CLIP_STAGES

    state = store.load("vid123")
    for clip in state.clips:
        for name in CLIP_STAGES:
            clip.stage(name).status = StageStatus.done
    store.save(state)
    queue = runner_client.runner.queue
    entry = queue.enqueue("source", RunOptions(force=True))
    queue.update(entry.entry_id, status="failed", video_id="vid123")
    response = runner_client.post(
        f"/api/queue/{entry.entry_id}/retry", json={"from_stage": "render"}
    )
    assert response.status_code == 202
    assert not queue.get(entry.entry_id).options.force
    state = store.load("vid123")
    assert state.is_done("ingest")
    assert all(
        clip.stage(name).status == StageStatus.pending
        for clip in state.clips
        for name in CLIP_STAGES
    )


def test_retry_old_entry_cannot_reset_job_owned_by_another_entry(runner_client, store):
    queue = runner_client.runner.queue
    old = queue.enqueue("old")
    queue.update(old.entry_id, status="failed", video_id="vid123")
    current = queue.enqueue("new")
    queue.update(current.entry_id, status="running", video_id="vid123")
    path = store.job_dir("vid123") / "state.json"
    before = path.read_bytes()
    assert (
        runner_client.post(
            f"/api/queue/{old.entry_id}/retry", json={"from_stage": "moments"}
        ).status_code
        == 409
    )
    assert path.read_bytes() == before
