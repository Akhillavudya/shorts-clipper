"""The review UI (Phase 4).

Nothing here starts a server: Flask's test client drives the same app object
``clipper review`` serves, and the store points at a temp directory. The two
things worth defending with tests are the ones a browser makes easy to get
wrong - that a verdict actually reaches ``state.json``, and that the media
route cannot be talked into serving something that is not a preview.
"""

from __future__ import annotations

import json

import pytest

from clipper.review import presenter
from clipper.review.app import create_app
from clipper.state import JobStore, RightsStatus
from clipper.transcribe.schema import Segment, Transcript, Word

#: The words the fixture transcript is built from; the clip spans index into it.
WORDS = [
    "the",
    "thing",
    "nobody",
    "tells",
    "you",
    "about",
    "index",
    "funds",
    "is",
    "that",
    "the",
    "boring",
    "one",
    "wins",
    "over",
    "thirty",
    "years",
    "and",
    "the",
    "exciting",
    "one",
    "does",
    "not",
]


@pytest.fixture
def store(tmp_path) -> JobStore:
    """A job with two scored clips, a transcript, and a source file on disk."""
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
        reason="States a counterintuitive point and resolves it.",
        rubric={"hook": 8.0, "payoff": 6.0, "quotability": 7.0},
        score=7.1,
        suggested_title="the boring fund wins",
        emphasis_words=["boring"],
        hook_rewrite="Nobody tells you this.",
    )
    state.add_clip(
        start_word_index=10,
        end_word_index=20,
        start=61.0,
        end=95.0,
        score=5.9,
        suggested_title="thirty years of nothing happening",
    )
    store.save(state)

    job_dir = store.job_dir("vid123")
    (job_dir / "source.mp4").write_bytes(b"\x00fake source mp4 payload")

    transcript = Transcript(
        words=[
            Word(w=word, start=float(i), end=float(i) + 0.4, speaker="SPEAKER_00")
            for i, word in enumerate(WORDS)
        ],
        segments=[Segment(start=0.0, end=10.0, speaker="SPEAKER_00", text=" ".join(WORDS))],
        backend="fake",
    )
    transcript.save(job_dir / "transcript.json")
    return store


@pytest.fixture
def client(store):
    app = create_app(store=store, weights={"weights": {"hook": 0.3, "payoff": 0.2}})
    app.config.update(TESTING=True)
    return app.test_client()


def _state(store):
    return store.load("vid123")


# -- pages ------------------------------------------------------------------


def test_index_lists_the_job(client):
    body = client.get("/").get_data(as_text=True)
    assert "An Episode" in body
    assert "vid123" in body
    assert "2 to do" in body


def test_job_page_shows_scores_titles_and_the_words(client):
    body = client.get("/jobs/vid123").get_data(as_text=True)
    assert "the boring fund wins" in body
    assert "7.10" in body
    # The transcript is the point of the page: a reviewer reads the clip before
    # deciding whether to spend a render on it.
    assert "index funds" in body
    assert "Nobody tells you this." in body
    # Weight shown next to the axis it weighted.
    assert "0.30" in body


def test_job_page_warns_when_the_rights_gate_would_refuse(client, store):
    state = _state(store)
    state.rights_status = RightsStatus.unknown
    store.save(state)

    body = client.get("/jobs/vid123").get_data(as_text=True)
    assert "render and publish will both refuse" in body


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


# -- verdicts ---------------------------------------------------------------


def test_approve_over_json_writes_state_and_returns_counts(client, store):
    response = client.post(
        "/jobs/vid123/clips/clip_01/review",
        json={"approved": True, "note": "  strong open  "},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["verdict"] == "approved"
    assert payload["counts"] == {"total": 2, "approved": 1, "rejected": 0, "pending": 1}

    clip = _state(store).clip("clip_01")
    assert clip.approved is True
    assert clip.review_note == "strong open"
    assert clip.reviewed_at is not None


def test_reject_over_a_plain_form_redirects_back_to_the_clip(client, store):
    response = client.post(
        "/jobs/vid123/clips/clip_02/review",
        data={"verdict": "reject", "note": "runs out of steam"},
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/jobs/vid123#clip_02")

    clip = _state(store).clip("clip_02")
    assert clip.approved is False
    assert clip.review_note == "runs out of steam"


def test_undeciding_keeps_the_note_and_the_timestamp(client, store):
    client.post("/jobs/vid123/clips/clip_01/review", json={"approved": False, "note": "maybe"})
    client.post("/jobs/vid123/clips/clip_01/review", json={"approved": None})

    clip = _state(store).clip("clip_01")
    # "Looked at and unsure" has to stay distinguishable from "never opened",
    # or the review queue cannot tell what is left to do.
    assert clip.approved is None
    assert clip.review_note == "maybe"
    assert clip.reviewed_at is not None


def test_an_emptied_note_clears_rather_than_storing_whitespace(client, store):
    client.post("/jobs/vid123/clips/clip_01/review", json={"approved": True, "note": "x"})
    client.post("/jobs/vid123/clips/clip_01/review", json={"approved": True, "note": "   "})
    assert _state(store).clip("clip_01").review_note is None


def test_a_verdict_does_not_disturb_the_rest_of_the_state(client, store):
    before = json.loads(store.state_path("vid123").read_text(encoding="utf-8"))
    client.post("/jobs/vid123/clips/clip_01/review", json={"approved": True})
    after = json.loads(store.state_path("vid123").read_text(encoding="utf-8"))

    assert after["clips"][1] == before["clips"][1]
    assert after["stages"] == before["stages"]
    assert after["rights_status"] == before["rights_status"]


def test_unknown_clip_is_404_and_bad_verdicts_are_400(client):
    def post(**kwargs) -> int:
        return client.post("/jobs/vid123/clips/clip_01/review", **kwargs).status_code

    missing = client.post("/jobs/vid123/clips/clip_99/review", json={"approved": True})
    assert missing.status_code == 404
    assert post(data={"verdict": "maybe"}) == 400
    assert post(json={"approved": "yes"}) == 400


# -- media ------------------------------------------------------------------


def test_source_is_served_with_range_support(client):
    plain = client.get("/media/vid123/source.mp4")
    assert plain.status_code == 200
    assert plain.headers["Accept-Ranges"] == "bytes"

    # Without ranges the browser would pull a whole episode to play 40 seconds
    # of it, which is why the source fallback is usable at all.
    partial = client.get("/media/vid123/source.mp4", headers={"Range": "bytes=0-3"})
    assert partial.status_code == 206
    assert partial.get_data() == b"\x00fak"


def test_media_refuses_non_media_and_anything_outside_the_job(client, store):
    other = store.create("other")
    store.save(other)
    (store.job_dir("other") / "source.mp4").write_bytes(b"someone else's episode")
    (store.root / "secret.mp4").write_bytes(b"not in any job")

    # The state file is not a preview, and neither is anything else the route
    # could be talked into reaching from a job directory.
    assert client.get("/media/vid123/state.json").status_code == 404
    assert client.get("/media/vid123/../secret.mp4").status_code == 404
    assert client.get("/media/vid123/..%2fother%2fsource.mp4").status_code == 404
    assert client.get("/media/vid123/clips/../../other/source.mp4").status_code == 404
    assert client.get("/media/nope/source.mp4").status_code == 404
    # Not rendered yet is a 404 too, not a 500.
    assert client.get("/media/vid123/clips/clip_01/final.mp4").status_code == 404


def test_media_serves_a_rendered_clip_once_it_exists(client, store):
    clip_dir = store.clip_dir("vid123", "clip_01")
    clip_dir.mkdir(parents=True)
    (clip_dir / "final.mp4").write_bytes(b"rendered")

    response = client.get("/media/vid123/clips/clip_01/final.mp4")
    assert response.status_code == 200
    assert response.get_data() == b"rendered"


# -- presenter --------------------------------------------------------------


def test_preview_prefers_the_finished_render_over_the_source(store):
    view = presenter.build_job(store, "vid123")
    # Nothing rendered yet: review happens before render, so this is the norm.
    assert view.clips[0].preview.kind == "source"
    assert view.clips[0].preview.start == pytest.approx(12.5)
    assert view.clips[0].preview.end == pytest.approx(52.0)

    clip_dir = store.clip_dir("vid123", "clip_01")
    clip_dir.mkdir(parents=True)
    (clip_dir / "cut.mp4").write_bytes(b"cut")
    (clip_dir / "final.mp4").write_bytes(b"final")

    view = presenter.build_job(store, "vid123")
    preview = view.clips[0].preview
    assert preview.kind == "final"
    assert preview.path == "clips/clip_01/final.mp4"
    assert preview.start == 0.0


def test_a_job_without_a_transcript_still_renders_its_scores(store):
    presenter.transcript_path(store.job_dir("vid123")).unlink()
    view = presenter.build_job(store, "vid123")

    assert view.clips[0].transcript == ""
    assert view.clips[0].score == pytest.approx(7.1)
    assert any("No transcript" in w for w in view.warnings)


def test_rubric_axes_come_back_in_the_canonical_order(store):
    view = presenter.build_job(store, "vid123", weights={"weights": {"hook": 0.3}})
    axes = [a.name for a in view.clips[0].rubric]
    assert axes == ["hook", "payoff", "quotability"]
    assert view.clips[0].rubric[0].weight == pytest.approx(0.3)
    assert view.clips[0].rubric[1].weight is None


def test_source_link_seeks_to_the_cut_point(store):
    view = presenter.build_job(store, "vid123")
    assert view.clips[0].source_link == "https://www.youtube.com/watch?v=vid123&t=12s"


def test_counts_and_index_ordering(store):
    presenter.apply_review(store, "vid123", "clip_01", approved=True)
    view = presenter.build_job(store, "vid123")
    assert (view.counts.approved, view.counts.rejected, view.counts.pending) == (1, 0, 1)

    summaries = presenter.build_index(store)
    assert [s.video_id for s in summaries] == ["vid123"]
    assert summaries[0].counts.approved == 1
    assert summaries[0].stage == "ingest"


def test_apply_review_rejects_an_unknown_clip(store):
    with pytest.raises(presenter.ClipNotFound):
        presenter.apply_review(store, "vid123", "clip_44", approved=True)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "-"), (0, "0:00"), (9.4, "0:09"), (95, "1:35"), (3725, "1:02:05")],
)
def test_timecode(seconds, expected):
    assert presenter.timecode(seconds) == expected


# -- the single-writer discipline (WEB_UI_PLAN.md section 3) -----------------


def _claim(store, video_id: str) -> str:
    """Put ``video_id`` in the queue as the job a worker is currently writing."""
    from clipper.server.queue import QueueStore

    queue = QueueStore(store.root)
    entry = queue.enqueue("https://youtu.be/vid123")
    queue.update(entry.entry_id, status="running", video_id=video_id, pid=4242)
    return entry.entry_id


def test_a_verdict_on_a_running_job_is_refused_with_409(client, store):
    """And - the assertion that matters - the state file is byte-identical.

    ``apply_review`` is load-mutate-save and the worker holds the same
    ``JobState`` for the whole of a multi-minute stage, so a verdict written
    now is one the next ``store.save`` silently discards, or one that discards
    the stage's work. There is no lock, and section 3 explains why a lock would
    not have helped: the read-modify-write window *is* the whole stage.
    """
    _claim(store, "vid123")
    path = store.job_dir("vid123") / "state.json"
    before = path.read_bytes()

    response = client.post("/jobs/vid123/clips/clip_01/review", json={"approved": True})

    assert response.status_code == 409
    assert response.get_json()["writable"] is False
    assert path.read_bytes() == before


def test_the_409_names_the_job_so_the_page_can_explain_itself(client, store):
    _claim(store, "vid123")

    body = client.post("/jobs/vid123/clips/clip_01/review", json={"approved": True}).get_json()

    assert body["job_id"] == "vid123"
    assert body["clip_id"] == "clip_01"
    assert "verdicts are refused" in body["error"]


def test_a_form_post_is_refused_the_same_way(client, store):
    """The Jinja page posts a form, and a 302 back to an unchanged page would
    read as "it worked"."""
    _claim(store, "vid123")

    response = client.post("/jobs/vid123/clips/clip_01/review", data={"verdict": "approve"})

    assert response.status_code == 409


def test_a_verdict_on_a_job_nobody_owns_still_lands(client, store):
    """The 409 is the backstop for a stale tab, not the normal path.

    Every job payload carries ``writable``, so the SPA greys the buttons
    instead of letting you discover this as an error.
    """
    _claim(store, "some-other-job")

    response = client.post("/jobs/vid123/clips/clip_01/review", json={"approved": True})

    assert response.status_code == 200
    assert store.load("vid123").clip("clip_01").approved is True


def test_a_finished_entry_does_not_hold_the_file(client, store):
    from clipper.server.queue import QueueStore

    entry_id = _claim(store, "vid123")
    QueueStore(store.root).update(entry_id, status="done")

    response = client.post("/jobs/vid123/clips/clip_01/review", json={"approved": True})

    assert response.status_code == 200
