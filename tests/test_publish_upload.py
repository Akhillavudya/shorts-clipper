"""``videos.insert`` - the resumable upload and the guards around it.

Nothing here touches a network: every request goes through an
``httpx.MockTransport`` that implements enough of the resumable protocol to be
lied to convincingly - 308s with a ``Range`` header, a dead session, a 503 that
clears on retry.

The tests lean on the parts that cost something when they are wrong: that a
failure mid-transfer re-asks the server for the offset instead of assuming, that
a clip already carrying a video id is never uploaded twice, that the rights gate
fires before any byte moves, and that the quota is charged against the same
ledger discovery spends from.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from clipper import pipeline
from clipper.publish import upload as upload_module
from clipper.publish.upload import (
    CHUNK_MULTIPLE,
    ResumableUpload,
    Session,
    SessionExpired,
    UploadError,
    insert_body,
    load_session,
    normalize_chunk_size,
    read_upload_json,
    save_session,
)
from clipper.state import JobState, JobStore, RightsStatus

SESSION_URI = "https://upload.test/session/abc"


def body() -> dict:
    return {
        "snippet": {"title": "A clip #Shorts", "description": "d", "categoryId": "22"},
        "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False},
    }


def uploader(handler, **kwargs) -> ResumableUpload:
    """An uploader wired to a mock transport, with sleeps recorded not taken."""
    slept: list[float] = []
    up = ResumableUpload(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=slept.append,
        **kwargs,
    )
    up.slept = slept  # type: ignore[attr-defined]
    return up


def written(tmp_path, size: int, name: str = "final.mp4"):
    path = tmp_path / name
    path.write_bytes(bytes(range(256)) * (size // 256) + b"\0" * (size % 256))
    return path


# --------------------------------------------------------------------------
# the request body
# --------------------------------------------------------------------------


class TestInsertBody:
    def test_only_snippet_and_status_are_sent(self) -> None:
        raw = {
            "clip_id": "clip_01",
            "snippet": {"title": "t"},
            "status": {"privacyStatus": "private"},
            "rights_status": "green",
            "source_url": "https://youtu.be/x",
        }
        assert set(insert_body(raw)) == {"snippet", "status"}

    def test_made_for_kids_is_declared_because_insert_requires_it(self) -> None:
        sent = insert_body({"snippet": {"title": "t"}, "status": {}})
        assert sent["status"]["selfDeclaredMadeForKids"] is False

    def test_an_untitled_video_is_refused_before_any_bytes_move(self) -> None:
        with pytest.raises(UploadError, match="no title"):
            insert_body({"snippet": {}, "status": {}})

    def test_a_metadata_object_is_accepted_too(self) -> None:
        from clipper.publish.metadata import UploadMetadata

        meta = UploadMetadata(clip_id="clip_01", video_id="v", title="t", description="d")
        assert insert_body(meta)["snippet"]["title"] == "t"


class TestReadUploadJson:
    def test_a_missing_file_names_the_command_that_writes_it(self, tmp_path) -> None:
        with pytest.raises(UploadError, match="clipper publish"):
            read_upload_json(tmp_path)

    def test_corrupt_json_is_reported_not_swallowed(self, tmp_path) -> None:
        (tmp_path / "upload.json").write_text("{nope", encoding="utf-8")
        with pytest.raises(UploadError, match="not valid JSON"):
            read_upload_json(tmp_path)


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


class TestChunkSize:
    def test_rounded_down_to_the_protocol_multiple(self) -> None:
        assert normalize_chunk_size(CHUNK_MULTIPLE + 5) == CHUNK_MULTIPLE
        assert normalize_chunk_size(3 * CHUNK_MULTIPLE) == 3 * CHUNK_MULTIPLE

    def test_never_zero(self) -> None:
        # A chunk size of zero reads no bytes and loops forever.
        assert normalize_chunk_size(0) == CHUNK_MULTIPLE


# --------------------------------------------------------------------------
# opening the session
# --------------------------------------------------------------------------


class TestStart:
    def test_the_location_header_is_the_session(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-upload-content-length"] == "1234"
            assert request.url.params["uploadType"] == "resumable"
            assert request.url.params["part"] == "snippet,status"
            return httpx.Response(200, headers={"Location": SESSION_URI})

        assert uploader(handler).start(body(), file_size=1234) == SESSION_URI

    def test_subscribers_are_not_notified_unless_asked(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["notify"] = request.url.params["notifySubscribers"]
            return httpx.Response(200, headers={"Location": SESSION_URI})

        uploader(handler).start(body(), file_size=10)
        assert seen["notify"] == "false"

    def test_a_session_without_a_location_is_an_error_not_a_none(self) -> None:
        with pytest.raises(UploadError, match="nowhere to send"):
            uploader(lambda r: httpx.Response(200)).start(body(), file_size=10)

    def test_a_503_is_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503, text="high demand")
            return httpx.Response(200, headers={"Location": SESSION_URI})

        up = uploader(handler)
        assert up.start(body(), file_size=10) == SESSION_URI
        assert calls["n"] == 3
        assert up.slept  # it waited rather than hammering

    def test_a_403_is_not_retried_and_explains_the_quota(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(403, text='{"error":{"errors":[{"reason":"quotaExceeded"}]}}')

        with pytest.raises(UploadError, match="1600 of 10,000"):
            uploader(handler).start(body(), file_size=10)
        assert calls["n"] == 1

    def test_a_rejected_title_says_so(self) -> None:
        with pytest.raises(UploadError, match="title, description and categoryId"):
            uploader(lambda r: httpx.Response(400, text="bad title")).start(body(), file_size=10)


# --------------------------------------------------------------------------
# sending the bytes
# --------------------------------------------------------------------------


class TestSend:
    def test_a_small_file_goes_in_one_chunk(self, tmp_path) -> None:
        path = written(tmp_path, 1000)
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["range"] = request.headers["content-range"]
            seen["len"] = str(len(request.content))
            return httpx.Response(200, json={"id": "yt123"})

        result = uploader(handler).send(path, SESSION_URI)
        assert result["id"] == "yt123"
        assert seen["range"] == "bytes 0-999/1000"
        assert seen["len"] == "1000"

    def test_chunks_follow_the_offset_the_server_reports(self, tmp_path) -> None:
        size = 3 * CHUNK_MULTIPLE
        path = written(tmp_path, size)
        ranges: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            ranges.append(request.headers["content-range"])
            end = int(request.headers["content-range"].split("-")[1].split("/")[0])
            if end + 1 >= size:
                return httpx.Response(200, json={"id": "yt123"})
            return httpx.Response(308, headers={"Range": f"bytes=0-{end}"})

        uploader(handler, chunk_bytes=CHUNK_MULTIPLE).send(path, SESSION_URI)
        assert ranges == [
            f"bytes 0-{CHUNK_MULTIPLE - 1}/{size}",
            f"bytes {CHUNK_MULTIPLE}-{2 * CHUNK_MULTIPLE - 1}/{size}",
            f"bytes {2 * CHUNK_MULTIPLE}-{size - 1}/{size}",
        ]

    def test_a_partial_308_resends_only_what_the_server_missed(self, tmp_path) -> None:
        # The server acknowledges half of the first chunk. Anything that
        # assumed the whole chunk landed would skip those bytes silently.
        size = 2 * CHUNK_MULTIPLE
        path = written(tmp_path, size)
        ranges: list[str] = []
        half = CHUNK_MULTIPLE // 2

        def handler(request: httpx.Request) -> httpx.Response:
            header = request.headers["content-range"]
            ranges.append(header)
            if len(ranges) == 1:
                return httpx.Response(308, headers={"Range": f"bytes=0-{half - 1}"})
            return httpx.Response(200, json={"id": "yt123"})

        uploader(handler, chunk_bytes=CHUNK_MULTIPLE).send(path, SESSION_URI)
        assert ranges[1].startswith(f"bytes {half}-")

    def test_a_308_with_no_range_header_means_start_over(self, tmp_path) -> None:
        path = written(tmp_path, 1000)
        ranges: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            ranges.append(request.headers["content-range"])
            if len(ranges) == 1:
                return httpx.Response(308)  # stored nothing
            return httpx.Response(200, json={"id": "yt123"})

        uploader(handler).send(path, SESSION_URI)
        assert ranges == ["bytes 0-999/1000", "bytes 0-999/1000"]

    def test_a_failure_re_asks_for_the_offset_rather_than_assuming(self, tmp_path) -> None:
        size = 2 * CHUNK_MULTIPLE
        path = written(tmp_path, size)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            header = request.headers["content-range"]
            calls.append(header)
            if len(calls) == 1:
                return httpx.Response(503, text="unavailable")
            if header == f"bytes */{size}":
                # The status query: the first chunk did land after all.
                return httpx.Response(308, headers={"Range": f"bytes=0-{CHUNK_MULTIPLE - 1}"})
            return httpx.Response(200, json={"id": "yt123"})

        uploader(handler, chunk_bytes=CHUNK_MULTIPLE).send(path, SESSION_URI)
        assert calls[1] == f"bytes */{size}"
        assert calls[2].startswith(f"bytes {CHUNK_MULTIPLE}-")

    def test_a_dead_session_raises_so_the_caller_can_restart(self, tmp_path) -> None:
        path = written(tmp_path, 100)
        with pytest.raises(SessionExpired):
            uploader(lambda r: httpx.Response(404, text="gone")).send(path, SESSION_URI)

    def test_progress_is_reported_against_the_real_size(self, tmp_path) -> None:
        size = 2 * CHUNK_MULTIPLE
        path = written(tmp_path, size)
        seen: list[tuple[int, int]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            end = int(request.headers["content-range"].split("-")[1].split("/")[0])
            if end + 1 >= size:
                return httpx.Response(200, json={"id": "yt123"})
            return httpx.Response(308, headers={"Range": f"bytes=0-{end}"})

        uploader(handler, chunk_bytes=CHUNK_MULTIPLE).send(
            path, SESSION_URI, on_progress=lambda sent, total: seen.append((sent, total))
        )
        assert seen[-1] == (size, size)
        assert all(total == size for _, total in seen)

    def test_starting_past_the_end_settles_it_with_a_status_query(self, tmp_path) -> None:
        # What a run that died between the last chunk and the state write looks
        # like: nothing left to send, but the video id has never been seen.
        path = written(tmp_path, 1000)

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["content-range"] == "bytes */1000"
            return httpx.Response(201, json={"id": "yt123"})

        assert uploader(handler).send(path, SESSION_URI, start_at=1000)["id"] == "yt123"

    def test_a_finished_upload_with_no_id_is_an_error(self, tmp_path) -> None:
        path = written(tmp_path, 100)
        with pytest.raises(UploadError, match="no video id"):
            uploader(lambda r: httpx.Response(200, json={"kind": "youtube#video"})).send(
                path, SESSION_URI
            )


class TestOffsetQuery:
    def test_reports_the_next_byte_to_send(self) -> None:
        up = uploader(lambda r: httpx.Response(308, headers={"Range": "bytes=0-524287"}))
        assert up.offset(SESSION_URI, 1_000_000) == 524288

    def test_a_completed_upload_comes_back_as_the_video(self) -> None:
        up = uploader(lambda r: httpx.Response(200, json={"id": "yt123"}))
        assert up.offset(SESSION_URI, 100) == {"id": "yt123"}

    def test_410_is_an_expired_session(self) -> None:
        up = uploader(lambda r: httpx.Response(410))
        with pytest.raises(SessionExpired):
            up.offset(SESSION_URI, 100)


# --------------------------------------------------------------------------
# the session file
# --------------------------------------------------------------------------


class TestSessionFile:
    def test_round_trips(self, tmp_path) -> None:
        session = Session(uri=SESSION_URI, file_size=10, file_digest="abc", started_at=100.0)
        save_session(tmp_path, session)
        assert load_session(tmp_path) == session

    def test_a_re_rendered_file_invalidates_the_session(self, tmp_path) -> None:
        path = written(tmp_path, 1000)
        session = Session.opening(SESSION_URI, path, now=100.0)
        assert session.matches(path, now=101.0)

        # Same length, different bytes - what a second render of the same clip
        # produces. Resuming here would splice two encodes together, and an
        # mtime is too weak a guard for a mistake that silent.
        path.write_bytes(b"\xff" * 1000)
        assert not session.matches(path, now=101.0)

    def test_a_shorter_file_invalidates_the_session_without_hashing_it(self, tmp_path) -> None:
        path = written(tmp_path, 1000)
        session = Session.opening(SESSION_URI, path, now=100.0)
        path.write_bytes(b"padding." * 50)
        assert not session.matches(path, now=101.0)

    def test_an_old_session_is_abandoned_before_google_expires_it(self, tmp_path) -> None:
        path = written(tmp_path, 100)
        session = Session.opening(SESSION_URI, path, now=0.0)
        assert not session.matches(path, now=upload_module.SESSION_TTL_S + 1)

    def test_a_corrupt_session_file_is_dropped_not_raised(self, tmp_path) -> None:
        upload_module.session_path(tmp_path).write_text("{", encoding="utf-8")
        assert load_session(tmp_path) is None
        assert not upload_module.session_path(tmp_path).exists()


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    from clipper.config import Settings

    settings = Settings(
        data={
            "publish": {
                "privacy_status": "private",
                "upload": {"daily_quota_ceiling": 8000, "chunk_mb": 1},
            }
        },
        path=tmp_path / "settings.yaml",
    )
    root = tmp_path / "jobs"
    return pipeline.Context(settings=settings, store=JobStore(root), jobs_root=root)


def build_job(ctx, *, rights=RightsStatus.green, size: int = 1000, with_metadata: bool = True):
    state = JobState(video_id="vid1", title="Episode", channel_title="Someone")
    state.rights_status = rights
    clip = state.add_clip(suggested_title="A clip", score=8.0)
    ctx.store.save(state)

    clip_dir = ctx.store.clip_dir("vid1", clip.clip_id)
    clip_dir.mkdir(parents=True, exist_ok=True)
    written(clip_dir, size)
    if with_metadata:
        (clip_dir / "upload.json").write_text(
            json.dumps(
                {
                    "snippet": {"title": "A clip #Shorts", "categoryId": "22"},
                    "status": {"privacyStatus": "private"},
                }
            ),
            encoding="utf-8",
        )
    return state, clip


class FakeLedger:
    def __init__(self) -> None:
        self.spent: list[tuple[int, str]] = []

    def spend(self, units: int, *, resource: str = "") -> None:
        self.spent.append((units, resource))


class TestUploadStage:
    def test_records_the_video_id_and_clears_the_session(self, ctx) -> None:
        state, clip = build_job(ctx)
        ledger = FakeLedger()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": SESSION_URI})
            return httpx.Response(200, json={"id": "yt999", "status": {"privacyStatus": "private"}})

        result = pipeline.upload(
            ctx, "vid1", clip_index=clip.index, uploader=uploader(handler), ledger=ledger
        )

        assert result.video_id == "yt999"
        assert result.watch_url.endswith("yt999")
        assert ledger.spent == [(1600, "videos.insert")]

        reloaded = ctx.store.load("vid1")
        saved = reloaded.clips[0]
        assert saved.youtube_video_id == "yt999"
        assert saved.published_at is not None
        assert saved.stage("upload").status.value == "done"
        assert saved.stage("upload").detail["youtube_video_id"] == "yt999"
        assert not upload_module.session_path(ctx.store.clip_dir("vid1", clip.clip_id)).exists()

    def test_the_rights_gate_fires_before_anything_is_sent(self, ctx) -> None:
        from clipper.state import RightsBlocked

        state, clip = build_job(ctx, rights=RightsStatus.unknown)

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("the gate let a request through")

        with pytest.raises(RightsBlocked):
            pipeline.upload(
                ctx, "vid1", clip_index=clip.index, uploader=uploader(handler), ledger=FakeLedger()
            )

    def test_an_override_is_recorded_on_the_state(self, ctx) -> None:
        state, clip = build_job(ctx, rights=RightsStatus.unknown)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": SESSION_URI})
            return httpx.Response(200, json={"id": "yt999"})

        pipeline.upload(
            ctx,
            "vid1",
            clip_index=clip.index,
            allow_red=True,
            override_reason="own footage",
            uploader=uploader(handler),
            ledger=FakeLedger(),
        )
        reloaded = ctx.store.load("vid1")
        assert reloaded.rights_override is True
        assert reloaded.rights_override_reason == "own footage"

    def test_a_published_clip_is_not_uploaded_twice(self, ctx) -> None:
        state, clip = build_job(ctx)
        clip.youtube_video_id = "already"
        ctx.store.save(state)

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("uploaded a second copy")

        with pytest.raises(pipeline.StageError, match="already published"):
            pipeline.upload(
                ctx, "vid1", clip_index=clip.index, uploader=uploader(handler), ledger=FakeLedger()
            )

    def test_force_overrides_the_duplicate_guard(self, ctx) -> None:
        state, clip = build_job(ctx)
        clip.youtube_video_id = "already"
        ctx.store.save(state)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": SESSION_URI})
            return httpx.Response(200, json={"id": "yt_second"})

        result = pipeline.upload(
            ctx,
            "vid1",
            clip_index=clip.index,
            force=True,
            uploader=uploader(handler),
            ledger=FakeLedger(),
        )
        assert result.video_id == "yt_second"

    def test_a_missing_render_is_refused_before_quota_is_charged(self, ctx) -> None:
        state, clip = build_job(ctx)
        (ctx.store.clip_dir("vid1", clip.clip_id) / "final.mp4").unlink()
        ledger = FakeLedger()

        with pytest.raises(pipeline.StageError, match="clipper render"):
            pipeline.upload(ctx, "vid1", clip_index=clip.index, ledger=ledger)
        assert ledger.spent == []

    def test_an_empty_render_is_refused(self, ctx) -> None:
        state, clip = build_job(ctx)
        (ctx.store.clip_dir("vid1", clip.clip_id) / "final.mp4").write_bytes(b"")

        with pytest.raises(pipeline.StageError, match="empty"):
            pipeline.upload(ctx, "vid1", clip_index=clip.index, ledger=FakeLedger())

    def test_privacy_can_be_overridden_for_one_upload(self, ctx) -> None:
        state, clip = build_job(ctx)
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                seen["privacy"] = json.loads(request.content)["status"]["privacyStatus"]
                return httpx.Response(200, headers={"Location": SESSION_URI})
            return httpx.Response(200, json={"id": "yt999"})

        pipeline.upload(
            ctx,
            "vid1",
            clip_index=clip.index,
            privacy="unlisted",
            uploader=uploader(handler),
            ledger=FakeLedger(),
        )
        assert seen["privacy"] == "unlisted"

    def test_metadata_is_built_when_publish_has_not_run(self, ctx, tmp_path) -> None:
        state, clip = build_job(ctx, with_metadata=False)
        watchlist = tmp_path / "watchlist.yaml"
        watchlist.write_text("channels: []\n", encoding="utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                assert "A clip" in json.loads(request.content)["snippet"]["title"]
                return httpx.Response(200, headers={"Location": SESSION_URI})
            return httpx.Response(200, json={"id": "yt999"})

        pipeline.upload(
            ctx,
            "vid1",
            clip_index=clip.index,
            watchlist_path=watchlist,
            uploader=uploader(handler),
            ledger=FakeLedger(),
        )
        assert (ctx.store.clip_dir("vid1", clip.clip_id) / "upload.json").is_file()

    def test_an_interrupted_upload_resumes_into_the_same_session(self, ctx) -> None:
        state, clip = build_job(ctx, size=4000)
        clip_dir = ctx.store.clip_dir("vid1", clip.clip_id)
        save_session(
            clip_dir,
            Session.opening(SESSION_URI, clip_dir / "final.mp4", now=time.time()),
        )
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":  # pragma: no cover
                raise AssertionError("opened a second session instead of resuming")
            header = request.headers["content-range"]
            seen.append(header)
            if header.startswith("bytes */"):
                return httpx.Response(308, headers={"Range": "bytes=0-1999"})
            return httpx.Response(200, json={"id": "yt999"})

        result = pipeline.upload(
            ctx, "vid1", clip_index=clip.index, uploader=uploader(handler), ledger=FakeLedger()
        )
        assert result.resumed_from == 2000
        assert result.bytes_sent == 2000
        assert seen[1] == "bytes 2000-3999/4000"

    def test_an_expired_session_restarts_once(self, ctx) -> None:
        state, clip = build_job(ctx, size=1000)
        clip_dir = ctx.store.clip_dir("vid1", clip.clip_id)
        save_session(
            clip_dir,
            Session.opening(SESSION_URI, clip_dir / "final.mp4", now=time.time()),
        )
        posts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                posts["n"] += 1
                return httpx.Response(200, headers={"Location": SESSION_URI + "/new"})
            if str(request.url).endswith("/new"):
                return httpx.Response(200, json={"id": "yt999"})
            return httpx.Response(410, text="gone")

        result = pipeline.upload(
            ctx, "vid1", clip_index=clip.index, uploader=uploader(handler), ledger=FakeLedger()
        )
        assert result.video_id == "yt999"
        assert posts["n"] == 1

    def test_a_failed_upload_leaves_the_stage_failed_and_the_session_behind(self, ctx) -> None:
        state, clip = build_job(ctx)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": SESSION_URI})
            return httpx.Response(400, text="nope")

        with pytest.raises(UploadError):
            pipeline.upload(
                ctx, "vid1", clip_index=clip.index, uploader=uploader(handler), ledger=FakeLedger()
            )

        reloaded = ctx.store.load("vid1")
        assert reloaded.clips[0].stage("upload").status.value == "failed"
        assert reloaded.clips[0].youtube_video_id is None
        # The session survives so the retry resumes rather than re-sending.
        assert upload_module.session_path(ctx.store.clip_dir("vid1", clip.clip_id)).exists()

    def test_quota_exhaustion_says_it_is_shared_with_discovery(self, ctx) -> None:
        from clipper.discover.youtube import QuotaExceeded

        state, clip = build_job(ctx)

        class FullLedger:
            def spend(self, units: int, *, resource: str = "") -> None:
                raise QuotaExceeded("spent 8000 of 8000")

        with pytest.raises(pipeline.StageError, match="shared with discovery"):
            pipeline.upload(ctx, "vid1", clip_index=clip.index, ledger=FullLedger())

    def test_an_unknown_clip_lists_what_there_is(self, ctx) -> None:
        build_job(ctx)
        with pytest.raises(pipeline.StageError, match="have: 1"):
            pipeline.upload(ctx, "vid1", clip_index=9, ledger=FakeLedger())
