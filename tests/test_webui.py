"""Serving the built SPA (WEB_UI_PLAN.md sections 6 and 8).

Shaped like ``tests/test_review.py``: no server is started, ``create_app()``
plus ``app.test_client()`` drives the same app object ``clipper ui`` serves.

Nothing here runs Vite. The bundle is committed, so its *existence* is a fact
about the repository rather than something a test should produce; what these
tests own is the Flask half - who answers ``/``, what the catch-all must never
swallow, and what a missing bundle says instead of raising.
"""

from __future__ import annotations

import pytest

from clipper import webui
from clipper.review.app import create_app
from clipper.state import JobStore, RightsStatus


@pytest.fixture
def store(tmp_path) -> JobStore:
    """One job, enough for the Jinja pages to render under ``/legacy/``."""
    store = JobStore(tmp_path / "jobs")
    store.create(
        "vid123",
        source_url="https://www.youtube.com/watch?v=vid123",
        title="An Episode",
        channel_title="A Channel",
        duration_s=3600.0,
        rights_status=RightsStatus.green,
        rights_note="CC BY.",
    )
    return store


@pytest.fixture
def dist(tmp_path, monkeypatch):
    """A stand-in for a built bundle: an ``index.html`` and one asset.

    Deliberately not the real ``src/clipper/webui/dist``, and pushed in by
    patching ``webui.DIST`` rather than by passing it: ``create_app`` calls
    ``register()`` with no arguments, so patching the module attribute is what
    lets these tests drive the exact wiring ``clipper ui`` gets. A test that
    asserted on the committed bundle's bytes would fail on every frontend
    change for no reason - these tests are about routing, not about what Vite
    emitted.
    """
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>clipper</title>", encoding="utf-8")
    (root / "assets" / "index.js").write_text("console.log('spa')", encoding="utf-8")
    # A file one level above the bundle, for the traversal test to try to
    # reach. If it is ever served, the guard is not doing its job.
    (tmp_path / "secret.txt").write_text("not part of the UI", encoding="utf-8")
    monkeypatch.setattr(webui, "DIST", root)
    return root


@pytest.fixture
def spa(store, dist):
    """The app ``clipper ui`` serves: SPA at ``/``, Jinja pages at ``/legacy/``."""
    return create_app(store=store, weights={}, ui="spa")


@pytest.fixture
def client(spa):
    return spa.test_client()


# -- is_built ------------------------------------------------------------


def test_is_built_wants_index_not_just_a_directory(tmp_path):
    """``--emptyOutDir`` can leave the directory behind when a build fails."""
    empty = tmp_path / "dist"
    empty.mkdir()

    assert not webui.is_built(empty)

    (empty / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert webui.is_built(empty)


def test_is_built_on_a_missing_directory(tmp_path):
    assert not webui.is_built(tmp_path / "never-built")


def test_the_committed_bundle_is_present():
    """The bundle is committed on purpose - `clipper ui` must need only Python.

    This is the test that fails when someone adds ``webui/dist`` to
    ``.gitignore``, or commits ``frontend/`` without rebuilding.
    """
    assert webui.is_built(), (
        f"{webui.DIST} has no index.html. Run `cd frontend && npm install && npm run build`; "
        f"the bundle is committed with its sources, in the same commit."
    )


# -- the catch-all -------------------------------------------------------


def test_root_serves_the_bundle(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"<!doctype html>" in response.data.lower()


def test_a_deep_link_serves_the_bundle(client):
    """`/jobs/vid123` typed into the address bar is a request to Flask.

    The SPA uses real URLs rather than a hash router, so the router only gets
    to see that path if Flask hands back ``index.html`` for it first.
    """
    response = client.get("/jobs/vid123")

    assert response.status_code == 200
    assert b"<title>clipper</title>" in response.data


def test_assets_are_served(client):
    response = client.get("/assets/index.js")

    assert response.status_code == 200
    assert b"console.log" in response.data


def test_a_missing_asset_is_a_404_not_the_index(client):
    """Otherwise a mistyped script tag downloads the HTML page as JavaScript
    and the failure shows up as a syntax error three layers away."""
    response = client.get("/assets/nope.js")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/assets/../../secret.txt",
        "/assets/..%2f..%2fsecret.txt",
        # Backslashes are the Windows-specific one, and the only form here that
        # Werkzeug does *not* normalise away before routing: it reaches the
        # view as a literal segment, and ``Path`` on Windows treats it as a
        # separator. Without the containment check this would be served.
        "/assets/..%5c..%5csecret.txt",
    ],
)
def test_asset_traversal_is_refused(client, path):
    """Same guard as the media route: resolve, then check containment."""
    response = client.get(path)

    assert response.status_code == 404
    assert b"not part of the UI" not in response.data


# -- the ordering rule ---------------------------------------------------


def test_the_catch_all_does_not_shadow_the_api(client):
    """The one rule ``register`` exists to keep.

    Registered before the blueprint, ``/<path:relpath>`` matches ``/api/jobs``
    and the whole API disappears behind an HTML page. ``create_app`` calls
    ``webui.register`` last; this is the assertion that says why.
    """
    response = client.get("/api/jobs")

    assert response.status_code == 200
    assert response.is_json
    assert [job["video_id"] for job in response.json["jobs"]] == ["vid123"]


def test_the_catch_all_does_not_shadow_media(client, store):
    (store.root / "vid123" / "clips").mkdir(parents=True)
    target = store.root / "vid123" / "clips" / "final.mp4"
    target.write_bytes(b"\x00\x01\x02")

    response = client.get("/media/vid123/clips/final.mp4")

    assert response.status_code == 200
    assert response.data == b"\x00\x01\x02"


def test_the_catch_all_does_not_shadow_healthz(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json["ok"] is True


@pytest.mark.parametrize("path", ["/api/jbos", "/media/vid123", "/healthz/nope"])
def test_a_reserved_prefix_that_matches_nothing_is_json_not_html(client, path):
    """A 404 under ``/api/`` must stay JSON.

    Handing ``index.html`` to a ``fetch()`` turns a typo into "Unexpected token
    '<'" inside the SPA, which is three steps removed from the actual mistake.
    """
    response = client.get(path)

    assert response.status_code == 404
    assert response.is_json


# -- a missing bundle ----------------------------------------------------


def test_an_unbuilt_bundle_explains_itself(store, tmp_path, monkeypatch):
    """Instructions, not a traceback and not a 404.

    A 404 at ``/`` reads as a broken install; a stack trace reads as a broken
    program. What is actually true is that two commands have not been run.
    """
    monkeypatch.setattr(webui, "DIST", tmp_path / "never-built")
    app = create_app(store=store, weights={}, ui="spa")

    response = app.test_client().get("/")

    assert response.status_code == 200
    body = response.data.decode("utf-8")
    assert "npm run build" in body
    assert "/legacy/" in body


def test_an_unbuilt_bundle_leaves_the_api_working(store, tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "DIST", tmp_path / "never-built")
    app = create_app(store=store, weights={}, ui="spa")

    response = app.test_client().get("/api/jobs")

    assert response.status_code == 200
    assert response.is_json


# -- the ui= switch ------------------------------------------------------


def test_jinja_mode_keeps_the_pages_at_the_root(store):
    """``clipper review`` is unchanged by any of this."""
    client = create_app(store=store, weights={}, ui="jinja").test_client()

    assert client.get("/").status_code == 200
    assert client.get("/jobs/vid123").status_code == 200
    # Nothing SPA-shaped is mounted, so an unknown path is a real 404 rather
    # than a page that renders an empty React root.
    assert client.get("/does-not-exist").status_code == 404


def test_spa_mode_moves_the_pages_to_legacy(spa):
    client = spa.test_client()

    assert client.get("/legacy/").status_code == 200
    assert client.get("/legacy/jobs/vid123").status_code == 200


def test_spa_mode_gives_the_root_to_the_bundle(spa):
    """Both UIs cannot own ``/``, and the SPA is the one that does here."""
    response = spa.test_client().get("/")

    assert response.status_code == 200
    assert b"<title>clipper</title>" in response.data


def test_spa_mode_keeps_the_review_post_pointing_at_its_own_page(spa, store):
    """The endpoint names do not move, only the URLs.

    ``url_for("job", ...)`` is what the review POST redirects to after a form
    submit, so a prefix that broke it would send every non-JSON verdict to a
    404 while the tests that use JSON kept passing.
    """
    state = store.load("vid123")
    state.add_clip(
        start_word_index=0,
        end_word_index=9,
        start=12.5,
        end=52.0,
        type="insight",
        rubric={"hook": 8.0},
        score=7.1,
    )
    store.save(state)
    clip_id = store.load("vid123").clips[0].clip_id

    response = spa.test_client().post(
        f"/legacy/jobs/vid123/clips/{clip_id}/review",
        data={"verdict": "approve"},
    )

    assert response.status_code == 302
    assert response.headers["Location"] == f"/legacy/jobs/vid123#{clip_id}"


def test_reserved_prefixes_cover_legacy(spa):
    """``/legacy/jobs/nope`` is a Jinja 404, not the SPA shell.

    Without ``legacy/`` in the reserved list, a mistyped review URL would
    quietly render the React app, which then routes it to the queue - and the
    mistyped id is never mentioned.
    """
    response = spa.test_client().get("/legacy/jobs/does-not-exist")

    assert response.status_code == 404
