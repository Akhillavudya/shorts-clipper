from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.review.app import serve
from clipper.server.operations import RunnerClaim, log_tail


def test_claim_excludes_second_owner_and_releases(tmp_path):
    first, second = RunnerClaim(tmp_path), RunnerClaim(tmp_path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="Another server"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_tail_is_bounded_and_preserves_ansi(tmp_path):
    path = tmp_path / "worker.log"
    path.write_bytes(b"x" * 2_000_000 + b"\nold\n\x1b[31merror\nlast\n")
    assert log_tail(path, 2) == "\x1b[31merror\nlast"


def test_remote_bind_refused_before_app_creation(monkeypatch):
    create = Mock()
    monkeypatch.setattr("clipper.review.app.create_app", create)
    with pytest.raises(ValueError, match="allow-remote"):
        serve(host="0.0.0.0", ui="spa")
    create.assert_not_called()


def test_port_conflict_never_starts_runner(tmp_path, monkeypatch):
    from clipper.state import JobStore

    runner = Mock()
    monkeypatch.setattr("clipper.review.app.build_runner", lambda *a, **k: runner)
    monkeypatch.setattr("clipper.review.app.create_app", lambda **k: Mock(config={}))
    monkeypatch.setattr("werkzeug.serving.make_server", Mock(side_effect=SystemExit(1)))
    with pytest.raises(RuntimeError, match="--port"):
        serve(store=JobStore(tmp_path), run_queue=True)
    runner.start.assert_not_called()


def test_server_exception_stops_worker_and_releases_claim(tmp_path, monkeypatch):
    from clipper.state import JobStore

    runner = Mock()
    server = Mock()
    server.serve_forever.side_effect = KeyboardInterrupt()
    app = Mock(config={"JOB_STORE": JobStore(tmp_path)})
    monkeypatch.setattr("clipper.review.app.build_runner", lambda *a, **k: runner)
    monkeypatch.setattr("clipper.review.app.create_app", lambda **k: app)
    monkeypatch.setattr("werkzeug.serving.make_server", lambda *a, **k: server)
    with pytest.raises(KeyboardInterrupt):
        serve(store=JobStore(tmp_path), run_queue=True)
    runner.shutdown.assert_called_once()
    server.server_close.assert_called_once()
    claim = RunnerClaim(tmp_path)
    claim.acquire()
    claim.release()
    assert (Path(tmp_path) / "_queue/logs/server.log").is_file()
