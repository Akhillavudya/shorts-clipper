"""The preflight checks (WEB_UI_PLAN.md section 9).

``clipper doctor`` and ``GET /api/doctor`` render the same list, so this file
tests the list rather than either renderer. The checks read the filesystem and
``os.environ`` and nothing else - no network, no subprocess beyond
``shutil.which`` - which is what makes them testable at all.

The one property worth defending hardest: a check reports whether a key is
*present*, never what it is. A doctor that leaked a value would be a worse
problem than any of the things it diagnoses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipper import config as config_module
from clipper.server import doctor
from clipper.server.doctor import Check


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A minimal repo root: the four config files, and no keys in the env."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "settings.yaml").write_text(
        "paths:\n"
        "  jobs_root: jobs\n"
        "providers:\n"
        "  gemini:\n"
        "    api_key_env: TEST_GEMINI_KEY\n"
        "edit:\n"
        "  reframe: centre\n",
        encoding="utf-8",
    )
    for name in ("watchlist.yaml", "caption_style.yaml", "scoring_weights.yaml"):
        (root / "config" / name).write_text("channels: []\n", encoding="utf-8")

    for var in ("TEST_GEMINI_KEY", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    return root


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


# -- overall_status ---------------------------------------------------------


@pytest.mark.parametrize(
    "statuses, expected",
    [
        ([], "ok"),
        (["ok", "ok"], "ok"),
        (["ok", "warn"], "warn"),
        (["warn", "fail", "ok"], "fail"),
    ],
)
def test_overall_status_is_the_worst_one(statuses, expected):
    checks = [Check(f"c{i}", s, "") for i, s in enumerate(statuses)]
    assert doctor.overall_status(checks) == expected


def test_a_warning_is_not_a_failure():
    """The distinction the CLI's exit code rests on.

    No face model means renders centre-crop; no OAuth means upload is
    unavailable. Neither stops a stage, so neither should break a script.
    """
    assert doctor.overall_status([Check("face tracking", "warn", "off")]) == "warn"


# -- the checks that read the filesystem ------------------------------------


def test_cwd_check_fails_where_there_is_no_config(tmp_path):
    check = doctor._check_cwd(tmp_path)
    assert check.status == "fail"
    assert "config" in check.detail
    assert check.hint and "repo root" in check.hint


def test_cwd_check_passes_at_the_repo_root(repo):
    assert doctor._check_cwd(repo).status == "ok"


def test_missing_config_files_are_failures(tmp_path):
    checks = doctor._check_config_files(tmp_path)
    assert {c.status for c in checks} == {"fail"}
    assert all("missing" in c.detail for c in checks)


def test_present_config_files_are_ok(repo):
    assert {c.status for c in doctor._check_config_files(repo)} == {"ok"}


def test_an_empty_watchlist_warns_loudly(repo, monkeypatch):
    monkeypatch.chdir(repo)
    check = doctor._check_watchlist()
    assert check.status == "warn"
    assert "unknown" in check.detail
    assert "render" in check.detail


def test_a_populated_watchlist_counts_its_blocked_channels(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "watchlist.yaml").write_text(
        "channels:\n"
        "  - id: UC_green\n"
        "    name: Green Channel\n"
        "    rights_status: green\n"
        "  - id: UC_red\n"
        "    name: Red Channel\n"
        "    rights_status: red\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    check = doctor._check_watchlist()
    assert check.status == "ok"
    assert check.detail == "2 channels (1 blocked)"


def test_a_jobs_root_inside_onedrive_warns(tmp_path):
    check = doctor._check_jobs_root(tmp_path / "OneDrive" / "Desktop" / "jobs")
    assert check.status == "warn"
    assert "OneDrive" in check.detail
    assert check.hint and "paths.jobs_root" in check.hint


def test_a_jobs_root_outside_onedrive_is_fine(tmp_path):
    assert doctor._check_jobs_root(tmp_path / "clipper-jobs").status == "ok"


def test_the_disk_check_walks_up_to_a_directory_that_exists(tmp_path):
    """A jobs root is measured before it is created, on its first run."""
    check = doctor._check_disk(tmp_path / "not" / "created" / "yet")
    assert check.status in {"ok", "warn"}
    assert "free" in check.detail


def test_human_bytes_reads_like_a_disk_panel():
    assert doctor.human_bytes(512) == "512 B"
    assert doctor.human_bytes(10 * 1024**3) == "10.0 GB"


# -- key presence, never key values -----------------------------------------


def test_provider_keys_report_presence_only(repo, monkeypatch):
    settings = config_module.Settings.load(repo / "config" / "settings.yaml")
    secret = "sk-not-a-real-key-0123456789"

    unset = doctor._check_provider_keys(settings)
    assert [(c.name, c.status) for c in unset] == [("TEST_GEMINI_KEY", "warn")]

    monkeypatch.setenv("TEST_GEMINI_KEY", secret)
    checks = doctor._check_provider_keys(settings)
    assert [(c.name, c.status) for c in checks] == [("TEST_GEMINI_KEY", "ok")]
    assert all(secret not in (c.detail + (c.hint or "")) for c in checks)


def test_oauth_is_reported_as_one_line_because_the_three_are_one_credential(monkeypatch):
    from clipper import google_auth

    for var in (
        google_auth.CLIENT_ID_ENV,
        google_auth.CLIENT_SECRET_ENV,
        google_auth.REFRESH_TOKEN_ENV,
        google_auth.TOKEN_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    assert doctor._check_oauth().status == "warn"

    monkeypatch.setenv(google_auth.CLIENT_ID_ENV, "id")
    partial = doctor._check_oauth()
    assert partial.status == "warn"
    assert "incomplete" in partial.detail

    monkeypatch.setenv(google_auth.CLIENT_SECRET_ENV, "secret")
    monkeypatch.setenv(google_auth.REFRESH_TOKEN_ENV, "refresh")
    assert doctor._check_oauth().status == "ok"


def test_no_check_ever_contains_an_environment_value(repo, monkeypatch):
    """The whole-list version of the rule, not just the provider check."""
    secret = "sk-not-a-real-key-0123456789"
    monkeypatch.chdir(repo)
    for var in ("TEST_GEMINI_KEY", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"):
        monkeypatch.setenv(var, secret)

    checks = doctor.diagnose(config_path=repo / "config" / "settings.yaml", cwd=repo)
    blob = " ".join(f"{c.name} {c.detail} {c.hint or ''}" for c in checks)
    assert secret not in blob


# -- diagnose ---------------------------------------------------------------


def test_diagnose_reports_an_unparseable_settings_file_rather_than_raising(tmp_path):
    broken = tmp_path / "settings.yaml"
    broken.write_text("paths: [unclosed\n", encoding="utf-8")

    checks = doctor.diagnose(config_path=broken, cwd=tmp_path)
    assert _by_name(checks)[str(broken)].status == "fail"


def test_diagnose_reports_a_missing_settings_file_rather_than_raising(tmp_path):
    missing = tmp_path / "nope.yaml"
    checks = doctor.diagnose(config_path=missing, cwd=tmp_path)
    assert _by_name(checks)[str(missing)].status == "fail"


def test_diagnose_runs_the_full_list_on_a_good_checkout(repo, monkeypatch):
    monkeypatch.chdir(repo)
    checks = doctor.diagnose(config_path=repo / "config" / "settings.yaml", cwd=repo)
    names = _by_name(checks)

    assert "ffmpeg + ffprobe" in names
    assert "working directory" in names
    assert "youtube oauth" in names
    assert "face tracking" in names
    assert "watchlist" in names
    assert "jobs root" in names
    assert "free disk" in names
    assert names["working directory"].status == "ok"


def test_diagnose_takes_the_jobs_root_from_settings_when_not_given_one(repo, monkeypatch):
    monkeypatch.chdir(repo)
    checks = doctor.diagnose(config_path=repo / "config" / "settings.yaml", cwd=repo)
    assert str(repo / "jobs") in _by_name(checks)["jobs root"].detail


def test_an_explicit_jobs_root_wins_over_settings(repo, monkeypatch):
    monkeypatch.chdir(repo)
    override = repo / "elsewhere"
    checks = doctor.diagnose(
        config_path=repo / "config" / "settings.yaml", jobs_root=override, cwd=repo
    )
    assert str(override) in _by_name(checks)["jobs root"].detail


def test_face_tracking_is_ok_when_deliberately_off(repo):
    settings = config_module.Settings.load(repo / "config" / "settings.yaml")
    check = doctor._check_face_tracking(settings)
    assert check.status == "ok"
    assert "centre" in check.detail


# -- the import cost of the extraction --------------------------------------


def test_importing_the_cli_does_not_import_flask():
    """``clipper moments`` must not pay to import a web framework.

    ``flask`` is a declared dependency of ``review/`` alone and is imported
    inside the ``review`` command for exactly this reason. Extracting the
    doctor into ``server/`` put a new import on the CLI's path, so the rule
    needs a test rather than a convention: ``server/__init__`` re-exports only
    the Flask-free half, and this is what holds it to that.
    """
    import subprocess
    import sys

    probe = (
        "import sys, clipper.cli; "
        "assert 'flask' not in sys.modules, 'clipper.cli imported flask'; "
        "assert 'clipper.review' not in sys.modules, 'clipper.cli imported clipper.review'"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_server_api_is_importable_from_either_direction():
    """The cycle ``review -> server.api -> review`` must stay broken.

    Both import orders in one subprocess each, because the failure only shows
    up on a cold interpreter: once either module is in ``sys.modules`` the
    cycle resolves by accident.
    """
    import subprocess
    import sys

    for first, second in (
        ("clipper.review", "clipper.server.api"),
        ("clipper.server.api", "clipper.review"),
    ):
        probe = (
            f"import {first}; import {second}; "
            "from clipper.review.app import create_app; create_app(jobs_root='jobs')"
        )
        subprocess.run([sys.executable, "-c", probe], check=True)


# -- the CLI renderer -------------------------------------------------------


def test_the_cli_escapes_rich_markup_in_check_text():
    r"""A Windows path ending in a backslash must not eat the closing tag.

    `C:\` makes rich read `\[` as an escaped bracket, so an unescaped
    `[/green]` printed literally at the end of the free-disk line. Check
    details carry paths and ffmpeg stderr, so escaping every one of them is
    the rule rather than a patch for this single case.
    """
    from typer.testing import CliRunner

    from clipper.cli import app

    output = CliRunner().invoke(app, ["doctor"]).output
    for tag in ("[/green]", "[/yellow]", "[/red]", "[/dim]"):
        assert tag not in output
