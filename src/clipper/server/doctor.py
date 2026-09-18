"""The preflight checks, in one place (WEB_UI_PLAN.md section 9).

These used to live inline in ``cli.doctor``, printing rich markup as they went,
which meant the web UI could only have them by reimplementing them. They are
data now: :func:`diagnose` returns a list of :class:`Check`, ``cli.doctor``
renders it with rich and ``GET /api/doctor`` serialises it. Adding a check
gains it in both places, which is the whole point of the extraction.

**Nothing here touches the network.** Every key check reports *presence* - a
boolean over ``os.environ`` - and never the value, and never whether the key
works. "Is the key set" is answerable offline and instantly; "is the key valid"
costs a request, an API quota unit and a test that has to be faked. A check
that blurred the two would be worse than no check.

Two checks named in the plan are not here yet, because the things they check do
not exist yet: ``webui/dist`` present (phase 3) and another runner holding the
claim file (phase 6). They join this list in those phases.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from clipper import config as config_module
from clipper import media

Status = Literal["ok", "warn", "fail"]

#: Below this, a 1440p ingest plus its render has a real chance of running the
#: disk out mid-encode. The same floor the job runner refuses to start under.
DISK_FLOOR_BYTES = 10 * 1024**3

#: Files the CLI already required, and what is lost when one is absent.
_REQUIRED_CONFIG: tuple[tuple[Path, str], ...] = (
    (config_module.WATCHLIST_PATH, "rights_status for every source resolves from this"),
    (config_module.CAPTION_STYLE_PATH, "every caption visual lives here"),
    (config_module.SCORING_WEIGHTS_PATH, "the rubric weights the calibration loop retunes"),
)


@dataclass(frozen=True)
class Check:
    """One diagnostic line.

    ``fail`` means a stage will not run. ``warn`` means something runs in a
    degraded mode that was chosen deliberately - no face model means a centre
    crop, no OAuth means upload is unavailable and nothing else is. Keeping
    those apart is what stops the doctor being a wall of red people stop
    reading.
    """

    name: str
    status: Status
    detail: str
    hint: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def overall_status(checks: list[Check]) -> Status:
    """The worst status in the list. Drives the CLI's exit code."""
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "ok"


def human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def _check_ffmpeg() -> Check:
    if media.available():
        return Check("ffmpeg + ffprobe", "ok", "found on PATH")
    return Check(
        "ffmpeg + ffprobe",
        "fail",
        "not on PATH - every stage past ingest needs them",
        hint="unpack a build under %LOCALAPPDATA%/Programs/ffmpeg/bin and add it to Path",
    )


def _check_cwd(cwd: Path) -> Check:
    """Whether the process can find ``config/`` from where it was started.

    ``config.CONFIG_DIR`` is the relative ``Path("config")``, so a server
    started from anywhere else silently reads no config at all. That is a
    baffling failure three stages in and an obvious one here.
    """
    if (cwd / config_module.SETTINGS_PATH).is_file():
        return Check("working directory", "ok", str(cwd))
    return Check(
        "working directory",
        "fail",
        f"{cwd} has no {config_module.SETTINGS_PATH} - config paths resolve against the CWD",
        hint="run from the repo root, or pass --config with an absolute path",
    )


def _check_config_files(cwd: Path) -> list[Check]:
    checks = []
    for path, why in _REQUIRED_CONFIG:
        exists = (cwd / path).is_file()
        checks.append(
            Check(
                str(path),
                "ok" if exists else "fail",
                "ok" if exists else f"missing - {why}",
            )
        )
    return checks


def _check_provider_keys(settings: config_module.Settings) -> list[Check]:
    """Presence only, never the value. See the module docstring."""
    checks = []
    for name, block in (settings.section("providers") or {}).items():
        env = block.get("api_key_env") if isinstance(block, dict) else None
        if not env:
            continue
        present = bool(os.environ.get(env))
        checks.append(
            Check(
                env,
                "ok" if present else "warn",
                f"{'set' if present else 'unset'} ({name})",
                hint=None if present else f"put {env} in .env - the LLM stages need it",
            )
        )
    return checks


def _check_oauth() -> Check:
    """``upload`` and ``calibrate`` act as the channel; both need all three.

    Reported as one line because a refresh token is only usable alongside the
    client that minted it - two of three set is not two thirds of working.
    """
    from clipper import google_auth

    present = {
        var: bool(os.environ.get(var))
        for var in (
            google_auth.CLIENT_ID_ENV,
            google_auth.CLIENT_SECRET_ENV,
            google_auth.REFRESH_TOKEN_ENV,
        )
    }
    if all(present.values()) or os.environ.get(google_auth.TOKEN_ENV):
        return Check("youtube oauth", "ok", "set (upload, calibrate)")
    if any(present.values()):
        unset = ", ".join(var for var, ok in present.items() if not ok)
        return Check(
            "youtube oauth",
            "warn",
            f"incomplete - unset: {unset}",
            hint="a refresh token only works with the client that minted it",
        )
    return Check(
        "youtube oauth",
        "warn",
        "unset - upload and calibrate are unavailable",
        hint="create a Desktop app OAuth client, then run `clipper auth` once",
    )


def _check_face_tracking(settings: config_module.Settings) -> Check:
    """Optional by design: without it ``render`` centre-crops and records why."""
    from clipper.edit import track

    edit_cfg = settings.edit
    if not track.tracking_enabled(edit_cfg):
        return Check("face tracking", "ok", "off (edit.reframe: centre)")

    missing = [
        package
        for module, package in (("cv2", "opencv-python"), ("mediapipe", "mediapipe"))
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return Check(
            "face tracking",
            "warn",
            f"{', '.join(missing)} not installed - renders will centre-crop",
            hint="uv sync, or set edit.reframe: centre to stop being asked",
        )

    options = track.options_from_config(edit_cfg)
    if not options.model_path.is_file():
        return Check(
            "face tracking",
            "warn",
            f"model missing at {options.model_path} - renders will centre-crop",
            hint="see assets/models/README.md",
        )
    return Check("face tracking", "ok", "ready")


def _check_watchlist() -> Check:
    """The loud one. An empty watchlist means nothing can ever be rendered."""
    try:
        from clipper.rights.registry import Watchlist

        watchlist = Watchlist.load()
    except Exception as exc:  # noqa: BLE001 - reported as a check, never raised
        return Check("watchlist", "fail", str(exc))

    total = len(watchlist)
    if total == 0:
        return Check(
            "watchlist",
            "warn",
            "empty - every source resolves to 'unknown', and render, publish and "
            "upload all refuse it",
            hint="add channels to config/watchlist.yaml; it is the highest-leverage "
            "file in the project",
        )
    blocked = sum(1 for c in watchlist.channels if not c.is_exportable)
    return Check("watchlist", "ok", f"{total} channels ({blocked} blocked)")


def _check_disk(jobs_root: Path) -> Check:
    """1440p sources are 0.5-2 GB, are never deleted, and land here."""
    probe = jobs_root if jobs_root.is_absolute() else (Path.cwd() / jobs_root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return Check("free disk", "warn", f"could not measure {probe}: {exc}")

    where = probe.anchor or str(probe)
    if usage.free < DISK_FLOOR_BYTES:
        return Check(
            "free disk",
            "warn",
            f"{human_bytes(usage.free)} free on {where} - under the "
            f"{human_bytes(DISK_FLOOR_BYTES)} a job should not start below",
            hint="delete a finished job's source.mp4; it is ~95% of a job's bytes",
        )
    return Check("free disk", "ok", f"{human_bytes(usage.free)} free on {where}")


def _check_jobs_root(jobs_root: Path) -> Check:
    """A jobs root inside OneDrive is a real hazard here, not a theoretical one.

    Multi-GB sources get sync-scanned, can be evicted to cloud-only
    placeholders, and ``os.replace`` can transiently raise ``PermissionError``
    while the sync client holds a handle on the file being replaced.
    """
    resolved = jobs_root if jobs_root.is_absolute() else (Path.cwd() / jobs_root)
    if any("onedrive" in part.lower() for part in resolved.parts):
        return Check(
            "jobs root",
            "warn",
            f"{resolved} is inside OneDrive - multi-GB sources will be synced, and "
            f"atomic writes can fail while the sync client holds a handle",
            hint="set paths.jobs_root somewhere outside OneDrive, e.g. D:/clipper-jobs",
        )
    return Check("jobs root", "ok", str(resolved))


# --------------------------------------------------------------------------
# the two entry points
# --------------------------------------------------------------------------


def run_checks(settings: config_module.Settings, *, jobs_root: Path, cwd: Path) -> list[Check]:
    """Every check, given an already-loaded ``settings``.

    Pure: takes what it needs, returns data, prints nothing and raises nothing.
    :func:`diagnose` is the wrapper that loads the settings for you.
    """
    return [
        _check_ffmpeg(),
        _check_cwd(cwd),
        *_check_config_files(cwd),
        *_check_provider_keys(settings),
        _check_oauth(),
        _check_face_tracking(settings),
        _check_watchlist(),
        _check_jobs_root(jobs_root),
        _check_disk(jobs_root),
    ]


def diagnose(
    *,
    config_path: Path | str = config_module.SETTINGS_PATH,
    jobs_root: Path | str | None = None,
    cwd: Path | None = None,
) -> list[Check]:
    """Load the settings, then run every check. The one call both UIs make.

    A settings file that will not load is reported as a failed check rather
    than raised, because "settings.yaml is unparseable" is exactly the sort of
    thing someone runs the doctor to be told.
    """
    config_path = Path(config_path)
    cwd = cwd or Path.cwd()

    try:
        settings = config_module.Settings.load(config_path)
    except Exception as exc:  # noqa: BLE001 - the doctor reports, it does not raise
        return [_check_ffmpeg(), _check_cwd(cwd), Check(str(config_path), "fail", str(exc))]

    root = Path(jobs_root) if jobs_root is not None else settings.jobs_root
    return [Check(str(config_path), "ok", "loaded"), *run_checks(settings, jobs_root=root, cwd=cwd)]


__all__ = [
    "DISK_FLOOR_BYTES",
    "Check",
    "Status",
    "diagnose",
    "human_bytes",
    "overall_status",
    "run_checks",
]
