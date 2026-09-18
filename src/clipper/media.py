"""ffmpeg / ffprobe plumbing, shared by ingest and edit.

Everything that shells out to ffmpeg goes through here, so there is one place
that knows how the binary is found, how a failure is reported, and how filter
arguments that embed a path are quoted.

That last one is the reason this module exists rather than each stage calling
``subprocess`` itself. Filters which take a filename (``ass=``, ``amovie=``)
are the one place ffmpeg's escaping differs per platform: a Windows path needs
``C\\:/dir/file.ass``. Callers avoid the problem entirely by running with
``cwd`` set to the directory holding the file and passing a bare filename.

Nothing here ever passes ``-c copy`` for a cut. Stream copy snaps to the
nearest keyframe and silently moves a boundary that ``moments/snap.py``
computed to the millisecond; see ``edit/cut.py``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# silencedetect reports to stderr as free text; it has no structured output.
_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


class FFmpegMissing(RuntimeError):
    """ffmpeg or ffprobe is not on PATH.

    Raised eagerly with an install hint, rather than letting a bare
    ``FileNotFoundError`` surface from inside a stage.
    """


class FFmpegError(RuntimeError):
    """An ffmpeg invocation exited non-zero. Carries the tail of its stderr."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        super().__init__(f"ffmpeg exited {returncode}\n  command: {' '.join(cmd[:14])} ...\n{tail}")


def require(binary: str = FFMPEG) -> str:
    """Return the resolved path to ``binary``, or explain how to install it."""
    found = shutil.which(binary)
    if found is None:
        raise FFmpegMissing(
            f"{binary!r} is not on PATH. Every stage past `ingest` needs it.\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   apt install ffmpeg"
        )
    return found


def available() -> bool:
    """Whether both binaries are present. For status output, not for gating."""
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


def run(
    args: list[str],
    *,
    cwd: Path | str | None = None,
    binary: str = FFMPEG,
    timeout_s: float | None = None,
) -> str:
    """Run ffmpeg with ``args`` (binary name excluded) and return its stderr.

    stderr is the return value because that is where ffmpeg writes everything
    worth reading: stream mapping, and filter output such as ``silencedetect``
    spans and ``loudnorm``'s measurement pass.
    """
    exe = require(binary)
    cmd = [exe, "-hide_banner", "-nostdin", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr or "")
    return proc.stderr or ""


def probe(path: Path | str) -> dict[str, Any]:
    """ffprobe a media file and return the parsed JSON."""
    exe = require(FFPROBE)
    cmd = [
        exe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr or "")
    return json.loads(proc.stdout or "{}")


@dataclass(frozen=True)
class VideoInfo:
    """The handful of source properties the pipeline actually branches on."""

    width: int
    height: int
    fps: float
    duration_s: float
    has_audio: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def _parse_fps(rate: str | None) -> float:
    """ffprobe reports frame rates as a string ratio, e.g. ``30000/1001``."""
    if not rate or rate in {"0/0", "N/A"}:
        return 0.0
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            divisor = float(den)
        except ValueError:
            return 0.0
        if not divisor:
            return 0.0
        try:
            return float(num) / divisor
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def video_info(path: Path | str) -> VideoInfo:
    """Width, height, fps, duration and audio presence for one file."""
    info = probe(path)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError([str(path)], 1, f"no video stream in {path}")

    duration = 0.0
    for source in ((info.get("format") or {}), video):
        raw = source.get("duration")
        if raw in (None, "N/A"):
            continue
        try:
            duration = float(raw)
        except (TypeError, ValueError):
            continue
        break

    return VideoInfo(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        duration_s=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def duration_s(path: Path | str) -> float:
    """Container duration in seconds. Works for audio-only files too."""
    info = probe(path)
    raw = (info.get("format") or {}).get("duration")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def parse_silences(stderr: str, *, total_s: float | None = None) -> list[tuple[float, float]]:
    """Pull ``(start, end)`` spans out of ``silencedetect`` stderr.

    Split out from :func:`detect_silences` so the parsing can be tested without
    ffmpeg installed - which matters, because the suite must run without it.

    A silence still open when the file ends has no ``silence_end`` line. Given
    ``total_s`` it is closed at the file duration; that trailing gap is often
    the most useful boundary in the file, so dropping it would be a real loss.
    """
    spans: list[tuple[float, float]] = []
    open_start: float | None = None

    for line in stderr.splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match is not None:
            open_start = max(0.0, float(start_match.group(1)))
            continue
        end_match = _SILENCE_END.search(line)
        if end_match is not None and open_start is not None:
            end = float(end_match.group(1))
            if end > open_start:
                spans.append((open_start, end))
            open_start = None

    if open_start is not None and total_s is not None and total_s > open_start:
        spans.append((open_start, total_s))

    spans.sort()
    return spans


def detect_silences(
    path: Path | str,
    *,
    threshold_db: float = -30.0,
    min_silence_ms: int = 250,
) -> list[tuple[float, float]]:
    """Silence spans in ``path``, as ``(start, end)`` seconds.

    Feeds ``moments/snap.py``. A cut placed inside a silence does not clip the
    first syllable, which is the artefact that makes an automated clip
    instantly recognisable.
    """
    stderr = run(
        [
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={threshold_db}dB:d={min_silence_ms / 1000.0:.3f}",
            "-f",
            "null",
            "-",
        ]
    )
    return parse_silences(stderr, total_s=duration_s(path))


__all__ = [
    "FFMPEG",
    "FFPROBE",
    "FFmpegError",
    "FFmpegMissing",
    "VideoInfo",
    "available",
    "detect_silences",
    "duration_s",
    "parse_silences",
    "probe",
    "require",
    "run",
    "video_info",
]
