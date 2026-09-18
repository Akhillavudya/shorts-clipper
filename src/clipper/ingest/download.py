"""yt-dlp wrapper (FULL_PLAN.md section 2).

Responsibility: fetch a source video to jobs/<video_id>/source.mp4 and record
the resolved metadata (title, channel, duration, publish date, licence) onto
the job state.

Also accepts a local file path - permissioned creators will often just send the
file, which sidesteps the ToS question around downloading entirely.

Picks a format tall enough to survive a 9:16 crop without upscaling: the crop
keeps a ~0.5625 slice of the width, so a 1080p source is the practical floor
for a 1080x1920 export.

A local file is *not* copied into the job directory. Duplicating a multi-GB
source to gain a consistent filename is a bad trade; the path goes on the job
state instead and :func:`source_video_path` resolves either shape. Every
downstream stage asks that function rather than assuming ``source.mp4``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clipper.state import JobState, JobStore

SOURCE_FILENAME = "source.mp4"

#: Extensions accepted as a local source. yt-dlp is not consulted for these.
LOCAL_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi", ".flv", ".ts", ".mpg", ".mpeg"}
)

#: 1080p floor. A 9:16 crop of a 16:9 frame keeps 0.5625 of the width, so a
#: 1920x1080 source yields 607x1080 before scaling - already an upscale to
#: 1080x1920. Below 1080p the result is visibly soft.
MIN_SOURCE_HEIGHT = 1080

#: Ceiling, opposite problem to the floor. ``bestvideo[height>=1080]`` takes
#: the *best* match, so a channel that uploads in 4K costs a gigabyte per
#: episode to produce a 40-second Short. 1440p is the compromise: the 9:16 crop
#: of a 2560x1440 frame is 810x1440, a 1.33x upscale to 1080x1920 rather than
#: 1080p's 1.78x, at 45% of the 2160p download. Raise it to 2160 for a source
#: worth the bytes - that is the only height that never upscales - or lower it
#: to 1080 on a metered connection.
DEFAULT_MAX_SOURCE_HEIGHT = 1440


def format_selector(max_height: int | None = DEFAULT_MAX_SOURCE_HEIGHT) -> str:
    """Preference chain for yt-dlp's ``format``.

    Each fallback loosens one constraint, ending at "whatever exists" so a
    lower-resolution source still runs rather than hard-failing; the actual
    height is recorded so the softness is traceable afterwards.
    """
    ceiling = f"[height<={max_height}]" if max_height else ""
    return (
        f"bestvideo[height>={MIN_SOURCE_HEIGHT}]{ceiling}[ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height>={MIN_SOURCE_HEIGHT}]{ceiling}+bestaudio/"
        f"best[height>={MIN_SOURCE_HEIGHT}]{ceiling}/"
        f"bestvideo[height>={MIN_SOURCE_HEIGHT}]+bestaudio/"
        f"bestvideo+bestaudio/best"
    )


#: The default chain, for callers that do not care.
FORMAT_SELECTOR = format_selector()

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


class IngestError(RuntimeError):
    """The source could not be resolved or fetched."""


def _slugify(text: str, *, limit: int = 48) -> str:
    cleaned = _UNSAFE_ID_CHARS.sub("_", text).strip("_.")
    return (cleaned or "source")[:limit]


def looks_like_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "www."))


@dataclass
class SourceRef:
    """A resolved source, before anything has been downloaded."""

    video_id: str
    is_local: bool
    url: str | None = None
    path: Path | None = None
    title: str | None = None
    channel_id: str | None = None
    channel_title: str | None = None
    duration_s: float | None = None
    published_at: datetime | None = None
    licence: str | None = None
    height: int | None = None
    info: dict[str, Any] = field(default_factory=dict)


def _parse_upload_date(raw: str | None) -> datetime | None:
    """yt-dlp gives ``upload_date`` as ``YYYYMMDD`` with no timezone."""
    if not raw or len(raw) != 8 or not raw.isdigit():
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def _ydl_options(**overrides: Any) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "format": FORMAT_SELECTOR,
        # Remux rather than transcode: the download is a faithful copy of the
        # source, and every quality decision happens later in edit/.
        "merge_output_format": "mp4",
        # An episode is a few hundred megabytes over a domestic connection, so
        # a read timeout part-way through is normal rather than exceptional.
        # Chunked requests give the retry something to resume *to*: without
        # http_chunk_size a stalled read can sit on one open request for the
        # whole file. continuedl then picks the .part up where it stopped, so a
        # retried ingest does not start from zero.
        "socket_timeout": 30,
        "retries": 20,
        "fragment_retries": 20,
        "continuedl": True,
        "http_chunk_size": 10 * 1024 * 1024,
    }
    opts.update(overrides)
    return opts


def _import_ydl():
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise IngestError(
            "yt-dlp is not installed. Run `uv sync`, or pass a local file path instead of a URL."
        ) from exc
    return YoutubeDL


def probe_local(path: Path) -> SourceRef:
    """Build a :class:`SourceRef` for a file already on disk.

    The video id is derived from the filename and prefixed ``local_`` so a
    local job can never collide with a YouTube id in ``jobs/``.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise IngestError(f"no such file: {path}")
    if path.suffix.lower() not in LOCAL_VIDEO_SUFFIXES:
        raise IngestError(
            f"{path.name}: unrecognised video extension {path.suffix!r}. "
            f"Expected one of {', '.join(sorted(LOCAL_VIDEO_SUFFIXES))}"
        )

    ref = SourceRef(
        video_id=f"local_{_slugify(path.stem)}",
        is_local=True,
        path=path,
        title=path.stem,
    )

    # ffprobe is optional here: ingest is the one stage that must work before
    # ffmpeg is installed, so a missing binary downgrades to "no duration yet"
    # rather than failing the stage.
    try:
        from clipper import media

        info = media.video_info(path)
        ref.duration_s = info.duration_s
        ref.height = info.height
    except Exception:  # noqa: BLE001 - any probe failure is non-fatal here
        pass

    return ref


def resolve(source: str, *, ydl_options: dict[str, Any] | None = None) -> SourceRef:
    """Resolve a URL or path to a :class:`SourceRef` without downloading.

    For a URL this is one metadata call - it is what gives ``ingest`` the video
    id it needs to name the job directory, and the channel id the rights gate
    keys off.
    """
    candidate = Path(source).expanduser()
    if not looks_like_url(source) and candidate.exists():
        return probe_local(candidate)

    if not looks_like_url(source):
        raise IngestError(
            f"{source!r} is neither an existing file nor a URL. Pass a YouTube "
            f"URL, or a path to a local video file."
        )

    YoutubeDL = _import_ydl()
    with YoutubeDL(_ydl_options(skip_download=True, **(ydl_options or {}))) as ydl:
        info = ydl.extract_info(source, download=False)

    if info is None:
        raise IngestError(f"yt-dlp returned no metadata for {source!r}")
    if info.get("_type") == "playlist":
        raise IngestError(
            f"{source!r} resolved to a playlist. Ingest takes one video at a "
            f"time; discovery is what walks a playlist (section 3)."
        )

    video_id = info.get("id")
    if not video_id:
        raise IngestError(f"yt-dlp returned no video id for {source!r}")

    return SourceRef(
        video_id=str(video_id),
        is_local=False,
        url=info.get("webpage_url") or source,
        title=info.get("title"),
        channel_id=info.get("channel_id") or info.get("uploader_id"),
        channel_title=info.get("channel") or info.get("uploader"),
        duration_s=float(info["duration"]) if info.get("duration") else None,
        published_at=_parse_upload_date(info.get("upload_date")),
        licence=info.get("license"),
        height=info.get("height"),
        info=info,
    )


def download(
    ref: SourceRef,
    job_dir: Path | str,
    *,
    overwrite: bool = False,
    max_height: int | None = DEFAULT_MAX_SOURCE_HEIGHT,
    ydl_options: dict[str, Any] | None = None,
) -> Path:
    """Fetch ``ref`` into ``job_dir``. Returns the path to the source media.

    A local source is returned as-is; nothing is copied. An existing
    ``source.mp4`` is reused unless ``overwrite`` is set, so rerunning ``ingest``
    after a later stage failed does not re-download gigabytes.
    """
    if ref.is_local:
        if ref.path is None:
            raise IngestError("local SourceRef has no path")
        return ref.path

    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / SOURCE_FILENAME

    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return target

    YoutubeDL = _import_ydl()
    # yt-dlp appends the real extension; the merge target is mp4 so this lands
    # on source.mp4, but the glob below covers a container it refuses to remux.
    outtmpl = str(job_dir / "source.%(ext)s")
    options = {"format": format_selector(max_height), **(ydl_options or {})}
    with YoutubeDL(_ydl_options(outtmpl=outtmpl, **options)) as ydl:
        ydl.download([ref.url or ""])

    if target.is_file():
        return target

    produced = sorted(p for p in job_dir.glob("source.*") if p.suffix.lower() != ".part")
    if not produced:
        raise IngestError(f"yt-dlp reported success but wrote nothing to {job_dir}")
    return produced[0]


def source_video_path(store: JobStore, state: JobState) -> Path:
    """Where the source media for this job actually lives.

    ``jobs/<id>/source.mp4`` for a download, the original path for a local
    file. Every stage that opens the source calls this rather than assuming.
    """
    downloaded = store.job_dir(state.video_id) / SOURCE_FILENAME
    if downloaded.is_file():
        return downloaded
    if state.source_path:
        path = Path(state.source_path)
        if path.is_file():
            return path
        raise IngestError(
            f"job {state.video_id} points at {path}, which no longer exists. "
            f"Re-run `clipper ingest` with the file in place."
        )
    raise IngestError(f"job {state.video_id} has no source media. Run `clipper ingest` first.")


def apply_metadata(state: JobState, ref: SourceRef) -> JobState:
    """Copy resolved source metadata onto the job state."""
    if ref.is_local:
        state.source_path = str(ref.path)
    else:
        state.source_url = ref.url

    state.title = ref.title or state.title
    state.channel_id = ref.channel_id or state.channel_id
    state.channel_title = ref.channel_title or state.channel_title
    state.duration_s = ref.duration_s if ref.duration_s is not None else state.duration_s
    state.published_at = ref.published_at or state.published_at
    state.licence = ref.licence or state.licence
    return state


__all__ = [
    "FORMAT_SELECTOR",
    "LOCAL_VIDEO_SUFFIXES",
    "MIN_SOURCE_HEIGHT",
    "SOURCE_FILENAME",
    "IngestError",
    "SourceRef",
    "apply_metadata",
    "download",
    "looks_like_url",
    "probe_local",
    "resolve",
    "source_video_path",
]
