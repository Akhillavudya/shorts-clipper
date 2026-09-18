"""Extract the clip span from the source (FULL_PLAN.md section 6, step 1).

Responsibility: source.mp4 plus a snapped span -> clip_NN/cut.mp4.

Re-encode; never stream-copy. Stream copy snaps to the nearest keyframe and
will be off by up to several seconds, silently ruining boundaries that
moments/snap.py computed to the millisecond.

That is worth being concrete about, because ``-c copy`` is the obvious
optimisation and it is wrong here. A typical YouTube encode has keyframes
every 2-10 seconds. Stream copy cannot start anywhere else, so it rounds the
in-point to the previous keyframe - which means the clip opens on up to ten
seconds of the *previous* sentence, and the careful silence-snapping in
``moments/snap.py`` was for nothing. The re-encode is the price of an exact
boundary.

Seeking is done with ``-ss`` before ``-i`` (fast: ffmpeg jumps in the
container rather than decoding to the in-point) combined with re-encoding,
which is accurate in any ffmpeg new enough to matter. ``-ss`` after ``-i``
would also be accurate but decodes everything before the in-point - minutes of
wasted work when cutting from two hours in.
"""

from __future__ import annotations

from pathlib import Path

from clipper import media

CUT_FILENAME = "cut.mp4"

#: Intermediate quality. Higher than the final CRF because this file is
#: re-encoded at least twice more before export, and generation loss compounds.
INTERMEDIATE_CRF = 16
INTERMEDIATE_PRESET = "veryfast"


def cut(
    source: Path | str,
    out_path: Path | str,
    start: float,
    end: float,
    *,
    crf: int = INTERMEDIATE_CRF,
    preset: str = INTERMEDIATE_PRESET,
    fps: float | None = None,
) -> Path:
    """Extract ``[start, end)`` from ``source`` into ``out_path``.

    Always re-encodes. There is deliberately no ``copy=True`` parameter: an
    option that silently moves cut points by seconds does not belong in the
    signature at all.
    """
    source = Path(source)
    out_path = Path(out_path)

    if not source.is_file():
        raise FileNotFoundError(f"no source media at {source}")

    duration = end - start
    if duration <= 0:
        raise ValueError(f"clip span is empty or reversed: start={start} end={end}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        # Container seeking can leave the first packets with negative
        # timestamps; without this the audio and video start a few frames apart.
        "-avoid_negative_ts",
        "make_zero",
        # No -force_key_frames here. Re-encoding always emits an IDR at frame 0,
        # so the cut is already seekable at its own start - and the obvious
        # expression for "keyframe at the beginning", expr:gte(t,0), is true for
        # every frame and silently produces an all-intra file several times the
        # size it should be.
    ]
    if fps:
        args += ["-r", f"{fps:g}"]
    args.append(str(out_path))

    media.run(args)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise media.FFmpegError([str(source)], 1, f"ffmpeg produced no output at {out_path}")
    return out_path


def cut_for_clip(
    source: Path | str,
    clip_dir: Path | str,
    start: float,
    end: float,
    **kwargs,
) -> Path:
    """``cut`` into the conventional ``clips/clip_NN/cut.mp4`` location."""
    return cut(source, Path(clip_dir) / CUT_FILENAME, start, end, **kwargs)


__all__ = ["CUT_FILENAME", "INTERMEDIATE_CRF", "INTERMEDIATE_PRESET", "cut", "cut_for_clip"]
