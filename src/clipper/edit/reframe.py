"""16:9 to 9:16 reframing (FULL_PLAN.md section 6, step 2).

Responsibility: cut.mp4 -> reframed.mp4, at the export resolution.

Two modes, and the geometry of both comes out of :func:`plan_crop`:

* **Centre crop.** A single static rectangle. Correct for the common case - a
  talking head roughly centred in frame - and the only mode Phase 1 had.
* **Tracked crop.** The same rectangle, moved horizontally over time to follow
  the active speaker. The path is built by ``edit/track.py`` and handed to
  ffmpeg as a ``sendcmd`` script; this module only decides the shape of the
  rectangle and writes the script out next to the output.

The tracked mode degrades rather than fails. If MediaPipe, OpenCV or the face
model is missing, or if too few frames contained a face for the path to mean
anything, :func:`reframe` falls back to the centre crop and reports why in
``ReframeResult.track_note`` - which the render stage records on the clip
state, so a Short that is framed badly can be traced to the reason rather than
guessed at.

Still deferred, and still in the plan: the two-person treatments (a stacked
split, or cutting between speakers on diarization turns) and using the
diarization label plus mouth movement to pick the active speaker rather than
size and position alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from clipper import media
from clipper.edit import track

REFRAMED_FILENAME = "reframed.mp4"

DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920


@dataclass(frozen=True)
class CropPlan:
    """The geometry for one clip: what to take, and what to do with it.

    The rectangle's *shape* is fixed for the whole clip; only ``crop_x`` moves,
    and only when a tracked path says so. ``crop_x`` here is where it starts.
    """

    src_width: int
    src_height: int
    crop_width: int
    crop_height: int
    crop_x: int
    crop_y: int
    out_width: int
    out_height: int
    pad: bool

    @property
    def filter_chain(self) -> str:
        """The ffmpeg ``-vf`` string for the static crop."""
        return self.build_filter_chain()

    def build_filter_chain(self, command_file: str | None = None) -> str:
        """The ffmpeg ``-vf`` string this plan describes.

        With ``command_file`` the crop is given an instance name and preceded
        by a ``sendcmd`` reading that file, which is what moves ``crop_x`` over
        the life of the clip. The filename is used bare and unescaped: a filter
        option value is parsed as a filter string, so ffmpeg must be run with
        ``cwd`` set to the directory holding it. See ``media.py``.
        """
        crop = f"crop={self.crop_width}:{self.crop_height}:{self.crop_x}:{self.crop_y}"
        chain: list[str] = []
        if command_file is not None:
            chain.append(f"sendcmd=f={command_file}")
            crop = (
                f"{track.FILTER_NAME}=w={self.crop_width}:h={self.crop_height}"
                f":x={self.crop_x}:y={self.crop_y}"
            )
        chain.append(crop)
        if self.pad:
            # A source narrower than 9:16 cannot be cropped to it without
            # throwing away the top and bottom of a face. Scale to fit and pad
            # instead - letterboxing beats decapitation.
            chain.append(
                f"scale={self.out_width}:{self.out_height}:force_original_aspect_ratio=decrease"
            )
            chain.append(f"pad={self.out_width}:{self.out_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        else:
            chain.append(f"scale={self.out_width}:{self.out_height}:flags=lanczos")
        chain.append("setsar=1")
        return ",".join(chain)

    @property
    def is_upscaling(self) -> bool:
        """Whether the crop is smaller than the export, i.e. output is soft."""
        return self.crop_width < self.out_width or self.crop_height < self.out_height


def plan_crop(
    src_width: int,
    src_height: int,
    *,
    out_width: int = DEFAULT_WIDTH,
    out_height: int = DEFAULT_HEIGHT,
    centre_x: float = 0.5,
) -> CropPlan:
    """Work out the crop rectangle for a source of this shape.

    ``centre_x`` is the horizontal centre of interest as a fraction of the
    frame width, and sets where a static crop sits. A tracked crop overwrites
    it from its own first keyframe.

    Crop dimensions are forced even. Odd values are rejected outright by
    ``yuv420p``, and the failure surfaces as an opaque ffmpeg error several
    stages later.
    """
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"bad source dimensions: {src_width}x{src_height}")

    target_aspect = out_width / out_height
    source_aspect = src_width / src_height

    if source_aspect > target_aspect:
        # Wider than 9:16 - the normal case. Full height, crop the width.
        crop_height = src_height
        crop_width = int(round(src_height * target_aspect))
        pad = False
    else:
        # Already as tall or taller than 9:16. Keep the full frame and pad.
        crop_width = src_width
        crop_height = src_height
        pad = True

    crop_width = min(src_width, crop_width - (crop_width % 2))
    crop_height = min(src_height, crop_height - (crop_height % 2))

    max_x = src_width - crop_width
    crop_x = int(round((src_width * centre_x) - crop_width / 2))
    crop_x = max(0, min(max_x, crop_x))
    crop_x -= crop_x % 2

    crop_y = max(0, (src_height - crop_height) // 2)
    crop_y -= crop_y % 2

    return CropPlan(
        src_width=src_width,
        src_height=src_height,
        crop_width=crop_width,
        crop_height=crop_height,
        crop_x=crop_x,
        crop_y=crop_y,
        out_width=out_width,
        out_height=out_height,
        pad=pad,
    )


def build_path(
    source: Path,
    plan: CropPlan,
    *,
    options: track.TrackingOptions,
    fps: float,
    duration_s: float | None,
) -> tuple[track.CropPath | None, str]:
    """Try to track ``source``. Returns the path and a note, either of which
    may be the interesting one.

    The note is empty when the path is usable. Everything else - a missing
    model, an unreadable file, too few faces to be believed - comes back as a
    reason string and a ``None`` path, and the caller centre-crops.
    """
    try:
        path = track.plan_path(source, plan, options=options, fps=fps, duration_s=duration_s)
    except track.TrackingUnavailable as exc:
        return None, str(exc)

    if not path.keys:
        return None, "no frames were sampled; nothing to track"
    if path.coverage < options.min_coverage:
        return None, (
            f"a face was found in {path.coverage:.0%} of sampled frames, under the "
            f"{options.min_coverage:.0%} floor; centre crop is the safer answer"
        )
    return path, ""


@dataclass(frozen=True)
class ReframeResult:
    """What one reframe produced, and how it decided to produce it."""

    path: Path
    plan: CropPlan
    mode: str
    crop_path: track.CropPath | None = None
    track_note: str = ""

    @property
    def tracked(self) -> bool:
        return self.crop_path is not None


def reframe(
    source: Path | str,
    out_path: Path | str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float | None = None,
    crf: int = 16,
    preset: str = "veryfast",
    centre_x: float = 0.5,
    tracking: track.TrackingOptions | None = None,
) -> ReframeResult:
    """Crop ``source`` to vertical and scale it to the export size.

    With ``tracking`` the crop follows the active speaker; without it, or when
    tracking cannot run, it is a static centre crop. Either way the result
    carries the plan used, so a soft output is traceable to an upscaled crop
    rather than being a mystery, and a badly framed one to the reason tracking
    was refused.

    ffmpeg runs with ``cwd`` set to the output directory so the sendcmd script
    can be named without escaping a Windows drive letter through the filter
    parser - the same reason ``render.encode_final`` does it for subtitles.
    """
    source = Path(source)
    out_path = Path(out_path)

    if not source.is_file():
        raise FileNotFoundError(f"no input at {source}")

    info = media.video_info(source)
    plan = plan_crop(info.width, info.height, out_width=width, out_height=height, centre_x=centre_x)

    work_dir = out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    command_path = work_dir / track.COMMAND_FILENAME

    crop_path: track.CropPath | None = None
    note = ""
    if tracking is not None and not plan.pad:
        crop_path, note = build_path(
            source,
            plan,
            options=tracking,
            fps=info.fps or 30.0,
            duration_s=info.duration_s,
        )
    elif tracking is not None:
        note = "source is no wider than 9:16, so there is nothing to pan across"

    if crop_path is not None:
        command_path.write_text(crop_path.sendcmd(), encoding="utf-8")
        # Start the static crop where the path starts. The t=0 command sets
        # the same value, but only if commands land before the first frame is
        # filtered; this way the first frame is right regardless.
        plan = replace(plan, crop_x=crop_path.keys[0][1])
        filter_chain = plan.build_filter_chain(command_path.name)
    else:
        # A stale script from a previous tracked render would be read by
        # nothing, but it would also claim this render was tracked.
        command_path.unlink(missing_ok=True)
        filter_chain = plan.filter_chain

    args = [
        "-y",
        "-i",
        str(source.resolve()),
        "-vf",
        filter_chain,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
    ]
    if fps:
        args += ["-r", f"{fps:g}"]
    args.append(out_path.name)

    media.run(args, cwd=work_dir)
    return ReframeResult(
        path=out_path,
        plan=plan,
        mode="track" if crop_path is not None else "centre_crop",
        crop_path=crop_path,
        track_note=note,
    )


def reframe_for_clip(source: Path | str, clip_dir: Path | str, **kwargs) -> ReframeResult:
    """``reframe`` into the conventional ``clips/clip_NN/reframed.mp4``."""
    return reframe(source, Path(clip_dir) / REFRAMED_FILENAME, **kwargs)


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "REFRAMED_FILENAME",
    "CropPlan",
    "ReframeResult",
    "build_path",
    "plan_crop",
    "reframe",
    "reframe_for_clip",
]
