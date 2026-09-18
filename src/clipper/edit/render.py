"""Final encode, and the last place the rights gate fires (FULL_PLAN.md section 6).

Responsibility: compose the finished stages into clip_NN/final.mp4.

    libx264 -crf 19 -preset slow -pix_fmt yuv420p
    1080x1920, AAC 192k, +faststart

Before writing any output this module calls rights/registry.py and refuses
anything marked red (or unknown) without an explicit override flag. The check
belongs here rather than earlier because this is the only stage that produces
an uploadable artefact.

The gate is the first statement in :func:`render_clip`, before any file is
written - not after the cut, not just before the mux. A red source that
produces an uploadable file in a temp directory has already failed; the point
is that the bytes never exist.

Per-clip stages run under ``store.stage(..., clip_id=...)``, so a clip that
dies in ``reframe`` is rerun from ``reframe`` rather than from the cut.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper import media
from clipper.edit import audio_mix, captions, cut, reframe, track
from clipper.rights import registry
from clipper.state import ClipState, JobState, JobStore
from clipper.transcribe.schema import Transcript

FINAL_FILENAME = "final.mp4"
META_FILENAME = "meta.json"

DEFAULT_CRF = 19
DEFAULT_PRESET = "slow"
DEFAULT_AUDIO_BITRATE = "192k"

Progress = Callable[[str, str], None] | None


class RenderError(RuntimeError):
    """A clip could not be rendered."""


@dataclass(frozen=True)
class RenderResult:
    """What one rendered clip produced."""

    clip_id: str
    path: Path
    duration_s: float
    width: int
    height: int
    upscaled: bool


def _note(on_progress: Progress, stage: str, detail: str = "") -> None:
    if on_progress is not None:
        on_progress(stage, detail)


def _fontsdir_argument(fonts_dir: Path, work_dir: Path) -> str | None:
    """``fontsdir`` for the ass filter, as a path the filter parser survives.

    Same problem as the subtitle file, one directory further away: a filter
    option value is parsed as a filter string, so ``C:`` reads as an option
    separator. A relative path has no drive letter to escape, which is why it
    is preferred; the escaped absolute form is the fallback for a jobs root on
    another drive, where no relative path exists.
    """
    if not fonts_dir.is_dir():
        return None
    try:
        relative = Path(os.path.relpath(fonts_dir, work_dir)).as_posix()
    except ValueError:
        return fonts_dir.as_posix().replace(":", "\\:")
    return relative


def encode_final(
    video: Path | str,
    audio: Path | str | None,
    subtitles: Path | str | None,
    out_path: Path | str,
    *,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    audio_bitrate: str = DEFAULT_AUDIO_BITRATE,
    fps: float | None = None,
    fonts_dir: Path | str | None = None,
) -> Path:
    """Mux and encode the finished clip.

    Subtitles are burned in, not muxed as a track: Shorts does not render a
    subtitle track, and karaoke timing is the point of the captions.

    ffmpeg is run with ``cwd`` set to the clip directory and the subtitle file
    named relative to it. The ``ass=`` filter parses its argument as a filter
    string, so a Windows drive letter has to be escaped as ``C\\:/...`` and a
    space or a comma in a parent directory name breaks it outright. Staying in
    the directory sidesteps all of it.

    ``fonts_dir`` is handed to libass so the caption font ships with the repo
    rather than being whatever the machine happens to have installed. Without
    it a missing font is not an error: libass substitutes silently, the encode
    succeeds, and every value in ``caption_style.yaml`` is applied to a
    typeface nobody chose.
    """
    video = Path(video)
    out_path = Path(out_path)
    work_dir = out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    args: list[str] = ["-y", "-i", video.name if video.parent == work_dir else str(video)]

    if audio is not None:
        audio = Path(audio)
        args += ["-i", audio.name if audio.parent == work_dir else str(audio)]

    if subtitles is not None:
        subtitles = Path(subtitles)
        if subtitles.parent != work_dir:
            raise RenderError(
                f"subtitle file {subtitles} must live in the clip directory "
                f"{work_dir}; ffmpeg's ass= filter cannot be given an absolute "
                f"path portably."
            )
        ass_filter = f"ass={subtitles.name}"
        if fonts_dir is not None:
            argument = _fontsdir_argument(Path(fonts_dir).resolve(), work_dir.resolve())
            if argument:
                ass_filter += f":fontsdir={argument}"
        args += ["-vf", ass_filter]

    args += [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0" if audio is not None else "0:a:0?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-ar",
        "48000",
        "-ac",
        "2",
        # Metadata at the front of the file, so playback starts before the
        # whole thing has downloaded.
        "-movflags",
        "+faststart",
        "-shortest",
    ]
    if fps:
        args += ["-r", f"{fps:g}"]
    args.append(out_path.name)

    media.run(args, cwd=work_dir)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RenderError(f"ffmpeg produced no output at {out_path}")
    return out_path


def render_clip(
    store: JobStore,
    state: JobState,
    clip: ClipState,
    *,
    source: Path,
    transcript: Transcript,
    edit_config: dict[str, Any],
    caption_style: dict[str, Any],
    fonts_dir: Path | str | None = None,
    music: audio_mix.MusicPlan | None = None,
    tracking: track.TrackingOptions | None = None,
    allow_red: bool = False,
    override_reason: str | None = None,
    force: bool = False,
    on_progress: Progress = None,
) -> RenderResult:
    """Cut, reframe, mix, caption and encode one clip.

    The rights gate fires first. Everything after it is stage-wrapped so a
    partial failure is resumable.
    """
    registry.check_export(state, allow_red=allow_red, reason=override_reason)
    store.save(state)

    if clip.start is None or clip.end is None:
        raise RenderError(
            f"{clip.clip_id} has no cut points. They are set by moments/snap.py; "
            f"run `clipper moments` first."
        )

    clip_dir = store.clip_dir(state.video_id, clip.clip_id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    width = int(edit_config.get("width", 1080))
    height = int(edit_config.get("height", 1920))
    fps = edit_config.get("fps")

    cut_path = clip_dir / cut.CUT_FILENAME
    reframed_path = clip_dir / reframe.REFRAMED_FILENAME
    mixed_path = clip_dir / audio_mix.MIXED_FILENAME
    music_path = clip_dir / audio_mix.MUSIC_MIXED_FILENAME
    captions_path = clip_dir / captions.CAPTIONS_FILENAME
    final_path = clip_dir / FINAL_FILENAME

    # -- cut ------------------------------------------------------------
    if store.should_run(clip.stage("cut"), force=force) or not cut_path.is_file():
        _note(on_progress, "cut", f"{clip.start:.2f}s - {clip.end:.2f}s")
        with store.stage(state, "cut", clip_id=clip.clip_id) as st:
            st.detail["start"] = clip.start
            st.detail["end"] = clip.end
            cut.cut(source, cut_path, clip.start, clip.end, fps=fps)

    # -- reframe --------------------------------------------------------
    # On a resumed render the reframe stage is skipped, so there is no plan to
    # read the geometry from; the previous run's answer is on the stage record.
    upscaled = bool(clip.stage("reframe").detail.get("upscaled", False))
    if store.should_run(clip.stage("reframe"), force=force) or not reframed_path.is_file():
        _note(
            on_progress,
            "reframe",
            f"{'tracked' if tracking else 'centre'} crop to {width}x{height}",
        )
        with store.stage(state, "reframe", clip_id=clip.clip_id) as st:
            # Unlike every other stage this one writes a different set of keys
            # depending on which mode it took, and stage detail survives a
            # rerun. Without the clear, a --centre-crop rerun reports the
            # previous tracked run's face coverage as if it were its own.
            st.detail.clear()
            result = reframe.reframe(
                cut_path,
                reframed_path,
                width=width,
                height=height,
                fps=fps,
                tracking=tracking,
            )
            plan = result.plan
            st.detail["mode"] = result.mode
            st.detail["crop"] = f"{plan.crop_width}x{plan.crop_height}+{plan.crop_x}+{plan.crop_y}"
            # Why a render came out centre-cropped is the question asked of
            # this stage, so the reason is recorded whether or not it is bad
            # news - a fallback note and a "nothing to pan across" note look
            # the same in the output and mean very different things.
            if result.track_note:
                st.detail["track_note"] = result.track_note
            if result.crop_path is not None:
                st.detail["face_coverage"] = round(result.crop_path.coverage, 3)
                st.detail["scene_cuts"] = len(result.crop_path.scene_cuts)
                st.detail["crop_travel_px"] = result.crop_path.travel_px
            upscaled = plan.is_upscaling
            st.detail["upscaled"] = upscaled
            if upscaled:
                st.detail["note"] = "source is below 1080p after the 9:16 crop; output will be soft"

    # -- audio ----------------------------------------------------------
    if store.should_run(clip.stage("audio_mix"), force=force) or not mixed_path.is_file():
        target = float(edit_config.get("loudness_lufs", audio_mix.TARGET_LUFS))
        _note(on_progress, "audio_mix", f"two-pass loudnorm to {target} LUFS")
        with store.stage(state, "audio_mix", clip_id=clip.clip_id) as st:
            st.detail["target_lufs"] = target
            audio_mix.normalize(cut_path, mixed_path, target_lufs=target)

    # -- music ----------------------------------------------------------
    # A separate stage and a separate file, rather than mixing into
    # mixed.wav. Keeping the speech-only mix means the bed can be re-cut, or
    # dropped entirely, without re-running loudnorm, and the two are there to
    # compare by ear.
    audio_for_encode = mixed_path
    if music is not None:
        audio_for_encode = music_path
        # A completed music stage is only reusable if it was the *same* bed.
        # Swapping the track or the level and getting the previous render back
        # is the kind of silence that costs an afternoon.
        previous = clip.stage("music").detail
        changed = (
            previous.get("track") != str(music.track)
            or previous.get("bed_lufs") != music.bed_lufs
            or previous.get("duck") != music.duck
        )
        if (
            store.should_run(clip.stage("music"), force=force)
            or changed
            or not music_path.is_file()
        ):
            _note(on_progress, "music", f"{music.track.name} at {music.bed_lufs:g} LUFS")
            with store.stage(state, "music", clip_id=clip.clip_id) as st:
                st.detail["track"] = str(music.track)
                st.detail["bed_lufs"] = music.bed_lufs
                st.detail["duck"] = music.duck
                audio_mix.mix_music(
                    mixed_path,
                    music_path,
                    music,
                    duration_s=clip.end - clip.start,
                )
    elif music_path.is_file():
        # A previous render laid a bed down and this one was asked not to.
        # Leaving the file would silently encode the stale mix.
        music_path.unlink()

    # -- captions -------------------------------------------------------
    if store.should_run(clip.stage("captions"), force=force) or not captions_path.is_file():
        words = [w for w in transcript.words if w.end > clip.start and w.start < clip.end]
        _note(on_progress, "captions", f"{len(words)} words")
        with store.stage(state, "captions", clip_id=clip.clip_id) as st:
            st.detail["word_count"] = len(words)
            st.detail["emphasis_words"] = clip.emphasis_words
            captions.write(
                words,
                caption_style,
                captions_path,
                width=width,
                height=height,
                emphasis_words=clip.emphasis_words,
                hook_text=clip.hook_rewrite,
                offset_s=clip.start,
                clip_duration_s=clip.end - clip.start,
            )

    # -- final encode ---------------------------------------------------
    # The encode is rerun when the audio it was built from changes, so
    # --no-music on an already-rendered clip does not return the musical one.
    audio_changed = clip.stage("render").detail.get("audio") != audio_for_encode.name
    if (
        store.should_run(clip.stage("render"), force=force)
        or audio_changed
        or not final_path.is_file()
    ):
        _note(on_progress, "render", final_path.name)
        with store.stage(state, "render", clip_id=clip.clip_id) as st:
            encode_final(
                reframed_path,
                audio_for_encode,
                captions_path,
                final_path,
                crf=int(edit_config.get("crf", DEFAULT_CRF)),
                preset=str(edit_config.get("preset", DEFAULT_PRESET)),
                audio_bitrate=str(edit_config.get("audio_bitrate", DEFAULT_AUDIO_BITRATE)),
                fps=fps,
                fonts_dir=fonts_dir,
            )
            st.detail["output"] = str(final_path)
            st.detail["audio"] = audio_for_encode.name

    info = media.video_info(final_path)
    write_meta(
        clip_dir,
        state=state,
        clip=clip,
        transcript=transcript,
        info=info,
        upscaled=upscaled,
        music=music,
    )

    return RenderResult(
        clip_id=clip.clip_id,
        path=final_path,
        duration_s=info.duration_s,
        width=info.width,
        height=info.height,
        upscaled=upscaled,
    )


def write_meta(
    clip_dir: Path,
    *,
    state: JobState,
    clip: ClipState,
    transcript: Transcript,
    info: media.VideoInfo,
    upscaled: bool = False,
    music: audio_mix.MusicPlan | None = None,
) -> Path:
    """Write ``meta.json`` - everything an upload or a review needs.

    Includes the attribution fields, because a green status that came from a
    CC-BY licence carries an attribution obligation, and the person uploading
    should not have to go back to the watchlist to find it.
    """
    payload = {
        "clip_id": clip.clip_id,
        "video_id": state.video_id,
        "source_url": state.source_url,
        "source_title": state.title,
        "channel_title": state.channel_title,
        "channel_id": state.channel_id,
        "rights_status": state.rights_status.value,
        "rights_note": state.rights_note,
        "rights_override": state.rights_override,
        "rights_override_reason": state.rights_override_reason,
        "licence": state.licence,
        "start": clip.start,
        "end": clip.end,
        "duration_s": info.duration_s,
        "width": info.width,
        "height": info.height,
        "upscaled": upscaled,
        "music": (
            {
                "track": music.track.name,
                "bed_lufs": music.bed_lufs,
                "duck": music.duck,
            }
            if music is not None
            else None
        ),
        "score": clip.score,
        "rubric": clip.rubric,
        "type": clip.type,
        "suggested_title": clip.suggested_title,
        "hook_rewrite": clip.hook_rewrite,
        "emphasis_words": clip.emphasis_words,
        "reason": clip.reason,
        "transcript": (
            transcript.text_for(clip.start_word_index, clip.end_word_index)
            if clip.start_word_index is not None and clip.end_word_index is not None
            else ""
        ),
    }
    path = clip_dir / META_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_AUDIO_BITRATE",
    "DEFAULT_CRF",
    "DEFAULT_PRESET",
    "FINAL_FILENAME",
    "META_FILENAME",
    "RenderError",
    "RenderResult",
    "encode_final",
    "render_clip",
    "write_meta",
]
