"""Loudness normalization and music ducking (FULL_PLAN.md sections 6 and 8).

Responsibility: make the clip sound like it belongs on the platform.

- highpass=f=80, optional afftdn denoise
- Two-pass loudnorm to -14 LUFS (YouTube's target). Two-pass because
  single-pass loudnorm cannot see the whole file and drifts.
- Music, if any: bed at ~-26 dB under speech, ducked out of the way of it
  by ``sidechaincompress`` keyed on the speech itself.

For dense talking-head content music usually hurts, so ``edit.music_enabled``
defaults to false and the bed is opt-in per render.

Speech is normalized first and the bed is added afterwards, never the other
way round. loudnorm measures a whole file: run it over speech+music and the
bed counts towards the -14 LUFS, so a louder bed quietly buys itself volume
by pushing the voice down. Normalizing the voice alone and adding a bed at a
known offset below it keeps the dialogue anchored where the platform wants
it, and costs ~0.3 LU of overshoot at -26 dB, which is inaudible.

The two passes are worth spelling out, since the first one looks like waste.
Pass 1 runs ``loudnorm`` with ``print_format=json`` and discards the audio; all
it produces is the measured integrated loudness, true peak, LRA and offset of
the *whole* file. Pass 2 feeds those measurements back in as
``measured_i``/``measured_tp``/``measured_lra``/``offset``, which turns
loudnorm from a look-ahead limiter guessing as it goes into a single known
gain. Single-pass on a 40-second clip audibly pumps: the opening is normalized
against a level the rest of the clip does not share.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper import media

MIXED_FILENAME = "mixed.wav"
MUSIC_MIXED_FILENAME = "mixed_music.wav"

TARGET_LUFS = -14.0
TARGET_TRUE_PEAK = -1.5
TARGET_LRA = 11.0

#: Rumble, HVAC and mic handling live below this. Nothing in speech does.
HIGHPASS_HZ = 80

#: Where the bed sits when nothing is ducking it, as an integrated loudness
#: target rather than a gain. A gain is meaningless across tracks - the same
#: "-26 dB" is a whisper under one file and a duet with the voice under
#: another - so the music is normalized to this the way the speech is
#: normalized to TARGET_LUFS. -26 LUFS is 12 LU under the speech: present
#: between sentences, texture underneath them.
DEFAULT_BED_LUFS = -26.0

#: The bed is never the thing that clips, so it is held well off the ceiling
#: and the final encode's headroom belongs to the voice.
BED_TRUE_PEAK = -3.0

#: sidechaincompress, keyed on speech. Threshold is linear amplitude, not dB:
#: 0.03 is about -30 dBFS, comfortably below normalized dialogue and above
#: room tone, so the bed ducks for speech and not for breathing.
DUCK_THRESHOLD = 0.03
DUCK_RATIO = 8.0
#: Fast enough that the bed is already out of the way by the first syllable.
DUCK_ATTACK_MS = 5.0
#: Slow enough that it does not pump between words. It should swell back in
#: over a real pause, not over a comma.
DUCK_RELEASE_MS = 400.0

DEFAULT_FADE_IN_S = 0.75
DEFAULT_FADE_OUT_S = 1.5

#: What a music file may be. Checked by extension rather than by probing,
#: because this runs over a directory the user drops files into by hand.
MUSIC_SUFFIXES = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")

_JSON_BLOCK = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class LoudnessStats:
    """What pass 1 measured. Fed back into pass 2 verbatim."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> LoudnessStats:
        def number(key: str, default: float = 0.0) -> float:
            try:
                return float(payload.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            input_i=number("input_i"),
            input_tp=number("input_tp"),
            input_lra=number("input_lra"),
            input_thresh=number("input_thresh"),
            target_offset=number("target_offset"),
        )


def _loudnorm_filter(
    *, target_lufs: float, true_peak: float, lra: float, stats: LoudnessStats | None = None
) -> str:
    parts = [f"loudnorm=I={target_lufs}", f"TP={true_peak}", f"LRA={lra}"]
    if stats is None:
        parts += ["print_format=json"]
    else:
        parts += [
            f"measured_I={stats.input_i}",
            f"measured_TP={stats.input_tp}",
            f"measured_LRA={stats.input_lra}",
            f"measured_thresh={stats.input_thresh}",
            f"offset={stats.target_offset}",
            "linear=true",
            "print_format=summary",
        ]
    return ":".join(parts)


def parse_loudnorm_stats(stderr: str) -> LoudnessStats | None:
    """Pull the measurement JSON out of pass 1's stderr.

    loudnorm prints its JSON among ordinary ffmpeg logging, so the block is
    located by regex rather than by parsing stderr as a whole. Returning
    ``None`` on a miss lets the caller fall back to single-pass, which is worse
    but not broken.
    """
    match = _JSON_BLOCK.search(stderr)
    if match is None:
        return None
    try:
        return LoudnessStats.from_json(json.loads(match.group(0)))
    except json.JSONDecodeError:
        return None


def measure(source: Path | str, *, target_lufs: float = TARGET_LUFS) -> LoudnessStats | None:
    """Pass 1: measure the whole file, produce no audio."""
    stderr = media.run(
        [
            "-i",
            str(source),
            "-af",
            _loudnorm_filter(target_lufs=target_lufs, true_peak=TARGET_TRUE_PEAK, lra=TARGET_LRA),
            "-f",
            "null",
            "-",
        ]
    )
    return parse_loudnorm_stats(stderr)


def build_filter(
    stats: LoudnessStats | None,
    *,
    target_lufs: float = TARGET_LUFS,
    highpass_hz: int = HIGHPASS_HZ,
    denoise: bool = False,
) -> str:
    """The audio filter chain, in the order the processing has to happen.

    Highpass before loudnorm: measuring loudness with rumble still in the
    signal biases the measurement, and the correction is then applied to
    material that no longer contains what it was measured on.
    """
    chain: list[str] = []
    if highpass_hz:
        chain.append(f"highpass=f={highpass_hz}")
    if denoise:
        chain.append("afftdn=nf=-25")
    chain.append(
        _loudnorm_filter(
            target_lufs=target_lufs,
            true_peak=TARGET_TRUE_PEAK,
            lra=TARGET_LRA,
            stats=stats,
        )
    )
    # loudnorm resamples to 192kHz internally; bring it back to something an
    # AAC encoder accepts without a warning.
    chain.append("aresample=48000")
    return ",".join(chain)


def normalize(
    source: Path | str,
    out_path: Path | str,
    *,
    target_lufs: float = TARGET_LUFS,
    highpass_hz: int = HIGHPASS_HZ,
    denoise: bool = False,
    two_pass: bool = True,
) -> Path:
    """Normalize ``source``'s audio to ``target_lufs`` into a PCM wav.

    Output is PCM because this file is muxed into the final encode, where the
    single AAC pass happens. Encoding to AAC here and again at export is a
    generation loss for nothing.
    """
    source = Path(source)
    out_path = Path(out_path)

    if not source.is_file():
        raise FileNotFoundError(f"no input at {source}")

    stats = measure(source, target_lufs=target_lufs) if two_pass else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    media.run(
        [
            "-y",
            "-i",
            str(source),
            "-vn",
            "-af",
            build_filter(stats, target_lufs=target_lufs, highpass_hz=highpass_hz, denoise=denoise),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            str(out_path),
        ]
    )
    return out_path


def mix_for_clip(
    source: Path | str,
    clip_dir: Path | str,
    *,
    target_lufs: float = TARGET_LUFS,
    **kwargs,
) -> Path:
    """``normalize`` into the conventional ``clips/clip_NN/mixed.wav``."""
    return normalize(source, Path(clip_dir) / MIXED_FILENAME, target_lufs=target_lufs, **kwargs)


# ---------------------------------------------------------------------------
# Music bed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MusicPlan:
    """One clip's bed: which track, how quiet, and whether speech ducks it."""

    track: Path
    bed_lufs: float = DEFAULT_BED_LUFS
    duck: bool = True
    fade_in_s: float = DEFAULT_FADE_IN_S
    fade_out_s: float = DEFAULT_FADE_OUT_S


class MusicMissing(RuntimeError):
    """Music was asked for and there is no track to use."""


def choose_track(music_dir: Path | str, name: str | None = None) -> Path | None:
    """Resolve a track from ``assets/music``.

    ``name`` may be a bare filename, a stem, or a path. Without it the
    directory is taken in sorted order so a render is reproducible - "the first
    file alphabetically" is a rule; "whatever the filesystem listed first" is
    not.
    """
    if name:
        candidate = Path(name)
        if candidate.is_file():
            return candidate
        in_dir = Path(music_dir) / name
        if in_dir.is_file():
            return in_dir
        matches = sorted(
            path
            for path in Path(music_dir).glob(f"{name}.*")
            if path.suffix.lower() in MUSIC_SUFFIXES
        )
        return matches[0] if matches else None

    directory = Path(music_dir)
    if not directory.is_dir():
        return None
    tracks = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in MUSIC_SUFFIXES
    )
    return tracks[0] if tracks else None


def build_music_filter(
    duration_s: float,
    *,
    bed_lufs: float = DEFAULT_BED_LUFS,
    duck: bool = True,
    fade_in_s: float = DEFAULT_FADE_IN_S,
    fade_out_s: float = DEFAULT_FADE_OUT_S,
) -> str:
    """The filter_complex that lays ``[1:a]`` under ``[0:a]``.

    Input 0 is the normalized speech, input 1 the music. Output label is
    ``[mix]``.

    Two details carry the whole thing:

    - ``amix`` is given ``normalize=0``. Its default divides every input by the
      number of inputs, so adding a bed would drop the dialogue 6 dB and undo
      the loudnorm pass that just ran.
    - the speech is split, one copy to the mixer and one to
      ``sidechaincompress`` as the key. The compressor's *main* input is the
      music: it is the music that gets squashed, keyed by the voice.
    """
    fade_out_start = max(0.0, duration_s - fade_out_s)
    stereo_48k = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"

    # Single-pass loudnorm here, unlike the two-pass over speech. A music bed
    # is near-constant in level, so the look-ahead has nothing to be caught out
    # by, and a second pass would mean probing the track on every render.
    bed = (
        f"[1:a]{stereo_48k},"
        f"atrim=0:{duration_s:.3f},asetpts=N/SR/TB,"
        f"loudnorm=I={bed_lufs:g}:TP={BED_TRUE_PEAK:g}:LRA={TARGET_LRA:g},"
        f"aresample=48000,"
        f"afade=t=in:st=0:d={fade_in_s:g},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_s:g}[bed]"
    )

    if duck:
        return ";".join(
            [
                f"[0:a]{stereo_48k},asplit=2[speech][key]",
                bed,
                f"[bed][key]sidechaincompress=threshold={DUCK_THRESHOLD:g}"
                f":ratio={DUCK_RATIO:g}:attack={DUCK_ATTACK_MS:g}"
                f":release={DUCK_RELEASE_MS:g}:level_sc=1[ducked]",
                "[speech][ducked]amix=inputs=2:duration=first:normalize=0[mix]",
            ]
        )

    return ";".join(
        [
            f"[0:a]{stereo_48k}[speech]",
            bed,
            "[speech][bed]amix=inputs=2:duration=first:normalize=0[mix]",
        ]
    )


def mix_music(
    speech: Path | str,
    out_path: Path | str,
    plan: MusicPlan,
    *,
    duration_s: float | None = None,
) -> Path:
    """Lay ``plan.track`` under ``speech`` and write a PCM wav.

    The track is looped with ``-stream_loop -1`` and cut back to the speech
    length, so a 90-second bed covers a 40-second clip and a 12-second one
    covers it too. Looping is audible at the seam on a track with a hard
    ending; that is a property of the track, not of the mix, and is why
    ``choose_track`` points at a directory the user curates.
    """
    speech = Path(speech)
    out_path = Path(out_path)

    if not speech.is_file():
        raise FileNotFoundError(f"no speech input at {speech}")
    if not plan.track.is_file():
        raise MusicMissing(f"no music track at {plan.track}")

    if duration_s is None:
        duration_s = media.duration_s(speech)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    media.run(
        [
            "-y",
            "-i",
            str(speech),
            "-stream_loop",
            "-1",
            "-i",
            str(plan.track),
            "-filter_complex",
            build_music_filter(
                duration_s,
                bed_lufs=plan.bed_lufs,
                duck=plan.duck,
                fade_in_s=plan.fade_in_s,
                fade_out_s=plan.fade_out_s,
            ),
            "-map",
            "[mix]",
            "-t",
            f"{duration_s:.3f}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            str(out_path),
        ]
    )
    return out_path


def plan_from_config(
    edit_config: dict[str, Any],
    *,
    music_dir: Path | str,
    override: str | None = None,
    disabled: bool = False,
) -> MusicPlan | None:
    """Read the ``edit:`` block into a plan, or ``None`` for no music.

    ``override`` is the CLI's ``--music``: naming a track turns the bed on for
    that render even when the config leaves it off, because that is the only
    reason to type it. ``disabled`` is ``--no-music`` and wins over both.
    """
    if disabled:
        return None
    if override is None and not bool(edit_config.get("music_enabled", False)):
        return None

    name = override if override is not None else edit_config.get("music_track")
    # music_bed_db was the original key, when the level was a gain. Still read,
    # because a config written against it means the same thing by the number.
    bed_lufs = edit_config.get("music_bed_lufs", edit_config.get("music_bed_db", DEFAULT_BED_LUFS))
    track = choose_track(music_dir, name)
    if track is None:
        where = name or f"{music_dir}/"
        raise MusicMissing(
            f"music was requested but no track was found at {where}. "
            f"Drop an audio file in {music_dir} or pass --music <path>."
        )

    return MusicPlan(
        track=track,
        bed_lufs=float(bed_lufs),
        duck=bool(edit_config.get("music_duck", True)),
        fade_in_s=float(edit_config.get("music_fade_in_s", DEFAULT_FADE_IN_S)),
        fade_out_s=float(edit_config.get("music_fade_out_s", DEFAULT_FADE_OUT_S)),
    )


__all__ = [
    "BED_TRUE_PEAK",
    "DEFAULT_BED_LUFS",
    "DEFAULT_FADE_IN_S",
    "DEFAULT_FADE_OUT_S",
    "DUCK_ATTACK_MS",
    "DUCK_RATIO",
    "DUCK_RELEASE_MS",
    "DUCK_THRESHOLD",
    "HIGHPASS_HZ",
    "MIXED_FILENAME",
    "MUSIC_MIXED_FILENAME",
    "MUSIC_SUFFIXES",
    "TARGET_LRA",
    "TARGET_LUFS",
    "TARGET_TRUE_PEAK",
    "LoudnessStats",
    "MusicMissing",
    "MusicPlan",
    "build_filter",
    "build_music_filter",
    "choose_track",
    "measure",
    "mix_for_clip",
    "mix_music",
    "normalize",
    "parse_loudnorm_stats",
    "plan_from_config",
]
