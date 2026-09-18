"""Audio extraction and pre-transcription normalization (FULL_PLAN.md section 2).

Responsibility: produce jobs/<video_id>/audio.wav from the source video in the
shape ASR wants - 16 kHz mono PCM - and nothing more.

Cosmetic audio processing for the final render belongs in edit/audio_mix.py;
doing it here would change what the transcriber hears.

"Nothing more" is the whole point. It is tempting to denoise or compress here
because the transcript would read better, but the transcript is not the
product - the cut points are. Loudness normalization in particular shifts
nothing audible to a model and costs a second pass over the file, while a
noise gate can swallow the exact quiet moment that ``moments/snap.py`` wants
to cut on.
"""

from __future__ import annotations

from pathlib import Path

from clipper import media

AUDIO_FILENAME = "audio.wav"

#: What every ASR backend in use here wants. Whisper and its derivatives
#: resample to 16 kHz mono internally regardless, so feeding anything richer
#: just moves the resample and costs disk.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1


def audio_path(job_dir: Path | str) -> Path:
    return Path(job_dir) / AUDIO_FILENAME


def extract(
    source: Path | str,
    out_path: Path | str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    overwrite: bool = True,
) -> Path:
    """Extract ``source``'s audio to a PCM wav at ``out_path``.

    ``pcm_s16le`` rather than a compressed codec: this file is read by the
    transcriber and by ``silencedetect``, and a lossy round-trip would move
    silence boundaries that later become cut points.
    """
    source = Path(source)
    out_path = Path(out_path)

    if not source.is_file():
        raise FileNotFoundError(f"no source media at {source}")

    if out_path.is_file() and out_path.stat().st_size > 0 and not overwrite:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    media.run(
        [
            "-y",
            "-i",
            str(source),
            "-vn",
            "-map",
            "a:0",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(out_path),
        ]
    )

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise media.FFmpegError([str(source)], 1, f"ffmpeg produced no audio at {out_path}")
    return out_path


def extract_for_job(
    source: Path | str,
    job_dir: Path | str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    overwrite: bool = True,
) -> Path:
    """``extract`` into the conventional ``jobs/<id>/audio.wav`` location."""
    return extract(
        source,
        audio_path(job_dir),
        sample_rate=sample_rate,
        channels=channels,
        overwrite=overwrite,
    )


def require_audio(job_dir: Path | str) -> Path:
    """Return ``audio.wav`` for a job, or explain which stage is missing."""
    path = audio_path(job_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"no audio at {path} - run `clipper ingest` first (it extracts audio)"
        )
    return path


__all__ = [
    "AUDIO_FILENAME",
    "DEFAULT_CHANNELS",
    "DEFAULT_SAMPLE_RATE",
    "audio_path",
    "extract",
    "extract_for_job",
    "require_audio",
]
