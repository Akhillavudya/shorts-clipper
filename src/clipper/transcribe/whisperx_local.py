"""Local transcription backend (FULL_PLAN.md sections 2 and 4).

Responsibility: same contract as transcribe/gemini.py, zero marginal cost.

WhisperX = faster-whisper + wav2vec2 forced alignment + pyannote diarization,
giving sub-100ms word timestamps, which is accurate enough for karaoke
captions. Needs an NVIDIA GPU with >=6GB VRAM; whisper.cpp is the Mac
equivalent.

Parakeet TDT 0.6B v3 is a later optimization, not a starting point:
English-only, but faster and slightly more accurate than Whisper large-v3
(6.32% vs 7.44% WER).

whisperx is deliberately **not** a declared dependency. It pulls torch and a
CUDA stack - gigabytes that are dead weight on a machine without a GPU, and
this project's default path is the cloud backend. Install it yourself when
there is a GPU worth pointing at it:

    uv pip install whisperx

The three-step shape below (transcribe, align, diarize) is what makes the
timings usable. Plain Whisper emits segment-level timings only; the alignment
pass is where per-word times come from, and skipping it produces a transcript
that satisfies the schema but drives visibly wrong captions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper.transcribe.schema import Segment, Transcript, Word

BACKEND_NAME = "whisperx"


class WhisperXUnavailable(RuntimeError):
    """whisperx is not installed, or its model weights cannot be loaded."""


@dataclass(frozen=True)
class WhisperXConfig:
    """The ``transcription.whisperx`` block of settings.yaml."""

    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "float16"
    batch_size: int = 16
    diarize: bool = True
    hf_token_env: str = "HF_TOKEN"
    language: str | None = None

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> WhisperXConfig:
        block = ((settings.get("transcription") or {}).get("whisperx")) or {}
        return cls(
            model=str(block.get("model", "large-v3")),
            device=str(block.get("device", "auto")),
            compute_type=str(block.get("compute_type", "float16")),
            batch_size=int(block.get("batch_size", 16)),
            diarize=bool(block.get("diarize", True)),
            hf_token_env=str(block.get("hf_token_env", "HF_TOKEN")),
            language=block.get("language"),
        )

    def resolved_device(self) -> str:
        """``auto`` means CUDA when it is actually there, CPU otherwise."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolved_compute_type(self, device: str) -> str:
        """float16 is a GPU-only claim; on CPU it silently costs a lot."""
        if device == "cpu" and self.compute_type == "float16":
            return "int8"
        return self.compute_type


def _import_whisperx() -> Any:
    try:
        import whisperx
    except ImportError as exc:
        raise WhisperXUnavailable(
            "whisperx is not installed. It is not a declared dependency "
            "because it pulls a multi-gigabyte CUDA stack.\n"
            "  uv pip install whisperx\n"
            "Or transcribe with the cloud backend: --backend gemini"
        ) from exc
    return whisperx


def parse_result(result: dict[str, Any], *, duration_s: float | None = None) -> Transcript:
    """Normalize whisperx's aligned output into the shared schema.

    Handles the post-alignment shape: ``{"segments": [{"start", "end", "text",
    "words": [{"word", "start", "end", "speaker"}]}]}``. Alignment can leave a
    word without timings when it fails to match audio (numerals and
    non-speech markers, mostly); such a word inherits its neighbours' bounds
    rather than being dropped, because dropping it would shift every index
    after it.
    """
    words: list[Word] = []
    segments: list[Segment] = []

    for raw_segment in result.get("segments") or []:
        seg_speaker = raw_segment.get("speaker")
        seg_start = float(raw_segment.get("start") or 0.0)
        seg_end = float(raw_segment.get("end") or seg_start)

        segment_words: list[Word] = []
        cursor = seg_start
        for raw_word in raw_segment.get("words") or []:
            text = str(raw_word.get("word") or raw_word.get("w") or "").strip()
            if not text:
                continue
            start = raw_word.get("start")
            end = raw_word.get("end")
            start_f = float(start) if start is not None else cursor
            end_f = float(end) if end is not None else start_f
            cursor = max(cursor, end_f)
            segment_words.append(
                Word(
                    w=text,
                    start=start_f,
                    end=max(end_f, start_f),
                    speaker=raw_word.get("speaker") or seg_speaker,
                )
            )

        words.extend(segment_words)
        text = str(raw_segment.get("text") or "").strip()
        if not text and segment_words:
            text = " ".join(w.w for w in segment_words)
        if text:
            segments.append(
                Segment(
                    start=segment_words[0].start if segment_words else seg_start,
                    end=segment_words[-1].end if segment_words else seg_end,
                    speaker=seg_speaker,
                    text=text,
                )
            )

    return Transcript(
        words=words,
        segments=segments,
        language=result.get("language"),
        backend=BACKEND_NAME,
        duration_s=duration_s,
    ).normalized()


def transcribe(
    audio: Path | str,
    settings: dict[str, Any],
    *,
    on_progress: Any = None,
) -> Transcript:
    """Transcribe, force-align, and diarize locally.

    Diarization is best-effort: pyannote needs an accepted model licence and a
    Hugging Face token, and a missing token should cost speaker labels, not the
    whole transcript. Word timings - the non-negotiable part - come from the
    alignment pass, which needs neither.
    """
    audio_path = Path(audio)
    if not audio_path.is_file():
        raise WhisperXUnavailable(f"no audio at {audio_path}")

    whisperx = _import_whisperx()
    config = WhisperXConfig.from_settings(settings)
    device = config.resolved_device()
    compute_type = config.resolved_compute_type(device)

    if on_progress is not None:
        on_progress("load", f"{config.model} on {device} ({compute_type})")

    audio_data = whisperx.load_audio(str(audio_path))
    model = whisperx.load_model(config.model, device, compute_type=compute_type)
    result = model.transcribe(audio_data, batch_size=config.batch_size, language=config.language)

    if on_progress is not None:
        on_progress("align", result.get("language", "?"))

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio_data,
        device,
        return_char_alignments=False,
    )

    if config.diarize:
        token = os.environ.get(config.hf_token_env)
        if not token:
            if on_progress is not None:
                on_progress(
                    "diarize",
                    f"skipped - {config.hf_token_env} is not set. Word timings "
                    f"are unaffected; clips will have no speaker labels.",
                )
        else:
            if on_progress is not None:
                on_progress("diarize", "pyannote")
            diarizer = whisperx.DiarizationPipeline(use_auth_token=token, device=device)
            result = whisperx.assign_word_speakers(diarizer(audio_data), result)

    duration = len(audio_data) / 16000.0 if hasattr(audio_data, "__len__") else None
    return parse_result(result, duration_s=duration)


__all__ = [
    "BACKEND_NAME",
    "WhisperXConfig",
    "WhisperXUnavailable",
    "parse_result",
    "transcribe",
]
