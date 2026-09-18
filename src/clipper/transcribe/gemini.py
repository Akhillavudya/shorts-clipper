"""Cloud transcription backend (FULL_PLAN.md sections 2 and 4).

Responsibility: transcribe audio.wav with word-level timestamps and speaker
diarization, then normalize the result into transcribe/schema.py.

The default starting backend: diarization is built in and the cost is pennies
per episode, so there is nothing to set up before the pipeline works end to
end. Move to whisperx_local once a GPU is available and marginal cost matters.

Two things this module works around, both worth knowing before trusting its
output:

- **Output length, not audio length, is the binding constraint.** A 2-hour
  episode is ~20k words, and one word object is ~40 tokens of JSON. Asked for
  the whole thing in one response the model runs into ``max_output_tokens``
  and returns *valid JSON that stops halfway*, which is far worse than an
  error because it looks like a short episode. So audio is split into windows
  (``transcription.gemini.chunk_minutes``), each transcribed on its own, and
  timestamps are offset back onto the episode timeline.

- **Word timings are approximate.** These are timings a model reports, not
  forced alignment, so they drift - typically tens of milliseconds, sometimes
  more after a long pause. Good enough for cut boundaries, which
  ``moments/snap.py`` re-snaps to detected silence anyway. Marginal for
  karaoke captions. When a caption looks out of sync, that is this, and
  ``whisperx_local`` is the fix.

The router is not used here: it is a text-only interface and this call carries
audio. The model id still comes from ``config/settings.yaml`` (CLAUDE.md hard
rule 4) - no model string appears below.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper import media
from clipper.transcribe.schema import Segment, Transcript, Word

BACKEND_NAME = "gemini"

SYSTEM_INSTRUCTION = """\
You are a transcription engine. You return JSON only.

Transcribe the audio verbatim with per-word timings and speaker labels.

Rules:
- Every spoken word gets its own object with a start and end time in SECONDS
  from the beginning of THIS audio clip, as a decimal number.
- Times must increase monotonically and never overlap.
- Keep natural punctuation and capitalisation attached to the word it belongs
  to ("really." not "really" + "."). Sentence boundaries are used downstream to
  decide where a clip starts, so terminal punctuation matters.
- Label speakers SPEAKER_00, SPEAKER_01, ... in order of first appearance, and
  keep a label consistent for the whole clip.
- Do not summarise, translate, censor, or add commentary. No filler-word
  removal - "um" and "you know" are real words with real timings.
- If a stretch is inaudible, emit the single word "[inaudible]" spanning it.
"""

USER_INSTRUCTION = """\
Transcribe this audio clip in full, following the rules exactly.
The clip is {duration:.1f} seconds long. Timestamps start at 0 for this clip.
"""

#: Native structured output. Segments are requested alongside words because
#: asking for the grouping directly is cheaper and more consistent than
#: reconstructing turns from per-word labels afterwards.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "w": {"type": "string"},
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                            },
                            "required": ["w", "start", "end"],
                        },
                    },
                },
                "required": ["start", "end", "text", "words"],
            },
        },
    },
    "required": ["segments"],
}


class TranscriptionError(RuntimeError):
    """The backend could not produce a usable transcript."""


@dataclass(frozen=True)
class GeminiConfig:
    """The ``transcription.gemini`` block of settings.yaml."""

    model: str
    api_key_env: str = "GEMINI_API_KEY"
    chunk_minutes: float = 10.0
    max_attempts: int = 3
    timeout_s: float = 600.0

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> GeminiConfig:
        block = ((settings.get("transcription") or {}).get("gemini")) or {}
        model = block.get("model")
        if not model:
            raise TranscriptionError(
                "config/settings.yaml has no transcription.gemini.model. Model "
                "ids live in config, never in src/ (CLAUDE.md hard rule 4)."
            )
        return cls(
            model=str(model),
            api_key_env=str(block.get("api_key_env", "GEMINI_API_KEY")),
            chunk_minutes=float(block.get("chunk_minutes", 10)),
            max_attempts=int(block.get("max_attempts", 3)),
            timeout_s=float(block.get("timeout_s", 600)),
        )


def plan_windows(total_s: float, chunk_s: float) -> list[tuple[float, float]]:
    """Split ``total_s`` into consecutive windows of at most ``chunk_s``.

    No overlap, unlike the recall pass. Overlapping audio windows would emit
    the same words twice with two sets of timings, and de-duplicating text by
    fuzzy match is a worse problem than the one it solves. A word clipped at a
    window edge costs one word; a duplicated span corrupts every index after
    it, and indices are how the LLM addresses cut points.
    """
    if total_s <= 0:
        return []
    if chunk_s <= 0 or total_s <= chunk_s:
        return [(0.0, total_s)]

    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < total_s - 0.05:
        end = min(start + chunk_s, total_s)
        # Absorb a stub tail rather than making a 4-second final call.
        if total_s - end < chunk_s * 0.1:
            end = total_s
        windows.append((start, end))
        start = end
    return windows


def split_audio(
    audio: Path | str, out_dir: Path | str, windows: list[tuple[float, float]]
) -> list[Path]:
    """Write one wav per window. Re-encoded PCM, so cuts land exactly."""
    audio = Path(audio)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    for i, (start, end) in enumerate(windows):
        part = out_dir / f"chunk_{i:03d}.wav"
        media.run(
            [
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(audio),
                "-t",
                f"{end - start:.3f}",
                "-acodec",
                "pcm_s16le",
                str(part),
            ]
        )
        parts.append(part)
    return parts


def _finish_reason(raw: Any) -> str:
    """The first candidate's finish reason, as an uppercase name.

    Tolerates every shape the SDK has used for it - an enum, an enum member's
    name, a bare string - because the only thing read off it is whether it says
    MAX_TOKENS.
    """
    candidates = getattr(raw, "candidates", None) or []
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return ""
    return str(getattr(reason, "name", reason)).upper()


def _extract_json(raw: Any) -> dict[str, Any]:
    # The failure this guards against is the one in the module docstring: a
    # window that overran max_output_tokens comes back as a *shorter
    # transcript*, not as an error, and the missing minutes are invisible from
    # here on. Structured output usually makes it a JSON parse failure instead,
    # but not reliably enough to depend on.
    if _finish_reason(raw).endswith("MAX_TOKENS"):
        raise TranscriptionError(
            "gemini hit max_output_tokens transcribing a window, so the transcript "
            "for it is incomplete. Lower transcription.gemini.chunk_minutes and "
            "rerun with --force."
        )

    text = getattr(raw, "text", None) or ""
    if not text.strip():
        raise TranscriptionError("gemini returned an empty transcription response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise TranscriptionError(f"response is not JSON: {text[:200]!r}") from exc
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise TranscriptionError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def parse_response(
    payload: dict[str, Any], *, offset_s: float = 0.0, window_s: float | None = None
) -> tuple[list[Word], list[Segment], str | None]:
    """Turn one window's JSON into schema objects on the episode timeline.

    Times that fall outside the window are clamped rather than dropped: a model
    that reports 601.2s in a 600s window has drifted, not hallucinated, and
    losing the last sentence of every window is the worse failure.
    """
    words: list[Word] = []
    segments: list[Segment] = []

    def shift(value: Any, fallback: float) -> float:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = fallback
        seconds = max(0.0, seconds)
        if window_s is not None:
            seconds = min(seconds, window_s)
        return seconds + offset_s

    for raw_segment in payload.get("segments") or []:
        if not isinstance(raw_segment, dict):
            continue
        speaker = raw_segment.get("speaker") or None
        seg_start = shift(raw_segment.get("start"), 0.0)
        seg_end = shift(raw_segment.get("end"), 0.0)

        segment_words: list[Word] = []
        for raw_word in raw_segment.get("words") or []:
            if not isinstance(raw_word, dict):
                continue
            text = str(raw_word.get("w") or "").strip()
            if not text:
                continue
            start = shift(raw_word.get("start"), seg_start - offset_s)
            end = shift(raw_word.get("end"), start - offset_s)
            segment_words.append(Word(w=text, start=start, end=max(end, start), speaker=speaker))

        words.extend(segment_words)
        text = str(raw_segment.get("text") or "").strip()
        if not text and segment_words:
            text = " ".join(w.w for w in segment_words)
        if text:
            segments.append(
                Segment(
                    start=segment_words[0].start if segment_words else seg_start,
                    end=segment_words[-1].end if segment_words else seg_end,
                    speaker=speaker,
                    text=text,
                )
            )

    language = payload.get("language")
    return words, segments, (str(language) if language else None)


class GeminiTranscriber:
    """Transcribes audio with the Gemini file API.

    The client is injectable so tests can drive the chunking, parsing and
    offset arithmetic without a network or a key - which is the only way to
    test them, since no test may hit a real API.
    """

    backend = BACKEND_NAME

    def __init__(
        self,
        config: GeminiConfig,
        *,
        client: Any = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._client = client
        # Injected, never called for real in tests: the suite must not spend
        # wall-clock seconds proving that a retry backs off.
        self._sleep = sleep_fn

    @classmethod
    def from_settings(
        cls,
        settings: dict[str, Any],
        *,
        client: Any = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> GeminiTranscriber:
        return cls(GeminiConfig.from_settings(settings), client=client, sleep_fn=sleep_fn)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise TranscriptionError(
                f"{self.config.api_key_env} is not set; cannot transcribe with "
                f"gemini. Set it in .env, or use --backend whisperx."
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TranscriptionError("google-genai is not installed") from exc
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _transcribe_window(self, part: Path, duration_s: float) -> dict[str, Any]:
        """One window, with retries. Returns the parsed JSON payload."""
        client = self._ensure_client()
        prompt = USER_INSTRUCTION.format(duration=duration_s)
        config = {
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0.0,
            "http_options": {"timeout": int(self.config.timeout_s * 1000)},
        }

        last: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                uploaded = client.files.upload(file=str(part))
                raw = client.models.generate_content(
                    model=self.config.model,
                    contents=[prompt, uploaded],
                    config=config,
                )
                return _extract_json(raw)
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
                last = exc
                if attempt < self.config.max_attempts:
                    self._sleep(min(2.0**attempt, 30.0))
        raise TranscriptionError(
            f"gemini transcription failed for {part.name} after "
            f"{self.config.max_attempts} attempts: {last}"
        ) from last

    def transcribe(
        self,
        audio: Path | str,
        *,
        work_dir: Path | str | None = None,
        total_s: float | None = None,
        on_progress: Any = None,
    ) -> Transcript:
        """Transcribe ``audio`` end to end and return a normalized transcript."""
        audio = Path(audio)
        if not audio.is_file():
            raise TranscriptionError(f"no audio at {audio}")

        if total_s is None:
            total_s = media.duration_s(audio)
        windows = plan_windows(total_s, self.config.chunk_minutes * 60.0)
        if not windows:
            raise TranscriptionError(f"{audio} has no measurable duration")

        work_dir = Path(work_dir or audio.parent / ".transcribe")
        parts = [audio] if len(windows) == 1 else split_audio(audio, work_dir, windows)

        all_words: list[Word] = []
        all_segments: list[Segment] = []
        language: str | None = None

        for i, (part, (start, end)) in enumerate(zip(parts, windows, strict=True)):
            if on_progress is not None:
                on_progress(i + 1, len(windows), start, end)
            payload = self._transcribe_window(part, end - start)
            words, segments, lang = parse_response(payload, offset_s=start, window_s=end - start)
            all_words.extend(words)
            all_segments.extend(segments)
            language = language or lang

        if not all_words:
            raise TranscriptionError(
                "gemini returned no words. Word-level timestamps are "
                "non-negotiable (CLAUDE.md hard rule 7); a transcript without "
                "them cannot drive cut boundaries or captions."
            )

        return Transcript(
            words=all_words,
            segments=all_segments,
            language=language,
            backend=BACKEND_NAME,
            duration_s=total_s,
        ).normalized()


def transcribe(
    audio: Path | str,
    settings: dict[str, Any],
    *,
    client: Any = None,
    work_dir: Path | str | None = None,
    on_progress: Any = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Transcript:
    """Module-level entrypoint, matching ``whisperx_local.transcribe``."""
    return GeminiTranscriber.from_settings(settings, client=client, sleep_fn=sleep_fn).transcribe(
        audio, work_dir=work_dir, on_progress=on_progress
    )


__all__ = [
    "BACKEND_NAME",
    "RESPONSE_SCHEMA",
    "SYSTEM_INSTRUCTION",
    "GeminiConfig",
    "GeminiTranscriber",
    "TranscriptionError",
    "parse_response",
    "plan_windows",
    "split_audio",
    "transcribe",
]
