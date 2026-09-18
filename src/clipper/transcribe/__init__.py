"""Transcription backends, and the one schema they all normalize into.

Which backend ran is a config choice (``transcription.backend``) or a CLI flag.
Nothing downstream is allowed to care: both backends return a
:class:`~clipper.transcribe.schema.Transcript`, and that is the only shape the
rest of the pipeline knows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clipper.transcribe.schema import (
    TRANSCRIPT_FILENAME,
    Segment,
    Transcript,
    Word,
    load_transcript,
    transcript_path,
)

#: Backend name -> the module implementing it. Import is deferred inside
#: :func:`transcribe`, because whisperx pulls torch and gemini pulls the Google
#: SDK; importing this package must not drag in either.
BACKENDS = ("gemini", "whisperx")


def transcribe(
    audio: Path | str,
    settings: dict[str, Any],
    *,
    backend: str | None = None,
    on_progress: Any = None,
    **kwargs: Any,
) -> Transcript:
    """Transcribe ``audio`` with the configured backend.

    ``backend`` overrides ``transcription.backend`` in settings.yaml.
    """
    name = (backend or (settings.get("transcription") or {}).get("backend") or "gemini").lower()

    if name == "gemini":
        from clipper.transcribe import gemini

        return gemini.transcribe(audio, settings, on_progress=on_progress, **kwargs)

    if name in {"whisperx", "whisperx_local", "local"}:
        from clipper.transcribe import whisperx_local

        return whisperx_local.transcribe(audio, settings, on_progress=on_progress)

    raise ValueError(
        f"unknown transcription backend {name!r}; expected one of {', '.join(BACKENDS)}"
    )


__all__ = [
    "BACKENDS",
    "TRANSCRIPT_FILENAME",
    "Segment",
    "Transcript",
    "Word",
    "load_transcript",
    "transcribe",
    "transcript_path",
]
