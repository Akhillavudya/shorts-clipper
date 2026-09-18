"""The one internal transcript format (FULL_PLAN.md section 4).

Responsibility: define the schema every transcription backend normalizes into,
so no downstream stage knows or cares which backend produced it.

    {
      "words": [{"w": "actually", "start": 412.34, "end": 412.71,
                 "speaker": "SPEAKER_01"}],
      "segments": [{"start": 410.2, "end": 438.9, "speaker": "SPEAKER_01",
                    "text": "..."}]
    }

Word-level timestamps are non-negotiable: they drive both cut boundaries
(moments/snap.py) and karaoke caption timing (edit/captions.py).

The other invariant worth stating out loud: **word index is the pipeline's
addressing scheme**. The LLM returns indices into ``Transcript.words`` and
never seconds (CLAUDE.md hard rule 2), so that list must stay stable and
densely indexed from the moment it is written. :meth:`Transcript.normalized`
enforces that on the way in - sorted, clamped, no zero-length words - because
a backend that emits one out-of-order word would otherwise silently shift
every cut point after it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

TRANSCRIPT_FILENAME = "transcript.json"

#: Terminators used to find a sentence start when extending a cut backwards.
SENTENCE_END_CHARS = ".!?"

#: A word with no measurable duration breaks karaoke timing, so a floor is
#: applied on load. 10ms is below perception and above zero.
MIN_WORD_DURATION_S = 0.01


class Word(BaseModel):
    """One word with its own timing. The atom the whole pipeline addresses."""

    model_config = ConfigDict(populate_by_name=True)

    w: str
    start: float
    end: float
    speaker: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end - self.start

    @property
    def ends_sentence(self) -> bool:
        """Whether this word carries terminal punctuation.

        Used by ``moments/snap.py`` to extend a span backwards to the start of
        a sentence, so a clip does not open mid-thought.
        """
        return self.w.rstrip("\"”')]").endswith(tuple(SENTENCE_END_CHARS))


class Segment(BaseModel):
    """A speaker turn, or a backend's natural chunk. Convenience, not truth.

    Nothing downstream cuts on a segment boundary - words are the source of
    truth. Segments exist so a transcript is readable and so the recall prompt
    can show speaker changes without reconstructing them per word.
    """

    start: float
    end: float
    speaker: str | None = None
    text: str = ""

    @property
    def duration_s(self) -> float:
        return self.end - self.start


class Transcript(BaseModel):
    """A whole transcript, plus which backend produced it."""

    words: list[Word] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    language: str | None = None
    backend: str | None = None
    duration_s: float | None = None

    # -- construction ---------------------------------------------------

    def normalized(self) -> Transcript:
        """Return a copy with the invariants downstream stages rely on.

        Sorted by start time, non-decreasing, every word at least
        ``MIN_WORD_DURATION_S`` long, blank words dropped. Backends call this
        before writing; nothing else should have to defend against a ragged
        word list.
        """
        kept = [w for w in self.words if w.w and w.w.strip()]
        kept.sort(key=lambda w: (w.start, w.end))

        fixed: list[Word] = []
        previous_end = 0.0
        for word in kept:
            start = max(word.start, 0.0)
            end = max(word.end, start + MIN_WORD_DURATION_S)
            # A backend that overlaps two words would otherwise make a caption
            # highlight two at once; nudge the later one forward instead.
            if start < previous_end:
                start = previous_end
                end = max(end, start + MIN_WORD_DURATION_S)
            fixed.append(Word(w=word.w.strip(), start=start, end=end, speaker=word.speaker))
            previous_end = end

        segments = sorted(self.segments, key=lambda s: s.start)
        duration = self.duration_s
        if duration is None and fixed:
            duration = fixed[-1].end

        return Transcript(
            words=fixed,
            segments=segments,
            language=self.language,
            backend=self.backend,
            duration_s=duration,
        )

    def with_segments_from_words(self, *, max_gap_s: float = 1.0) -> Transcript:
        """Fill in ``segments`` by grouping words on speaker and pauses.

        For a backend that returns words but no turns. Grouping breaks on a
        speaker change or a pause longer than ``max_gap_s``.
        """
        if self.segments or not self.words:
            return self

        segments: list[Segment] = []
        run: list[Word] = []

        def flush() -> None:
            if not run:
                return
            segments.append(
                Segment(
                    start=run[0].start,
                    end=run[-1].end,
                    speaker=run[0].speaker,
                    text=" ".join(w.w for w in run),
                )
            )

        for word in self.words:
            if run and (word.speaker != run[0].speaker or word.start - run[-1].end > max_gap_s):
                flush()
                run = []
            run.append(word)
        flush()

        return self.model_copy(update={"segments": segments})

    # -- queries --------------------------------------------------------

    @property
    def word_count(self) -> int:
        return len(self.words)

    def text(self) -> str:
        return " ".join(w.w for w in self.words)

    def text_for(self, start_index: int, end_index: int) -> str:
        """Plain text for an inclusive word-index span."""
        return " ".join(w.w for w in self.slice(start_index, end_index))

    def slice(self, start_index: int, end_index: int) -> list[Word]:
        """Words for an inclusive index span, clamped to the transcript.

        Inclusive because the model is asked for ``end_word_index`` as the last
        word of the moment, not one past it. Getting that wrong drops the final
        word of every clip, which is exactly the kind of off-by-one that is
        invisible until you watch the output.
        """
        if not self.words:
            return []
        lo = max(0, min(start_index, len(self.words) - 1))
        hi = max(lo, min(end_index, len(self.words) - 1))
        return self.words[lo : hi + 1]

    def span_seconds(self, start_index: int, end_index: int) -> tuple[float, float]:
        """Raw ``(start, end)`` seconds for a word span, before any snapping.

        ``moments/snap.py`` takes it from here. Deliberately not called
        anywhere that writes a cut point directly.
        """
        words = self.slice(start_index, end_index)
        if not words:
            return (0.0, 0.0)
        return (words[0].start, words[-1].end)

    def index_at(self, seconds: float) -> int:
        """Index of the word being spoken at ``seconds`` (or the next one)."""
        for i, word in enumerate(self.words):
            if word.end >= seconds:
                return i
        return max(0, len(self.words) - 1)

    def speakers(self) -> list[str]:
        """Distinct speaker labels, in first-appearance order."""
        seen: list[str] = []
        for word in self.words:
            if word.speaker and word.speaker not in seen:
                seen.append(word.speaker)
        return seen

    def has_speaker_change(self, start_index: int, end_index: int) -> bool:
        """Whether a span crosses a speaker turn.

        ``moments/score.py`` applies a penalty for this: a span with two
        speakers is harder to reframe cleanly onto one face.
        """
        labels = {w.speaker for w in self.slice(start_index, end_index) if w.speaker}
        return len(labels) > 1

    # -- io -------------------------------------------------------------

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=False)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> Transcript:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no transcript at {path} - run `clipper transcribe` first")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Transcript:
        return cls.model_validate(payload)


def transcript_path(job_dir: Path | str) -> Path:
    return Path(job_dir) / TRANSCRIPT_FILENAME


def load_transcript(job_dir: Path | str) -> Transcript:
    """Load ``jobs/<video_id>/transcript.json``."""
    return Transcript.load(transcript_path(job_dir))


def dump_words(words: list[Word]) -> list[dict[str, Any]]:
    """Words as plain dicts, for embedding in another JSON artefact."""
    return [w.model_dump() for w in words]


__all__ = [
    "MIN_WORD_DURATION_S",
    "SENTENCE_END_CHARS",
    "TRANSCRIPT_FILENAME",
    "Segment",
    "Transcript",
    "Word",
    "dump_words",
    "load_transcript",
    "transcript_path",
]
