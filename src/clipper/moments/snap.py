"""Deterministic boundary snapping (FULL_PLAN.md section 5).

Responsibility: convert the model's word indices into cut timestamps that do
not sound cheap. No LLM involved.

    word indices -> timestamps
                 -> extend backwards to the start of the sentence
                 -> snap outward to the nearest silence gap (>250ms, found with
                    the ffmpeg silencedetect filter)
                 -> pad 150ms each side

This is what kills the clipped-first-syllable artefact that makes automated
clips instantly recognizable.

This module is the enforcement point for CLAUDE.md hard rule 2: the LLM never
emits cut points. Every timestamp anywhere downstream traces back to this
function, and it is pure - words in, seconds out - so the same transcript and
the same indices always produce the same cut, and a bad boundary can be
reproduced without spending a token.

Order matters and is not arbitrary. Sentence extension runs first because it
moves the boundary the furthest; snapping to silence then refines whatever
that landed on. Reversing them snaps to a gap mid-sentence and then extends
past it, which throws the snap away.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from clipper.transcribe.schema import Transcript, Word

#: How far back sentence extension will look for a sentence start, in words.
#: Past this the "sentence" is a run-on and extending further does more harm
#: than opening mid-thought would.
MAX_SENTENCE_LOOKBACK_WORDS = 60

#: How far from a boundary to accept a silence, in seconds. Beyond this the
#: nearest gap belongs to a different sentence, and snapping to it would swallow
#: unrelated speech.
SILENCE_SEARCH_S = 1.2


@dataclass(frozen=True)
class Snapped:
    """A resolved cut, with enough provenance to debug a bad boundary."""

    start: float
    end: float
    start_word_index: int
    end_word_index: int
    snapped_start_to_silence: bool = False
    snapped_end_to_silence: bool = False
    extended_to_sentence: bool = False

    @property
    def duration_s(self) -> float:
        return self.end - self.start


def extend_to_sentence_start(
    words: Sequence[Word], start_index: int, *, max_lookback: int = MAX_SENTENCE_LOOKBACK_WORDS
) -> int:
    """Walk back to the first word of the sentence containing ``start_index``.

    Stops at a word ending in terminal punctuation, at a speaker change, or
    after ``max_lookback`` words. The speaker stop matters: a sentence does not
    span a turn, and extending across one opens the clip on a different
    person's half-finished line.
    """
    if not words:
        return start_index

    index = max(0, min(start_index, len(words) - 1))
    speaker = words[index].speaker
    floor = max(0, index - max_lookback)

    while index > floor:
        previous = words[index - 1]
        if previous.ends_sentence:
            break
        if previous.speaker != speaker:
            break
        index -= 1

    return index


def nearest_silence_edge(
    silences: Sequence[tuple[float, float]],
    time_s: float,
    *,
    direction: str,
    search_s: float = SILENCE_SEARCH_S,
) -> float | None:
    """Find a silence to move ``time_s`` onto, or ``None`` if there is none near.

    ``direction`` is ``"before"`` for a clip start and ``"after"`` for a clip
    end - both move the boundary *outward*, widening the clip. Widening into
    silence adds a beat of air; narrowing would eat a syllable, which is the
    exact artefact this module exists to prevent.

    A start lands on the *end* of the silence that precedes it (speech begins
    right after), and an end lands on the *start* of the silence that follows
    it (speech has just stopped).
    """
    if not silences:
        return None

    if direction == "before":
        candidates = [end for _, end in silences if 0 <= time_s - end <= search_s]
        return max(candidates) if candidates else None

    if direction == "after":
        candidates = [start for start, _ in silences if 0 <= start - time_s <= search_s]
        return min(candidates) if candidates else None

    raise ValueError(f"direction must be 'before' or 'after', got {direction!r}")


def snap(
    transcript: Transcript,
    start_index: int,
    end_index: int,
    *,
    silences: Sequence[tuple[float, float]] = (),
    pad_ms: int = 150,
    max_lookback: int = MAX_SENTENCE_LOOKBACK_WORDS,
    search_s: float = SILENCE_SEARCH_S,
    max_duration_s: float | None = None,
) -> Snapped:
    """Turn an inclusive word-index span into a cut, in seconds.

    ``silences`` comes from :func:`clipper.media.detect_silences` over
    ``audio.wav``. Passing none is supported and degrades gracefully to
    sentence extension plus padding - which is what happens on a machine
    without ffmpeg, and is still much better than raw word boundaries.
    """
    if not transcript.words:
        raise ValueError("cannot snap against an empty transcript")

    last = len(transcript.words) - 1
    start_index = max(0, min(start_index, last))
    end_index = max(start_index, min(end_index, last))

    extended_index = extend_to_sentence_start(
        transcript.words, start_index, max_lookback=max_lookback
    )
    extended = extended_index != start_index

    start_s = transcript.words[extended_index].start
    end_s = transcript.words[end_index].end

    silence_start = nearest_silence_edge(silences, start_s, direction="before", search_s=search_s)
    snapped_start = silence_start is not None
    if silence_start is not None:
        start_s = silence_start

    silence_end = nearest_silence_edge(silences, end_s, direction="after", search_s=search_s)
    snapped_end = silence_end is not None
    if silence_end is not None:
        end_s = silence_end

    pad = max(0, pad_ms) / 1000.0
    start_s = max(0.0, start_s - pad)
    end_s = end_s + pad

    # Only ever trim the tail. Moving the start would clip the first syllable,
    # which is the artefact this whole module is here to avoid; a clip that
    # runs slightly long is a trim in review, not a defect in the output.
    if max_duration_s is not None and end_s - start_s > max_duration_s:
        end_s = start_s + max_duration_s

    if end_s <= start_s:
        end_s = start_s + 0.1

    return Snapped(
        start=round(start_s, 3),
        end=round(end_s, 3),
        start_word_index=extended_index,
        end_word_index=end_index,
        snapped_start_to_silence=snapped_start,
        snapped_end_to_silence=snapped_end,
        extended_to_sentence=extended,
    )


def snap_all(
    transcript: Transcript,
    spans: Sequence[tuple[int, int]],
    *,
    silences: Sequence[tuple[float, float]] = (),
    pad_ms: int = 150,
    max_duration_s: float | None = None,
) -> list[Snapped]:
    """Snap several spans against one silence map."""
    return [
        snap(
            transcript,
            start,
            end,
            silences=silences,
            pad_ms=pad_ms,
            max_duration_s=max_duration_s,
        )
        for start, end in spans
    ]


def load_silences(
    audio_path, *, threshold_db: float = -30.0, min_silence_ms: int = 250
) -> list[tuple[float, float]]:
    """Detect silences, returning an empty list if ffmpeg is unavailable.

    Snapping degrades to sentence extension without ffmpeg rather than failing
    the stage. ``moments`` is otherwise a pure-text stage, and blocking it on a
    binary it only needs for polish would be the wrong trade.
    """
    from clipper import media

    try:
        return media.detect_silences(
            audio_path, threshold_db=threshold_db, min_silence_ms=min_silence_ms
        )
    except (media.FFmpegMissing, media.FFmpegError, FileNotFoundError):
        return []


__all__ = [
    "MAX_SENTENCE_LOOKBACK_WORDS",
    "SILENCE_SEARCH_S",
    "Snapped",
    "extend_to_sentence_start",
    "load_silences",
    "nearest_silence_edge",
    "snap",
    "snap_all",
]
