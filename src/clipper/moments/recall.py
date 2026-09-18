"""Pass 1 - candidate moment recall (FULL_PLAN.md section 5).

Responsibility: sweep the whole transcript and return many candidate spans.
This is a recall task, so it runs on the cheap `recall` tier and is judged on
coverage, not precision - pass 2 does the filtering.

Chunk the transcript into ~10-minute windows with 60s overlap; the overlap
stops a moment straddling a chunk boundary from being missed. Per candidate,
ask for:

    start_word_index, end_word_index, type, one_line_reason

where type is one of: hook, surprise, insight, story, emotion, contrarian,
educational, conclusion.

Target 30-50 candidates across a 2-hour episode. Strict JSON output. Word
indices, never timestamps - the model is not trusted with cut points.

Writes jobs/<video_id>/candidates.json.

The overlap is what makes this pass produce duplicates, and that is fine: two
windows that both see a moment near their shared boundary will both report it,
usually with slightly different edges. :func:`dedupe` merges them here rather
than paying to score the same moment twice in pass 2.

Indices are absolute into ``Transcript.words`` throughout. The prompt shows
real indices rather than renumbering per window, so nothing has to be
translated back afterwards - and a model that echoes an index from the
transcript it was shown is right by construction. Out-of-range indices are
dropped rather than clamped: an index outside the window means the model
invented it, and inventing cut points is the one thing it is not allowed to do.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clipper.llm import Router
from clipper.moments import prompts
from clipper.transcribe.schema import Transcript, Word

CANDIDATES_FILENAME = "candidates.json"

MOMENT_TYPES = (
    "hook",
    "surprise",
    "insight",
    "story",
    "emotion",
    "contrarian",
    "educational",
    "conclusion",
)

#: Sanity bounds on a returned span, in words. A three-word "moment" is a
#: parsing accident; a 900-word one is the model handing back a whole chapter.
#: Both are dropped rather than trimmed - a bad span is not worth guessing at.
MIN_SPAN_WORDS = 15
MAX_SPAN_WORDS = 400

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_word_index": {"type": "integer"},
                    "end_word_index": {"type": "integer"},
                    "type": {"type": "string", "enum": list(MOMENT_TYPES)},
                    "one_line_reason": {"type": "string"},
                },
                "required": [
                    "start_word_index",
                    "end_word_index",
                    "type",
                    "one_line_reason",
                ],
            },
        }
    },
    "required": ["candidates"],
}


@dataclass
class Candidate:
    """One span from pass 1. Word indices only - never seconds."""

    start_word_index: int
    end_word_index: int
    type: str
    one_line_reason: str
    windows: list[int] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return self.end_word_index - self.start_word_index + 1

    def overlaps(self, other: Candidate) -> bool:
        return (
            self.start_word_index <= other.end_word_index
            and other.start_word_index <= self.end_word_index
        )

    def overlap_ratio(self, other: Candidate) -> float:
        """Shared words as a fraction of the shorter span."""
        lo = max(self.start_word_index, other.start_word_index)
        hi = min(self.end_word_index, other.end_word_index)
        if hi < lo:
            return 0.0
        shared = hi - lo + 1
        return shared / max(1, min(self.word_count, other.word_count))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        return cls(
            start_word_index=int(data["start_word_index"]),
            end_word_index=int(data["end_word_index"]),
            type=str(data.get("type") or "insight"),
            one_line_reason=str(data.get("one_line_reason") or ""),
            windows=list(data.get("windows") or []),
        )


@dataclass(frozen=True)
class Window:
    """A slice of the transcript sent as one prompt. Inclusive on both ends."""

    number: int
    first_index: int
    last_index: int

    @property
    def word_count(self) -> int:
        return self.last_index - self.first_index + 1

    def contains(self, index: int) -> bool:
        return self.first_index <= index <= self.last_index


def plan_windows(
    words: list[Word], *, chunk_minutes: float = 10.0, overlap_s: float = 60.0
) -> list[Window]:
    """Split the transcript into overlapping windows, by wall-clock time.

    By time rather than word count, because speech rate varies enough that a
    fixed word count produces windows of wildly different durations - and it is
    duration, not word count, that determines whether the response fits in one
    completion.
    """
    if not words:
        return []

    chunk_s = max(60.0, chunk_minutes * 60.0)
    overlap_s = max(0.0, min(overlap_s, chunk_s / 2))
    total_s = words[-1].end

    if total_s <= chunk_s:
        return [Window(number=1, first_index=0, last_index=len(words) - 1)]

    windows: list[Window] = []
    start_s = 0.0
    number = 1
    cursor = 0

    while start_s < total_s - 0.5:
        end_s = min(start_s + chunk_s, total_s)

        first_index = cursor
        while first_index > 0 and words[first_index - 1].end > start_s:
            first_index -= 1

        last_index = first_index
        while last_index + 1 < len(words) and words[last_index + 1].start < end_s:
            last_index += 1

        if not windows or last_index > windows[-1].last_index:
            windows.append(Window(number=number, first_index=first_index, last_index=last_index))
            number += 1

        if end_s >= total_s:
            break

        start_s = end_s - overlap_s
        # Advance the search hint to the first word of the next window, so the
        # backwards walk above stays O(overlap) rather than O(transcript).
        while cursor < len(words) and words[cursor].end <= start_s:
            cursor += 1

    return windows


def render_words(transcript: Transcript, window: Window) -> str:
    """Number every word in the window as ``index|word`` for the prompt.

    A speaker marker is emitted on each change. Speaker identity matters to
    self-containment - a span that starts mid-answer to an unheard question is
    a bad clip - and this is cheaper than describing it in prose.
    """
    lines: list[str] = []
    current_speaker: str | None = None
    for index in range(window.first_index, window.last_index + 1):
        word = transcript.words[index]
        if word.speaker and word.speaker != current_speaker:
            lines.append(f"[{word.speaker}]")
            current_speaker = word.speaker
        lines.append(f"{index}|{word.w}")
    return " ".join(lines)


def parse_candidates(payload: dict[str, Any], window: Window) -> list[Candidate]:
    """Validate one window's response into :class:`Candidate` objects.

    Everything questionable is dropped, not repaired. This is the recall pass -
    there are dozens of other candidates, so discarding a malformed one costs
    nothing, while silently fixing up a span the model did not mean invents a
    cut point out of thin air.
    """
    kept: list[Candidate] = []
    for raw in payload.get("candidates") or []:
        if not isinstance(raw, dict):
            continue
        try:
            start = int(raw["start_word_index"])
            end = int(raw["end_word_index"])
        except (KeyError, TypeError, ValueError):
            continue

        if start > end:
            start, end = end, start
        if not window.contains(start) or not window.contains(end):
            continue
        if not (MIN_SPAN_WORDS <= end - start + 1 <= MAX_SPAN_WORDS):
            continue

        moment_type = str(raw.get("type") or "").strip().lower()
        if moment_type not in MOMENT_TYPES:
            moment_type = "insight"

        kept.append(
            Candidate(
                start_word_index=start,
                end_word_index=end,
                type=moment_type,
                one_line_reason=str(raw.get("one_line_reason") or "").strip(),
                windows=[window.number],
            )
        )
    return kept


def dedupe(candidates: list[Candidate], *, threshold: float = 0.6) -> list[Candidate]:
    """Merge candidates that describe the same moment.

    Two spans overlapping by more than ``threshold`` of the shorter one are the
    same moment seen twice - almost always the same moment reported by two
    overlapping windows. The merged span is the union, since pass 1 is a recall
    task and ``moments/snap.py`` will pull the real boundaries to silence
    anyway; a span that is slightly too wide is recoverable, one that is too
    narrow is not.
    """
    ordered = sorted(candidates, key=lambda c: (c.start_word_index, c.end_word_index))
    merged: list[Candidate] = []

    for candidate in ordered:
        target = next((m for m in merged if m.overlap_ratio(candidate) >= threshold), None)
        if target is None:
            merged.append(candidate)
            continue

        target.start_word_index = min(target.start_word_index, candidate.start_word_index)
        target.end_word_index = max(target.end_word_index, candidate.end_word_index)
        # Keep the longer reason: the more specific one is usually the longer
        # one, and it is what a human reads during review.
        if len(candidate.one_line_reason) > len(target.one_line_reason):
            target.one_line_reason = candidate.one_line_reason
            target.type = candidate.type
        for window_number in candidate.windows:
            if window_number not in target.windows:
                target.windows.append(window_number)

    return merged


def run(
    transcript: Transcript,
    router: Router,
    *,
    title: str | None = None,
    chunk_minutes: float = 10.0,
    overlap_s: float = 60.0,
    target_candidates: int = 40,
    on_progress: Any = None,
) -> list[Candidate]:
    """Sweep the transcript and return deduped candidates.

    ``target_candidates`` is an episode-wide target; it is divided across the
    windows so a 2-hour episode does not ask each of twelve windows for forty.
    """
    windows = plan_windows(transcript.words, chunk_minutes=chunk_minutes, overlap_s=overlap_s)
    if not windows:
        return []

    per_window = max(3, round(target_candidates / len(windows)))
    found: list[Candidate] = []

    for window in windows:
        if on_progress is not None:
            on_progress(window, len(windows))

        prompt = prompts.render(
            "recall.user",
            title=title or "(untitled)",
            window_number=window.number,
            window_count=len(windows),
            first_index=window.first_index,
            last_index=window.last_index,
            target_candidates=per_window,
            words=render_words(transcript, window),
        )
        payload = router.complete(
            "recall",
            prompt,
            system=prompts.load("recall.system"),
            json_schema=RESPONSE_SCHEMA,
            temperature=0.4,
        )
        found.extend(parse_candidates(payload, window))

    return dedupe(found)


# -- persistence ----------------------------------------------------------


def candidates_path(job_dir: Path | str) -> Path:
    return Path(job_dir) / CANDIDATES_FILENAME


def save(candidates: list[Candidate], job_dir: Path | str) -> Path:
    path = candidates_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": [c.to_dict() for c in candidates]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(job_dir: Path | str) -> list[Candidate]:
    path = candidates_path(job_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no candidates at {path} - run `clipper moments` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Candidate.from_dict(c) for c in payload.get("candidates") or []]


__all__ = [
    "CANDIDATES_FILENAME",
    "MAX_SPAN_WORDS",
    "MIN_SPAN_WORDS",
    "MOMENT_TYPES",
    "RESPONSE_SCHEMA",
    "Candidate",
    "Window",
    "candidates_path",
    "dedupe",
    "load",
    "parse_candidates",
    "plan_windows",
    "render_words",
    "run",
    "save",
]
