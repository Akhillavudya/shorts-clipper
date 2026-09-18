"""Pass 1 - windowing, response validation, and dedupe.

The LLM call itself is faked. What is worth testing here is everything around
it: that windows cover the transcript, that a model cannot smuggle an invented
index through, and that the overlap between windows does not produce two clips
of the same moment.
"""

from __future__ import annotations

from typing import Any

import pytest

from clipper.moments.recall import (
    MAX_SPAN_WORDS,
    MIN_SPAN_WORDS,
    Candidate,
    Window,
    dedupe,
    load,
    parse_candidates,
    plan_windows,
    render_words,
    run,
    save,
)
from clipper.transcribe.schema import Transcript, Word


def build(word_count: int, *, seconds_per_word: float = 0.5, speaker: str = "S0") -> Transcript:
    return Transcript(
        words=[
            Word(
                w=f"w{i}",
                start=i * seconds_per_word,
                end=i * seconds_per_word + seconds_per_word * 0.9,
                speaker=speaker,
            )
            for i in range(word_count)
        ],
        duration_s=word_count * seconds_per_word,
    )


class FakeRouter:
    """Records what it was asked and replays canned payloads.

    Matches ``Router.complete``'s signature, which is the only surface
    ``recall.run`` touches.
    """

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def complete(self, tier: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"tier": tier, "prompt": prompt, **kwargs})
        return self.payloads.pop(0) if self.payloads else {"candidates": []}


class TestPlanWindows:
    def test_short_transcript_is_one_window(self) -> None:
        transcript = build(100, seconds_per_word=0.5)  # 50s
        windows = plan_windows(transcript.words, chunk_minutes=10)
        assert len(windows) == 1
        assert windows[0].first_index == 0
        assert windows[0].last_index == 99

    def test_long_transcript_is_split(self) -> None:
        transcript = build(4000, seconds_per_word=0.5)  # ~33 min
        windows = plan_windows(transcript.words, chunk_minutes=10, overlap_s=60)
        assert len(windows) >= 3

    def test_windows_cover_every_word(self) -> None:
        transcript = build(4000, seconds_per_word=0.5)
        windows = plan_windows(transcript.words, chunk_minutes=10, overlap_s=60)
        covered: set[int] = set()
        for window in windows:
            covered.update(range(window.first_index, window.last_index + 1))
        assert covered == set(range(4000))

    def test_windows_overlap(self) -> None:
        """A moment on a boundary must be visible to both neighbours."""
        transcript = build(4000, seconds_per_word=0.5)
        windows = plan_windows(transcript.words, chunk_minutes=10, overlap_s=60)
        for earlier, later in zip(windows, windows[1:], strict=False):
            assert later.first_index <= earlier.last_index

    def test_windows_are_numbered_from_one(self) -> None:
        transcript = build(4000, seconds_per_word=0.5)
        windows = plan_windows(transcript.words, chunk_minutes=10)
        assert [w.number for w in windows] == list(range(1, len(windows) + 1))

    def test_empty_transcript(self) -> None:
        assert plan_windows([], chunk_minutes=10) == []

    def test_overlap_larger_than_the_chunk_does_not_hang(self) -> None:
        transcript = build(2000, seconds_per_word=0.5)
        windows = plan_windows(transcript.words, chunk_minutes=5, overlap_s=9999)
        assert 0 < len(windows) < 100


class TestRenderWords:
    def test_numbers_words_with_absolute_indices(self) -> None:
        transcript = build(5)
        rendered = render_words(transcript, Window(1, 2, 4))
        assert "2|w2" in rendered
        assert "0|w0" not in rendered

    def test_marks_speaker_changes(self) -> None:
        transcript = Transcript(
            words=[
                Word(w="a", start=0, end=1, speaker="S0"),
                Word(w="b", start=1, end=2, speaker="S1"),
            ]
        )
        rendered = render_words(transcript, Window(1, 0, 1))
        assert "[S0]" in rendered
        assert "[S1]" in rendered


class TestParseCandidates:
    WINDOW = Window(number=1, first_index=0, last_index=500)

    def payload(self, **overrides: Any) -> dict[str, Any]:
        entry = {
            "start_word_index": 10,
            "end_word_index": 60,
            "type": "insight",
            "one_line_reason": "because",
        }
        entry.update(overrides)
        return {"candidates": [entry]}

    def test_parses_a_good_candidate(self) -> None:
        [candidate] = parse_candidates(self.payload(), self.WINDOW)
        assert candidate.start_word_index == 10
        assert candidate.windows == [1]

    def test_drops_indices_outside_the_window(self) -> None:
        """An out-of-range index is invented, and invented cut points are the
        one thing the model is not allowed to produce."""
        assert parse_candidates(self.payload(end_word_index=9999), self.WINDOW) == []

    def test_swaps_a_reversed_span(self) -> None:
        [candidate] = parse_candidates(
            self.payload(start_word_index=60, end_word_index=10), self.WINDOW
        )
        assert (candidate.start_word_index, candidate.end_word_index) == (10, 60)

    def test_drops_a_span_that_is_too_short(self) -> None:
        payload = self.payload(end_word_index=10 + MIN_SPAN_WORDS - 2)
        assert parse_candidates(payload, self.WINDOW) == []

    def test_drops_a_span_that_is_too_long(self) -> None:
        payload = self.payload(end_word_index=10 + MAX_SPAN_WORDS + 5)
        assert parse_candidates(payload, self.WINDOW) == []

    def test_falls_back_on_an_unknown_type(self) -> None:
        [candidate] = parse_candidates(self.payload(type="vibes"), self.WINDOW)
        assert candidate.type == "insight"

    def test_normalizes_type_casing(self) -> None:
        [candidate] = parse_candidates(self.payload(type="SURPRISE"), self.WINDOW)
        assert candidate.type == "surprise"

    def test_tolerates_junk_entries(self) -> None:
        payload = {"candidates": ["nope", None, {"start_word_index": "x"}, {}]}
        assert parse_candidates(payload, self.WINDOW) == []

    def test_tolerates_a_missing_candidates_key(self) -> None:
        assert parse_candidates({}, self.WINDOW) == []


class TestDedupe:
    def make(self, start: int, end: int, window: int = 1, reason: str = "r") -> Candidate:
        return Candidate(
            start_word_index=start,
            end_word_index=end,
            type="insight",
            one_line_reason=reason,
            windows=[window],
        )

    def test_merges_heavily_overlapping_spans(self) -> None:
        merged = dedupe([self.make(100, 200, 1), self.make(110, 205, 2)])
        assert len(merged) == 1
        assert merged[0].start_word_index == 100
        assert merged[0].end_word_index == 205

    def test_merged_span_is_the_union(self) -> None:
        """Too wide is recoverable by snapping; too narrow is not."""
        merged = dedupe([self.make(100, 200), self.make(90, 195)])
        assert (merged[0].start_word_index, merged[0].end_word_index) == (90, 200)

    def test_records_both_source_windows(self) -> None:
        merged = dedupe([self.make(100, 200, 1), self.make(110, 205, 2)])
        assert sorted(merged[0].windows) == [1, 2]

    def test_keeps_distinct_moments_apart(self) -> None:
        assert len(dedupe([self.make(0, 100), self.make(500, 600)])) == 2

    def test_slight_overlap_is_not_a_duplicate(self) -> None:
        assert len(dedupe([self.make(0, 100), self.make(95, 300)])) == 2

    def test_prefers_the_more_specific_reason(self) -> None:
        merged = dedupe(
            [self.make(100, 200, 1, "short"), self.make(105, 200, 2, "much longer reason")]
        )
        assert merged[0].one_line_reason == "much longer reason"

    def test_output_is_ordered_by_position(self) -> None:
        merged = dedupe([self.make(500, 600), self.make(0, 100), self.make(200, 300)])
        assert [c.start_word_index for c in merged] == [0, 200, 500]

    def test_empty_input(self) -> None:
        assert dedupe([]) == []


class TestRun:
    def test_calls_the_recall_tier_once_per_window(self) -> None:
        transcript = build(4000, seconds_per_word=0.5)
        router = FakeRouter([])
        run(transcript, router, chunk_minutes=10, overlap_s=60)
        windows = plan_windows(transcript.words, chunk_minutes=10, overlap_s=60)
        assert len(router.calls) == len(windows)
        assert {c["tier"] for c in router.calls} == {"recall"}

    def test_requests_structured_output(self) -> None:
        """Every call in this pipeline returns JSON; none are free text."""
        router = FakeRouter([])
        run(build(100), router)
        assert router.calls[0]["json_schema"]["required"] == ["candidates"]

    def test_divides_the_target_across_windows(self) -> None:
        transcript = build(8000, seconds_per_word=0.5)  # ~67 min, many windows
        router = FakeRouter([])
        run(transcript, router, chunk_minutes=10, target_candidates=40)

        windows = plan_windows(transcript.words, chunk_minutes=10, overlap_s=60)
        assert len(windows) > 1
        # A multi-window episode must not ask each window for forty.
        expected = max(3, round(40 / len(windows)))
        assert f"{expected}, but return more" in router.calls[0]["prompt"]

    def test_returns_deduped_candidates(self) -> None:
        transcript = build(200)
        payload = {
            "candidates": [
                {
                    "start_word_index": 10,
                    "end_word_index": 80,
                    "type": "story",
                    "one_line_reason": "a",
                }
            ]
        }
        candidates = run(transcript, FakeRouter([payload]))
        assert len(candidates) == 1
        assert candidates[0].type == "story"

    def test_empty_transcript_makes_no_calls(self) -> None:
        router = FakeRouter([])
        assert run(Transcript(), router) == []
        assert router.calls == []


class TestPersistence:
    def test_round_trips(self, tmp_path) -> None:
        candidates = [
            Candidate(
                start_word_index=1,
                end_word_index=90,
                type="hook",
                one_line_reason="why",
                windows=[1, 2],
            )
        ]
        save(candidates, tmp_path)
        [loaded] = load(tmp_path)
        assert loaded == candidates[0]

    def test_missing_file_names_the_stage(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="clipper moments"):
            load(tmp_path)
