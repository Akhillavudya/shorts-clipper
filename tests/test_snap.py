"""Boundary snapping - the deterministic half of `moments`.

This is where CLAUDE.md hard rule 2 is enforced, so these tests are less about
edge cases and more about the guarantee: word indices in, timestamps out,
same answer every time, and never a boundary that clips a syllable.
"""

from __future__ import annotations

import pytest

from clipper.moments.snap import (
    Snapped,
    extend_to_sentence_start,
    nearest_silence_edge,
    snap,
    snap_all,
)
from clipper.transcribe.schema import Transcript, Word


def build(*specs: tuple[str, float, float, str | None]) -> Transcript:
    return Transcript(words=[Word(w=w, start=s, end=e, speaker=sp) for w, s, e, sp in specs])


@pytest.fixture
def transcript() -> Transcript:
    """Two sentences from one speaker, then a second speaker."""
    return build(
        ("So", 10.0, 10.2, "S0"),
        ("anyway.", 10.2, 10.8, "S0"),
        ("The", 11.5, 11.7, "S0"),
        ("real", 11.7, 12.0, "S0"),
        ("problem", 12.0, 12.5, "S0"),
        ("is", 12.5, 12.7, "S0"),
        ("cost.", 12.7, 13.4, "S0"),
        ("Right,", 14.0, 14.4, "S1"),
        ("exactly.", 14.4, 15.0, "S1"),
    )


class TestExtendToSentenceStart:
    def test_walks_back_to_the_sentence_start(self, transcript: Transcript) -> None:
        # "problem" (index 4) belongs to the sentence beginning at "The" (2).
        assert extend_to_sentence_start(transcript.words, 4) == 2

    def test_stops_at_terminal_punctuation(self, transcript: Transcript) -> None:
        assert extend_to_sentence_start(transcript.words, 2) == 2

    def test_does_not_cross_a_speaker_change(self, transcript: Transcript) -> None:
        # "exactly." is S1; extending back must not swallow S0's line.
        assert extend_to_sentence_start(transcript.words, 8) == 7

    def test_respects_the_lookback_limit(self) -> None:
        run_on = build(*[(f"w{i}", i * 0.1, i * 0.1 + 0.09, "S0") for i in range(200)])
        assert extend_to_sentence_start(run_on.words, 150, max_lookback=10) == 140

    def test_handles_an_empty_word_list(self) -> None:
        assert extend_to_sentence_start([], 3) == 3


class TestNearestSilenceEdge:
    SILENCES = [(1.0, 1.6), (5.0, 5.4), (20.0, 21.0)]

    def test_start_lands_on_the_end_of_the_preceding_silence(self) -> None:
        assert nearest_silence_edge(self.SILENCES, 2.0, direction="before") == 1.6

    def test_end_lands_on_the_start_of_the_following_silence(self) -> None:
        assert nearest_silence_edge(self.SILENCES, 4.5, direction="after") == 5.0

    def test_returns_none_when_nothing_is_near(self) -> None:
        assert nearest_silence_edge(self.SILENCES, 12.0, direction="before") is None
        assert nearest_silence_edge(self.SILENCES, 12.0, direction="after") is None

    def test_respects_the_search_window(self) -> None:
        assert nearest_silence_edge(self.SILENCES, 3.0, direction="before", search_s=0.5) is None
        assert nearest_silence_edge(self.SILENCES, 3.0, direction="before", search_s=2.0) == 1.6

    def test_picks_the_closest_of_several(self) -> None:
        silences = [(1.0, 1.2), (1.5, 1.8), (1.9, 1.95)]
        assert nearest_silence_edge(silences, 2.0, direction="before") == 1.95

    def test_empty_silence_map(self) -> None:
        assert nearest_silence_edge([], 5.0, direction="before") is None

    def test_rejects_an_unknown_direction(self) -> None:
        with pytest.raises(ValueError, match="before"):
            nearest_silence_edge(self.SILENCES, 1.0, direction="sideways")


class TestSnap:
    def test_extends_to_the_sentence_start_and_pads(self, transcript: Transcript) -> None:
        result = snap(transcript, 4, 6, pad_ms=150)
        assert result.extended_to_sentence is True
        assert result.start_word_index == 2
        # "The" starts at 11.5, minus 150ms of padding.
        assert result.start == pytest.approx(11.35)
        assert result.end == pytest.approx(13.55)

    def test_snaps_outward_onto_silence(self, transcript: Transcript) -> None:
        silences = [(10.8, 11.4), (13.4, 13.9)]
        result = snap(transcript, 2, 6, silences=silences, pad_ms=0)
        assert result.snapped_start_to_silence is True
        assert result.snapped_end_to_silence is True
        assert result.start == pytest.approx(11.4)
        assert result.end == pytest.approx(13.4)

    def test_snapping_only_ever_widens(self, transcript: Transcript) -> None:
        """A snap that narrowed the clip would eat the first syllable."""
        raw_start, raw_end = transcript.span_seconds(2, 6)
        silences = [(10.8, 11.4), (13.4, 13.9)]
        result = snap(transcript, 2, 6, silences=silences, pad_ms=0)
        assert result.start <= raw_start
        assert result.end >= raw_end

    def test_padding_never_produces_a_negative_start(self) -> None:
        at_zero = build(("Go.", 0.0, 0.4, "S0"))
        assert snap(at_zero, 0, 0, pad_ms=500).start == 0.0

    def test_clamps_indices_into_range(self, transcript: Transcript) -> None:
        result = snap(transcript, -10, 999)
        assert result.start_word_index == 0
        assert result.end_word_index == transcript.word_count - 1

    def test_reversed_span_does_not_produce_a_negative_duration(
        self, transcript: Transcript
    ) -> None:
        assert snap(transcript, 6, 2).duration_s > 0

    def test_max_duration_trims_the_tail_not_the_head(self, transcript: Transcript) -> None:
        untrimmed = snap(transcript, 2, 8, pad_ms=0)
        trimmed = snap(transcript, 2, 8, pad_ms=0, max_duration_s=1.0)
        assert trimmed.start == untrimmed.start
        assert trimmed.duration_s == pytest.approx(1.0)

    def test_works_with_no_silence_map(self, transcript: Transcript) -> None:
        """Degrades to sentence extension when ffmpeg is unavailable."""
        result = snap(transcript, 4, 6, silences=[])
        assert result.snapped_start_to_silence is False
        assert result.extended_to_sentence is True

    def test_is_deterministic(self, transcript: Transcript) -> None:
        silences = [(10.8, 11.4), (13.4, 13.9)]
        runs = {snap(transcript, 3, 6, silences=silences) for _ in range(5)}
        assert len(runs) == 1

    def test_rejects_an_empty_transcript(self) -> None:
        with pytest.raises(ValueError, match="empty transcript"):
            snap(Transcript(), 0, 1)

    def test_returns_a_frozen_result(self, transcript: Transcript) -> None:
        result = snap(transcript, 2, 6)
        assert isinstance(result, Snapped)
        with pytest.raises((AttributeError, TypeError)):
            result.start = 0.0  # type: ignore[misc]


class TestSnapAll:
    def test_snaps_several_spans_against_one_silence_map(self, transcript: Transcript) -> None:
        results = snap_all(transcript, [(2, 6), (7, 8)], silences=[(13.4, 13.9)])
        assert len(results) == 2
        assert results[0].end < results[1].start
