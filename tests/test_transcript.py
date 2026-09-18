"""The transcript schema, and the invariants everything downstream assumes.

Word index is the pipeline's addressing scheme, so most of what is checked
here is really "does index N still mean the same word after this". A backend
that returns a ragged word list must not be able to shift every cut point in
the episode by one.
"""

from __future__ import annotations

import pytest

from clipper.transcribe.schema import Segment, Transcript, Word, load_transcript


def make_words(*specs: tuple[str, float, float, str | None]) -> list[Word]:
    return [Word(w=w, start=s, end=e, speaker=sp) for w, s, e, sp in specs]


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        words=make_words(
            ("The", 0.0, 0.2, "SPEAKER_00"),
            ("thing", 0.2, 0.5, "SPEAKER_00"),
            ("is,", 0.5, 0.8, "SPEAKER_00"),
            ("nobody", 0.8, 1.2, "SPEAKER_00"),
            ("checks.", 1.2, 1.7, "SPEAKER_00"),
            ("Really?", 2.5, 3.0, "SPEAKER_01"),
            ("Never.", 3.2, 3.8, "SPEAKER_00"),
        ),
        duration_s=4.0,
    )


class TestWord:
    def test_duration(self) -> None:
        assert Word(w="hi", start=1.0, end=1.5).duration_s == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("checks.", True),
            ("Really?", True),
            ("Stop!", True),
            ("thing", False),
            ("is,", False),
            # Terminal punctuation behind a closing quote still ends a
            # sentence; snap.py walks back from here, so missing it would
            # extend the clip into the previous paragraph.
            ('done."', True),
            ("(yes.)", True),
        ],
    )
    def test_ends_sentence(self, text: str, expected: bool) -> None:
        assert Word(w=text, start=0.0, end=0.1).ends_sentence is expected


class TestNormalized:
    def test_sorts_out_of_order_words(self) -> None:
        ragged = Transcript(
            words=make_words(
                ("second", 1.0, 1.5, None),
                ("first", 0.0, 0.5, None),
            )
        )
        assert [w.w for w in ragged.normalized().words] == ["first", "second"]

    def test_drops_blank_words(self) -> None:
        ragged = Transcript(words=make_words(("real", 0.0, 0.5, None), ("   ", 0.6, 0.7, None)))
        assert [w.w for w in ragged.normalized().words] == ["real"]

    def test_gives_zero_length_words_a_duration(self) -> None:
        ragged = Transcript(words=make_words(("blip", 1.0, 1.0, None)))
        assert ragged.normalized().words[0].duration_s > 0

    def test_pushes_overlapping_words_apart(self) -> None:
        # Two words highlighting at once is a visible caption bug.
        ragged = Transcript(words=make_words(("one", 0.0, 1.0, None), ("two", 0.5, 1.5, None)))
        first, second = ragged.normalized().words
        assert second.start >= first.end

    def test_infers_duration_from_last_word(self) -> None:
        ragged = Transcript(words=make_words(("end", 9.0, 9.9, None)))
        assert ragged.normalized().duration_s == pytest.approx(9.9)

    def test_is_idempotent(self, transcript: Transcript) -> None:
        once = transcript.normalized()
        assert once.model_dump() == once.normalized().model_dump()


class TestQueries:
    def test_slice_is_inclusive_of_the_end_index(self, transcript: Transcript) -> None:
        # Exclusive slicing here would silently drop the last word of every clip.
        assert [w.w for w in transcript.slice(0, 2)] == ["The", "thing", "is,"]

    def test_slice_clamps_out_of_range_indices(self, transcript: Transcript) -> None:
        assert len(transcript.slice(-5, 999)) == transcript.word_count

    def test_text_for_span(self, transcript: Transcript) -> None:
        assert transcript.text_for(3, 4) == "nobody checks."

    def test_span_seconds_uses_the_span_edges(self, transcript: Transcript) -> None:
        assert transcript.span_seconds(1, 4) == (pytest.approx(0.2), pytest.approx(1.7))

    def test_span_seconds_on_empty_transcript(self) -> None:
        assert Transcript().span_seconds(0, 5) == (0.0, 0.0)

    def test_index_at_finds_the_word_being_spoken(self, transcript: Transcript) -> None:
        assert transcript.index_at(0.6) == 2

    def test_index_at_past_the_end_returns_the_last_word(self, transcript: Transcript) -> None:
        assert transcript.index_at(99.0) == transcript.word_count - 1

    def test_speakers_in_first_appearance_order(self, transcript: Transcript) -> None:
        assert transcript.speakers() == ["SPEAKER_00", "SPEAKER_01"]

    def test_has_speaker_change(self, transcript: Transcript) -> None:
        assert transcript.has_speaker_change(0, 6) is True
        assert transcript.has_speaker_change(0, 4) is False


class TestSegments:
    def test_builds_segments_from_words(self, transcript: Transcript) -> None:
        built = transcript.with_segments_from_words()
        assert [s.speaker for s in built.segments] == [
            "SPEAKER_00",
            "SPEAKER_01",
            "SPEAKER_00",
        ]

    def test_splits_on_a_long_pause(self) -> None:
        words = make_words(("a", 0.0, 0.2, "S0"), ("b", 5.0, 5.2, "S0"))
        built = Transcript(words=words).with_segments_from_words(max_gap_s=1.0)
        assert len(built.segments) == 2

    def test_leaves_existing_segments_alone(self, transcript: Transcript) -> None:
        existing = transcript.model_copy(
            update={"segments": [Segment(start=0, end=1, text="kept")]}
        )
        assert existing.with_segments_from_words().segments[0].text == "kept"


class TestIO:
    def test_round_trips_through_disk(self, tmp_path, transcript: Transcript) -> None:
        transcript.save(tmp_path / "transcript.json")
        assert load_transcript(tmp_path).model_dump() == transcript.model_dump()

    def test_missing_file_names_the_stage_to_run(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="clipper transcribe"):
            load_transcript(tmp_path)
