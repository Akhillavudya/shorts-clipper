"""The Gemini transcription backend, with a fake client.

No test may hit a real API, so the client is injected. What matters here is
the arithmetic around the call - window planning and, above all, the offset
that puts a window's timestamps back onto the episode timeline. Getting that
offset wrong produces a transcript that validates perfectly and drives every
cut point to the wrong place.
"""

from __future__ import annotations

from typing import Any

import pytest

from clipper.transcribe.gemini import (
    GeminiConfig,
    GeminiTranscriber,
    TranscriptionError,
    _extract_json,
    parse_response,
    plan_windows,
)

SETTINGS = {
    "transcription": {
        "gemini": {
            "model": "test-model",
            "api_key_env": "TEST_KEY",
            "chunk_minutes": 10,
            "max_attempts": 2,
        }
    }
}


def window_payload(*words: tuple[str, float, float], speaker: str = "SPEAKER_00") -> dict[str, Any]:
    return {
        "language": "en",
        "segments": [
            {
                "speaker": speaker,
                "start": words[0][1],
                "end": words[-1][2],
                "text": " ".join(w for w, _, _ in words),
                "words": [{"w": w, "start": s, "end": e} for w, s, e in words],
            }
        ],
    }


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeClient:
    """Stands in for ``genai.Client``. Records uploads and returns payloads."""

    def __init__(self, payloads: list[dict[str, Any]], *, fail_times: int = 0) -> None:
        import json

        self._payloads = [json.dumps(p) for p in payloads]
        self._fail_times = fail_times
        self.uploaded: list[str] = []
        self.models = self._Models(self)
        self.files = self._Files(self)

    class _Files:
        def __init__(self, outer: FakeClient) -> None:
            self._outer = outer

        def upload(self, *, file: str) -> str:
            self._outer.uploaded.append(file)
            return f"uploaded:{file}"

    class _Models:
        def __init__(self, outer: FakeClient) -> None:
            self._outer = outer

        def generate_content(self, **kwargs: Any) -> FakeResponse:
            if self._outer._fail_times > 0:
                self._outer._fail_times -= 1
                raise RuntimeError("transient upstream failure")
            return FakeResponse(self._outer._payloads.pop(0))


class TestGeminiConfig:
    def test_reads_the_settings_block(self) -> None:
        config = GeminiConfig.from_settings(SETTINGS)
        assert config.model == "test-model"
        assert config.chunk_minutes == 10

    def test_missing_model_names_the_rule_it_breaks(self) -> None:
        """Model ids live in config, never in src/."""
        with pytest.raises(TranscriptionError, match="hard rule 4"):
            GeminiConfig.from_settings({"transcription": {"gemini": {}}})


class TestPlanWindows:
    def test_short_audio_is_one_window(self) -> None:
        assert plan_windows(120.0, 600.0) == [(0.0, 120.0)]

    def test_long_audio_is_split(self) -> None:
        assert plan_windows(1500.0, 600.0) == [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1500.0)]

    def test_windows_do_not_overlap(self) -> None:
        """Overlapping audio would duplicate words and corrupt every index
        after the duplicate."""
        windows = plan_windows(5000.0, 600.0)
        for earlier, later in zip(windows, windows[1:], strict=False):
            assert earlier[1] == later[0]

    def test_windows_cover_the_whole_file(self) -> None:
        windows = plan_windows(5000.0, 600.0)
        assert windows[0][0] == 0.0
        assert windows[-1][1] == 5000.0

    def test_absorbs_a_stub_tail(self) -> None:
        """A four-second final window is not worth its own API call."""
        windows = plan_windows(1204.0, 600.0)
        assert len(windows) == 2
        assert windows[-1] == (600.0, 1204.0)

    def test_zero_duration(self) -> None:
        assert plan_windows(0.0, 600.0) == []


class TestParseResponse:
    def test_parses_words_and_segments(self) -> None:
        payload = window_payload(("hello", 0.0, 0.5), ("world", 0.5, 1.0))
        words, segments, language = parse_response(payload)
        assert [w.w for w in words] == ["hello", "world"]
        assert len(segments) == 1
        assert language == "en"

    def test_applies_the_window_offset(self) -> None:
        """This is the arithmetic that puts a window back on the episode
        timeline; an error here is invisible and total."""
        payload = window_payload(("late", 5.0, 5.5))
        words, segments, _ = parse_response(payload, offset_s=600.0)
        assert words[0].start == pytest.approx(605.0)
        assert segments[0].start == pytest.approx(605.0)

    def test_clamps_a_time_past_the_window_end(self) -> None:
        """Drift past the window end is drift, not hallucination - keep it."""
        payload = window_payload(("over", 599.0, 601.5))
        words, _, _ = parse_response(payload, offset_s=0.0, window_s=600.0)
        assert words[0].end == pytest.approx(600.0)

    def test_carries_the_speaker_onto_every_word(self) -> None:
        payload = window_payload(("a", 0.0, 0.2), speaker="SPEAKER_03")
        words, _, _ = parse_response(payload)
        assert words[0].speaker == "SPEAKER_03"

    def test_skips_blank_words(self) -> None:
        payload = {
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "x",
                    "words": [
                        {"w": "  ", "start": 0, "end": 0.1},
                        {"w": "real", "start": 0.1, "end": 0.5},
                    ],
                }
            ]
        }
        words, _, _ = parse_response(payload)
        assert [w.w for w in words] == ["real"]

    def test_rebuilds_missing_segment_text_from_the_words(self) -> None:
        payload = {
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "",
                    "words": [
                        {"w": "one", "start": 0, "end": 0.4},
                        {"w": "two", "start": 0.4, "end": 0.8},
                    ],
                }
            ]
        }
        _, segments, _ = parse_response(payload)
        assert segments[0].text == "one two"

    def test_tolerates_junk(self) -> None:
        words, segments, _ = parse_response({"segments": ["nope", None, {}]})
        assert words == []
        assert segments == []

    def test_empty_payload(self) -> None:
        assert parse_response({}) == ([], [], None)


class TestTruncationGuard:
    """A window that overran the output limit must fail, not return short."""

    class Candidate:
        def __init__(self, reason: Any) -> None:
            self.finish_reason = reason

    class Raw:
        def __init__(self, text: str, reason: Any) -> None:
            self.text = text
            self.candidates = [TestTruncationGuard.Candidate(reason)]

    def test_max_tokens_is_an_error_even_with_parseable_json(self) -> None:
        raw = self.Raw('{"segments": []}', "MAX_TOKENS")
        with pytest.raises(TranscriptionError, match="chunk_minutes"):
            _extract_json(raw)

    def test_reads_an_enum_shaped_reason(self) -> None:
        class Reason:
            name = "MAX_TOKENS"

        with pytest.raises(TranscriptionError, match="chunk_minutes"):
            _extract_json(self.Raw('{"segments": []}', Reason()))

    def test_a_normal_stop_is_left_alone(self) -> None:
        assert _extract_json(self.Raw('{"segments": []}', "STOP")) == {"segments": []}

    def test_no_candidates_is_not_a_truncation(self) -> None:
        assert _extract_json(FakeResponse('{"segments": []}')) == {"segments": []}


class TestTranscribe:
    def transcriber(self, payloads: list[dict[str, Any]], **kwargs: Any) -> GeminiTranscriber:
        self.slept: list[float] = []
        return GeminiTranscriber(
            GeminiConfig.from_settings(SETTINGS),
            client=FakeClient(payloads, **kwargs),
            # Time is injected, never slept: a real backoff would make this
            # suite take seconds to prove arithmetic.
            sleep_fn=self.slept.append,
        )

    def test_single_window_transcription(self, tmp_path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"not really audio")
        payload = window_payload(("hello", 0.0, 0.4), ("there.", 0.4, 0.9))

        transcript = self.transcriber([payload]).transcribe(audio, total_s=60.0)

        assert [w.w for w in transcript.words] == ["hello", "there."]
        assert transcript.backend == "gemini"
        assert transcript.duration_s == 60.0

    def test_result_is_normalized(self, tmp_path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        payload = window_payload(("second", 1.0, 1.4), ("first", 0.0, 0.4))

        transcript = self.transcriber([payload]).transcribe(audio, total_s=10.0)
        assert [w.w for w in transcript.words] == ["first", "second"]

    def test_retries_a_transient_failure(self, tmp_path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        payload = window_payload(("ok", 0.0, 0.4))

        transcriber = self.transcriber([payload], fail_times=1)
        transcript = transcriber.transcribe(audio, total_s=10.0)
        assert transcript.word_count == 1
        assert self.slept == [2.0]  # backed off once, without waiting for it

    def test_gives_up_after_max_attempts(self, tmp_path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        transcriber = self.transcriber([window_payload(("x", 0, 1))], fail_times=99)
        with pytest.raises(TranscriptionError, match="after 2 attempts"):
            transcriber.transcribe(audio, total_s=10.0)

    def test_a_wordless_response_is_an_error(self, tmp_path) -> None:
        """Word-level timestamps are non-negotiable (hard rule 7)."""
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        transcriber = self.transcriber([{"segments": []}])
        with pytest.raises(TranscriptionError, match="hard rule 7"):
            transcriber.transcribe(audio, total_s=10.0)

    def test_missing_audio_file(self, tmp_path) -> None:
        with pytest.raises(TranscriptionError, match="no audio"):
            self.transcriber([]).transcribe(tmp_path / "nope.wav", total_s=10.0)

    def test_missing_api_key_is_reported_clearly(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("TEST_KEY", raising=False)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"x")
        transcriber = GeminiTranscriber(GeminiConfig.from_settings(SETTINGS))
        with pytest.raises(TranscriptionError, match="TEST_KEY is not set"):
            transcriber.transcribe(audio, total_s=10.0)
