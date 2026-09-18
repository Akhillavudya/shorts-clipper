"""Pass 2 - selection, the weighted sum, and what gets archived.

The arithmetic is deliberately in Python rather than the prompt, so it is
testable, and the calibration loop depends on ``scored.json`` holding the raw
axes. Both of those are load-bearing (CLAUDE.md hard rule 8) and both are
checked here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from clipper.moments.recall import Candidate
from clipper.moments.score import (
    RUBRIC_AXES,
    ScoredClip,
    apply_scores,
    build_clips,
    load,
    render_candidates,
    run,
    save,
    select_for_rerank,
    shortlist,
    weighted_total,
)
from clipper.transcribe.schema import Transcript, Word

WEIGHTS: dict[str, Any] = {
    "weights": {
        "hook": 0.25,
        "self_containment": 0.20,
        "payoff": 0.15,
        "surprise_emotion": 0.15,
        "visual_independence": 0.05,
        "quotability": 0.10,
        "length_fit": 0.10,
    },
    "modifiers": {
        "length_penalty": {"enabled": True, "under_min_s": 0.85, "over_max_s": 0.80},
        "speaker_change_penalty": 0.95,
    },
    "min_score": 5.5,
    "max_clips_per_video": 3,
}


def transcript_of(word_count: int, *, seconds_per_word: float = 0.4) -> Transcript:
    return Transcript(
        words=[
            Word(
                w=f"word{i}",
                start=i * seconds_per_word,
                end=(i + 1) * seconds_per_word,
                speaker="S0",
            )
            for i in range(word_count)
        ],
        duration_s=word_count * seconds_per_word,
    )


def candidate(start: int, end: int, kind: str = "insight") -> Candidate:
    return Candidate(
        start_word_index=start,
        end_word_index=end,
        type=kind,
        one_line_reason="reason",
    )


class FakeRouter:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def complete(self, tier: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"tier": tier, "prompt": prompt, **kwargs})
        return self.payload


def full_rubric(value: float = 8.0) -> dict[str, float]:
    return dict.fromkeys(RUBRIC_AXES, value)


class TestSelectForRerank:
    def test_returns_everything_when_under_the_limit(self) -> None:
        candidates = [candidate(i * 100, i * 100 + 80) for i in range(5)]
        assert len(select_for_rerank(candidates, 20)) == 5

    def test_caps_at_top_n(self) -> None:
        candidates = [candidate(i * 100, i * 100 + 80) for i in range(50)]
        assert len(select_for_rerank(candidates, 20)) == 20

    def test_spreads_across_the_timeline(self) -> None:
        """Taking the first N would score the opening and ignore the rest."""
        candidates = [candidate(i * 100, i * 100 + 80) for i in range(50)]
        picked = select_for_rerank(candidates, 10)
        assert picked[-1].start_word_index > 4000

    def test_output_is_ordered_by_position(self) -> None:
        candidates = [candidate(i * 100, i * 100 + 80) for i in range(30)]
        picked = select_for_rerank(candidates, 10)
        assert picked == sorted(picked, key=lambda c: c.start_word_index)

    def test_no_duplicates_when_barely_over_the_limit(self) -> None:
        candidates = [candidate(i * 100, i * 100 + 80) for i in range(21)]
        picked = select_for_rerank(candidates, 20)
        assert len(picked) == 20
        assert len({id(c) for c in picked}) == 20

    def test_empty_input(self) -> None:
        assert select_for_rerank([], 20) == []


class TestWeightedTotal:
    def test_all_tens_is_ten(self) -> None:
        assert weighted_total(full_rubric(10.0), WEIGHTS["weights"]) == pytest.approx(10.0)

    def test_all_ones_is_one(self) -> None:
        assert weighted_total(full_rubric(1.0), WEIGHTS["weights"]) == pytest.approx(1.0)

    def test_hook_is_weighted_hardest(self) -> None:
        strong_hook = full_rubric(5.0) | {"hook": 10.0}
        strong_visual = full_rubric(5.0) | {"visual_independence": 10.0}
        assert weighted_total(strong_hook, WEIGHTS["weights"]) > weighted_total(
            strong_visual, WEIGHTS["weights"]
        )

    def test_weights_that_do_not_sum_to_one_still_scale_correctly(self) -> None:
        """Normalizing by the weights used keeps scores comparable."""
        assert weighted_total(full_rubric(8.0), {"hook": 2.0, "payoff": 3.0}) == pytest.approx(8.0)

    def test_missing_axis_is_skipped_not_zeroed(self) -> None:
        partial = {"hook": 9.0}
        assert weighted_total(partial, WEIGHTS["weights"]) == pytest.approx(9.0)

    def test_empty_rubric_is_zero(self) -> None:
        assert weighted_total({}, WEIGHTS["weights"]) == 0.0


class TestApplyScores:
    def clips(self, *, word_count: int = 100, span: tuple[int, int] = (0, 99)) -> list[ScoredClip]:
        return build_clips([candidate(*span)], transcript_of(word_count))

    def test_records_the_raw_rubric(self) -> None:
        clips = self.clips()
        payload = {
            "scores": [{"clip_id": "clip_01", "rubric": full_rubric(7.0), "suggested_title": "t"}]
        }
        apply_scores(clips, payload, WEIGHTS)
        assert clips[0].rubric == full_rubric(7.0)

    def test_clamps_out_of_range_axes(self) -> None:
        clips = self.clips()
        payload = {
            "scores": [
                {
                    "clip_id": "clip_01",
                    "rubric": full_rubric(5.0) | {"hook": 99.0, "payoff": -3.0},
                    "suggested_title": "t",
                }
            ]
        }
        apply_scores(clips, payload, WEIGHTS)
        assert clips[0].rubric["hook"] == 10.0
        assert clips[0].rubric["payoff"] == 1.0

    def test_coerces_a_non_numeric_axis(self) -> None:
        clips = self.clips()
        payload = {
            "scores": [
                {
                    "clip_id": "clip_01",
                    "rubric": full_rubric(5.0) | {"hook": "great"},
                    "suggested_title": "t",
                }
            ]
        }
        apply_scores(clips, payload, WEIGHTS)
        assert clips[0].rubric["hook"] == 1.0

    def test_length_penalty_applies_under_the_minimum(self) -> None:
        # 20 words at 0.4s = 8s, well under the 25s floor.
        clips = build_clips([candidate(0, 19)], transcript_of(100))
        payload = {
            "scores": [{"clip_id": "clip_01", "rubric": full_rubric(8.0), "suggested_title": "t"}]
        }
        apply_scores(clips, payload, WEIGHTS, clip_length={"min": 25, "max": 55})
        assert clips[0].modifiers == {"under_min_s": 0.85}
        assert clips[0].score == pytest.approx(clips[0].base_score * 0.85)

    def test_length_penalty_applies_over_the_maximum(self) -> None:
        # 300 words at 0.4s = 120s.
        clips = build_clips([candidate(0, 299)], transcript_of(400))
        payload = {
            "scores": [{"clip_id": "clip_01", "rubric": full_rubric(8.0), "suggested_title": "t"}]
        }
        apply_scores(clips, payload, WEIGHTS, clip_length={"min": 25, "max": 55})
        assert clips[0].modifiers == {"over_max_s": 0.80}

    def test_no_penalty_inside_the_target_range(self) -> None:
        # 100 words at 0.4s = 40s.
        clips = build_clips([candidate(0, 99)], transcript_of(200))
        payload = {
            "scores": [{"clip_id": "clip_01", "rubric": full_rubric(8.0), "suggested_title": "t"}]
        }
        apply_scores(clips, payload, WEIGHTS, clip_length={"min": 25, "max": 55})
        assert clips[0].modifiers == {}
        assert clips[0].score == pytest.approx(clips[0].base_score)

    def test_speaker_change_penalty(self) -> None:
        transcript = Transcript(
            words=[
                Word(w=f"w{i}", start=i * 0.4, end=(i + 1) * 0.4, speaker="S0" if i < 50 else "S1")
                for i in range(100)
            ],
            duration_s=40.0,
        )
        clips = build_clips([candidate(0, 99)], transcript)
        payload = {
            "scores": [{"clip_id": "clip_01", "rubric": full_rubric(8.0), "suggested_title": "t"}]
        }
        apply_scores(clips, payload, WEIGHTS, clip_length={"min": 25, "max": 55})
        assert clips[0].modifiers == {"speaker_change": 0.95}

    def test_filters_emphasis_words_not_in_the_clip(self) -> None:
        """A word that is not in the span would silently never highlight."""
        clips = self.clips()
        payload = {
            "scores": [
                {
                    "clip_id": "clip_01",
                    "rubric": full_rubric(7.0),
                    "suggested_title": "t",
                    "emphasis_words": ["word3", "nonsense", "WORD7"],
                }
            ]
        }
        apply_scores(clips, payload, WEIGHTS)
        assert clips[0].emphasis_words == ["word3", "WORD7"]

    def test_unscored_clip_is_kept_with_a_zero_score(self) -> None:
        """ "Considered and not scored" is a different fact from "scored badly"."""
        clips = self.clips()
        apply_scores(clips, {"scores": []}, WEIGHTS)
        assert clips[0].score == 0.0
        assert clips[0].rubric == {}

    def test_ignores_an_unknown_clip_id(self) -> None:
        clips = self.clips()
        payload = {
            "scores": [{"clip_id": "clip_99", "rubric": full_rubric(9.0), "suggested_title": "t"}]
        }
        apply_scores(clips, payload, WEIGHTS)
        assert clips[0].score == 0.0

    def test_blank_title_becomes_none(self) -> None:
        clips = self.clips()
        payload = {
            "scores": [{"clip_id": "clip_01", "rubric": full_rubric(7.0), "suggested_title": "   "}]
        }
        apply_scores(clips, payload, WEIGHTS)
        assert clips[0].suggested_title is None


class TestRun:
    def test_uses_the_rerank_tier_and_returns_ranked_clips(self) -> None:
        transcript = transcript_of(600)
        candidates = [candidate(i * 100, i * 100 + 99) for i in range(5)]
        payload = {
            "scores": [
                {
                    "clip_id": f"clip_{i + 1:02d}",
                    "rubric": full_rubric(float(i + 4)),
                    "suggested_title": f"title {i}",
                }
                for i in range(5)
            ]
        }
        router = FakeRouter(payload)
        scored = run(candidates, transcript, router, WEIGHTS, top_n=20)

        assert router.calls[0]["tier"] == "rerank"
        assert [c.clip_id for c in scored][0] == "clip_05"
        assert scored == sorted(scored, key=lambda c: c.score, reverse=True)

    def test_makes_one_call_for_all_candidates(self) -> None:
        transcript = transcript_of(600)
        candidates = [candidate(i * 100, i * 100 + 99) for i in range(5)]
        router = FakeRouter({"scores": []})
        run(candidates, transcript, router, WEIGHTS)
        assert len(router.calls) == 1

    def test_no_candidates_makes_no_call(self) -> None:
        router = FakeRouter({"scores": []})
        assert run([], transcript_of(100), router, WEIGHTS) == []
        assert router.calls == []


class TestShortlist:
    def make(self, score_value: float, index: int) -> ScoredClip:
        return ScoredClip(
            clip_id=f"clip_{index:02d}", candidate=candidate(0, 10), score=score_value
        )

    def test_drops_clips_below_min_score(self) -> None:
        clips = [self.make(9.0, 1), self.make(2.0, 2)]
        assert [c.clip_id for c in shortlist(clips, WEIGHTS)] == ["clip_01"]

    def test_caps_at_max_clips_per_video(self) -> None:
        clips = [self.make(9.0, i) for i in range(10)]
        assert len(shortlist(clips, WEIGHTS)) == 3

    def test_returns_the_highest_scoring(self) -> None:
        clips = [self.make(float(i), i) for i in range(10)]
        assert [c.score for c in shortlist(clips, WEIGHTS)] == [9.0, 8.0, 7.0]


class TestPersistence:
    def test_archives_the_raw_rubric_and_the_weights(self, tmp_path) -> None:
        """The calibration loop re-fits against these; a weighted total alone
        would throw away the only real edge this project has."""
        clips = build_clips([candidate(0, 99)], transcript_of(200))
        apply_scores(
            clips,
            {
                "scores": [
                    {"clip_id": "clip_01", "rubric": full_rubric(7.0), "suggested_title": "t"}
                ]
            },
            WEIGHTS,
        )
        save(clips, tmp_path, weights_config=WEIGHTS)

        payload = load(tmp_path)
        assert payload["weights"] == WEIGHTS["weights"]
        assert payload["clips"][0]["rubric"] == full_rubric(7.0)
        assert "base_score" in payload["clips"][0]

    def test_records_the_span_and_the_text(self, tmp_path) -> None:
        clips = build_clips([candidate(5, 40)], transcript_of(200))
        save(clips, tmp_path)
        entry = json.loads((tmp_path / "scored.json").read_text(encoding="utf-8"))["clips"][0]
        assert entry["start_word_index"] == 5
        assert entry["end_word_index"] == 40
        assert entry["text"].startswith("word5 ")

    def test_missing_file_names_the_stage(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="clipper moments"):
            load(tmp_path)


class TestRenderCandidates:
    def test_includes_the_full_text_of_each_clip(self) -> None:
        clips = build_clips([candidate(0, 20)], transcript_of(100))
        rendered = render_candidates(clips)
        assert "clip_01" in rendered
        assert "word19" in rendered

    def test_flags_a_speaker_change(self) -> None:
        transcript = Transcript(
            words=[
                Word(w=f"w{i}", start=i * 0.4, end=(i + 1) * 0.4, speaker="S0" if i < 10 else "S1")
                for i in range(30)
            ]
        )
        rendered = render_candidates(build_clips([candidate(0, 29)], transcript))
        assert "crosses a speaker change" in rendered
