"""Pass 2 - precision rerank (FULL_PLAN.md section 5).

Responsibility: take the top ~20 candidates from pass 1 (deduped, and merged
where they overlap) and score each against a fixed rubric, 1-10 per axis. Runs
on the stronger `rerank` tier; the input is small, so this stays cheap.

Rubric:

- hook                - does the first sentence stop a scroll on its own?
- self_containment    - does it make sense with zero context about the episode?
- payoff              - is there a resolution, not just a setup?
- surprise_emotion    - is there a reason to feel something?
- visual_independence - does it work when the video is just a talking head?
- quotability         - is there one line a viewer would repeat?
- length_fit          - natural boundary in the 25-55s range?

Plus suggested_title, emphasis_words (drives caption highlighting) and
hook_rewrite (on-screen text overlay for the first 2s).

Weights come from config/scoring_weights.yaml and the weighted sum is computed
in Python, not by the model. Every raw rubric score is persisted to
jobs/<video_id>/scored.json - that record is what the calibration loop later
correlates against real retention.

Keeping the arithmetic here rather than in the prompt is not a style choice.
The calibration loop re-fits the weights against real retention and re-scores
the archive; that is only possible because ``scored.json`` holds the raw axes,
so a new weighting is a dict lookup rather than a re-run of every LLM call
ever made (CLAUDE.md hard rule 8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipper.llm import Router
from clipper.moments import prompts
from clipper.moments.recall import Candidate
from clipper.transcribe.schema import Transcript

SCORED_FILENAME = "scored.json"

#: The rubric. Order is the order a reviewer reads them in.
RUBRIC_AXES = (
    "hook",
    "self_containment",
    "payoff",
    "surprise_emotion",
    "visual_independence",
    "quotability",
    "length_fit",
)

#: Used when scoring_weights.yaml is missing an axis entirely. Equal weighting
#: is the honest default - it says "we have no opinion", which beats silently
#: dropping an axis from the sum.
FALLBACK_WEIGHT = 1.0 / len(RUBRIC_AXES)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip_id": {"type": "string"},
                    "rubric": {
                        "type": "object",
                        "properties": {axis: {"type": "number"} for axis in RUBRIC_AXES},
                        "required": list(RUBRIC_AXES),
                    },
                    "suggested_title": {"type": "string"},
                    "emphasis_words": {"type": "array", "items": {"type": "string"}},
                    "hook_rewrite": {"type": "string"},
                },
                "required": ["clip_id", "rubric", "suggested_title"],
            },
        }
    },
    "required": ["scores"],
}


@dataclass
class ScoredClip:
    """One candidate with its raw rubric scores and the weighted total.

    ``rubric`` is the archive. ``score`` is derived and may be recomputed at
    any time from ``rubric`` plus a weights file - which is the entire point.
    """

    clip_id: str
    candidate: Candidate
    rubric: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    base_score: float = 0.0
    suggested_title: str | None = None
    emphasis_words: list[str] = field(default_factory=list)
    hook_rewrite: str | None = None
    modifiers: dict[str, float] = field(default_factory=dict)
    duration_s: float | None = None
    speaker_change: bool = False
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "start_word_index": self.candidate.start_word_index,
            "end_word_index": self.candidate.end_word_index,
            "type": self.candidate.type,
            "one_line_reason": self.candidate.one_line_reason,
            "rubric": self.rubric,
            "base_score": round(self.base_score, 4),
            "modifiers": self.modifiers,
            "score": round(self.score, 4),
            "suggested_title": self.suggested_title,
            "emphasis_words": self.emphasis_words,
            "hook_rewrite": self.hook_rewrite,
            "duration_s": self.duration_s,
            "speaker_change": self.speaker_change,
            "text": self.text,
        }


def select_for_rerank(candidates: list[Candidate], top_n: int) -> list[Candidate]:
    """Cut pass 1's output down to what pass 2 will score.

    Pass 1 returns no ranking, so there is no "top" to take - and taking the
    first N would score the opening twenty minutes of an episode and ignore the
    rest. Instead the list is thinned evenly across the timeline, which
    preserves coverage. Order is by position, so a reviewer reads them in the
    order they occur.
    """
    ordered = sorted(candidates, key=lambda c: c.start_word_index)
    if top_n <= 0 or len(ordered) <= top_n:
        return ordered

    step = len(ordered) / top_n
    picked = [ordered[min(len(ordered) - 1, int(i * step))] for i in range(top_n)]

    # Even spacing can land twice on the same candidate when the list is barely
    # over the limit; fill any shortfall with what was skipped.
    seen = {id(c) for c in picked}
    for candidate in ordered:
        if len(picked) >= top_n:
            break
        if id(candidate) not in seen:
            picked.append(candidate)
            seen.add(id(candidate))

    return sorted(picked, key=lambda c: c.start_word_index)


def render_candidates(candidates: list[ScoredClip]) -> str:
    """Format the candidate block for the rerank prompt.

    The transcript text is included in full. Pass 2 is judging writing quality -
    whether the first sentence lands, whether a line is quotable - and it
    cannot do that from a summary.
    """
    blocks: list[str] = []
    for clip in candidates:
        duration = f"{clip.duration_s:.0f}s" if clip.duration_s else "unknown length"
        speakers = " (crosses a speaker change)" if clip.speaker_change else ""
        blocks.append(
            f"### {clip.clip_id}\n"
            f"type: {clip.candidate.type} | {duration}{speakers}\n"
            f"pass-1 note: {clip.candidate.one_line_reason or '(none)'}\n\n"
            f"{clip.text}\n"
        )
    return "\n".join(blocks)


def build_clips(candidates: list[Candidate], transcript: Transcript) -> list[ScoredClip]:
    """Wrap candidates with the derived facts the prompt and modifiers need."""
    clips: list[ScoredClip] = []
    for index, candidate in enumerate(candidates, start=1):
        start_s, end_s = transcript.span_seconds(
            candidate.start_word_index, candidate.end_word_index
        )
        clips.append(
            ScoredClip(
                clip_id=f"clip_{index:02d}",
                candidate=candidate,
                duration_s=max(0.0, end_s - start_s),
                speaker_change=transcript.has_speaker_change(
                    candidate.start_word_index, candidate.end_word_index
                ),
                text=transcript.text_for(candidate.start_word_index, candidate.end_word_index),
            )
        )
    return clips


def _clamp_axis(value: Any) -> float:
    """Rubric values are 1-10. Anything else is coerced into range.

    Clamping rather than dropping: a model that returns 0 or 12 still expressed
    "bad" or "great", and the axis is one of seven, so a coerced value distorts
    the total far less than a missing one would.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(1.0, min(10.0, number))


def weighted_total(rubric: dict[str, float], weights: dict[str, float]) -> float:
    """The weighted sum, on the same 1-10 scale as the axes.

    Normalized by the weights actually used, so a weights file that omits an
    axis or does not sum to 1.0 still produces a comparable number instead of
    quietly shrinking every score.
    """
    total = 0.0
    used = 0.0
    for axis in RUBRIC_AXES:
        if axis not in rubric:
            continue
        weight = float(weights.get(axis, FALLBACK_WEIGHT))
        total += rubric[axis] * weight
        used += weight
    return total / used if used else 0.0


def apply_modifiers(
    clip: ScoredClip, config: dict[str, Any], *, clip_length: dict[str, Any]
) -> dict[str, float]:
    """Length and speaker-change penalties, applied after the weighted sum.

    Penalties rather than filters: a great 70-second moment should still
    surface for manual trimming, just below an equally good 40-second one.
    """
    modifiers: dict[str, float] = {}
    length_cfg = config.get("length_penalty") or {}

    if length_cfg.get("enabled", True) and clip.duration_s:
        min_s = float(clip_length.get("min", 25))
        max_s = float(clip_length.get("max", 55))
        if clip.duration_s < min_s:
            modifiers["under_min_s"] = float(length_cfg.get("under_min_s", 0.85))
        elif clip.duration_s > max_s:
            modifiers["over_max_s"] = float(length_cfg.get("over_max_s", 0.80))

    if clip.speaker_change:
        penalty = float(config.get("speaker_change_penalty", 0.95))
        if penalty != 1.0:
            modifiers["speaker_change"] = penalty

    return modifiers


def apply_scores(
    clips: list[ScoredClip],
    payload: dict[str, Any],
    weights_config: dict[str, Any],
    *,
    clip_length: dict[str, Any] | None = None,
) -> list[ScoredClip]:
    """Fold the model's response into the clips and compute the totals.

    A clip the model skipped keeps ``score=0`` and an empty rubric. It is not
    dropped: it stays visible in ``scored.json`` as something that was
    considered and not scored, which is a different fact from "scored badly".
    """
    weights = weights_config.get("weights") or {}
    modifier_config = weights_config.get("modifiers") or {}
    clip_length = clip_length or {"min": 25, "max": 55}

    by_id = {clip.clip_id: clip for clip in clips}

    for raw in payload.get("scores") or []:
        if not isinstance(raw, dict):
            continue
        clip = by_id.get(str(raw.get("clip_id") or ""))
        if clip is None:
            continue

        raw_rubric = raw.get("rubric") or {}
        clip.rubric = {axis: _clamp_axis(raw_rubric.get(axis)) for axis in RUBRIC_AXES}
        clip.base_score = weighted_total(clip.rubric, weights)

        clip.modifiers = apply_modifiers(clip, modifier_config, clip_length=clip_length)
        score = clip.base_score
        for factor in clip.modifiers.values():
            score *= factor
        clip.score = score

        title = str(raw.get("suggested_title") or "").strip()
        clip.suggested_title = title or None
        hook = str(raw.get("hook_rewrite") or "").strip()
        clip.hook_rewrite = hook or None

        # Emphasis words only work if they are literally in the clip; a word
        # that is not there simply never highlights, so filtering here turns a
        # silent no-op into something visible in scored.json.
        spoken = {w.strip(".,!?;:\"'").lower() for w in clip.text.split()}
        clip.emphasis_words = [
            word
            for word in (str(w).strip() for w in raw.get("emphasis_words") or [])
            if word and word.strip(".,!?;:\"'").lower() in spoken
        ]

    return clips


def run(
    candidates: list[Candidate],
    transcript: Transcript,
    router: Router,
    weights_config: dict[str, Any],
    *,
    title: str | None = None,
    top_n: int = 20,
    clip_length: dict[str, Any] | None = None,
) -> list[ScoredClip]:
    """Score candidates and return them ranked, highest first."""
    selected = select_for_rerank(candidates, top_n)
    if not selected:
        return []

    clips = build_clips(selected, transcript)
    duration_min = int((transcript.duration_s or 0) / 60) or 1

    prompt = prompts.render(
        "rerank.user",
        title=title or "(untitled)",
        candidate_count=len(clips),
        duration_min=duration_min,
        candidates=render_candidates(clips),
    )
    payload = router.complete(
        "rerank",
        prompt,
        system=prompts.load("rerank.system"),
        json_schema=RESPONSE_SCHEMA,
        temperature=0.1,
    )

    apply_scores(clips, payload, weights_config, clip_length=clip_length)
    return sorted(clips, key=lambda c: c.score, reverse=True)


def shortlist(clips: list[ScoredClip], weights_config: dict[str, Any]) -> list[ScoredClip]:
    """The clips worth rendering: above ``min_score``, capped per video."""
    min_score = float(weights_config.get("min_score", 5.5))
    max_clips = int(weights_config.get("max_clips_per_video", 3))
    passing = [c for c in clips if c.score >= min_score]
    return sorted(passing, key=lambda c: c.score, reverse=True)[:max_clips]


# -- persistence ----------------------------------------------------------


def scored_path(job_dir: Path | str) -> Path:
    return Path(job_dir) / SCORED_FILENAME


def save(
    clips: list[ScoredClip], job_dir: Path | str, *, weights_config: dict[str, Any] | None = None
) -> Path:
    """Write ``scored.json``. Keep this file forever (CLAUDE.md hard rule 8).

    The weights in force at scoring time are written alongside the raw scores.
    Without them a re-fit cannot tell whether a historical total came from a
    different weighting or a different clip, and the correlation is worthless.
    """
    path = scored_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": (weights_config or {}).get("weights") or {},
        "modifiers": (weights_config or {}).get("modifiers") or {},
        "clips": [c.to_dict() for c in clips],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(job_dir: Path | str) -> dict[str, Any]:
    """Read ``scored.json`` back as raw JSON, for the calibration loop."""
    path = scored_path(job_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no scores at {path} - run `clipper moments` first")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "FALLBACK_WEIGHT",
    "RESPONSE_SCHEMA",
    "RUBRIC_AXES",
    "SCORED_FILENAME",
    "ScoredClip",
    "apply_modifiers",
    "apply_scores",
    "build_clips",
    "load",
    "render_candidates",
    "run",
    "save",
    "scored_path",
    "select_for_rerank",
    "shortlist",
    "weighted_total",
]
