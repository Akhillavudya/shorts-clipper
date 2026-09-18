"""The single LLM call in discovery (FULL_PLAN.md section 3, step 7).

Responsibility: take the top ~15 candidates by velocity score and return, for
each, a short "why this is interesting" blurb plus probable moment types. Runs
on the `recall` tier via clipper.llm.router.

Deliberately last in the pipeline: everything cheap and deterministic runs
first, so the model only ever sees a pre-filtered shortlist.

Output: a markdown table to stdout for human review, plus a JSON file.

The model is given titles, channels, durations and the opening of each
description - never a transcript, because at this point no transcript exists
and downloading fifteen videos to find three worth clipping is the cost this
whole stage exists to avoid. It is therefore asked for a judgement it can
actually make from that evidence, and asked to say ``low`` confidence when the
evidence is thin. A blurb is a reading aid for the human who picks; the
velocity number is the ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clipper.discover import prompts
from clipper.discover.velocity import VelocityScore
from clipper.llm import Router

#: The moment vocabulary from FULL_PLAN.md section 5, so a discovery blurb and
#: a pass-1 candidate speak the same language.
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

CONFIDENCE_LEVELS = ("high", "medium", "low")

#: How much of each description the model sees. Enough for a show to state its
#: topic and guest; short of the sponsor block and the timestamp list that
#: follow, which are pure token cost.
DESCRIPTION_CHARS = 400

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shortlist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "why": {"type": "string"},
                    "probable_moments": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(MOMENT_TYPES)},
                    },
                    "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                    "skip_reason": {"type": "string"},
                },
                "required": ["video_id", "why", "probable_moments"],
            },
        }
    },
    "required": ["shortlist"],
}


@dataclass
class Candidate:
    """One shortlisted video: the arithmetic, the rights status, the blurb.

    Rights status rides along from the watchlist rather than being looked up by
    a reader later. A shortlist that does not say which entries can actually be
    rendered invites exactly the mistake the gate exists to prevent.
    """

    score: VelocityScore
    rights_status: str = "unknown"
    rights_note: str | None = None
    why: str | None = None
    probable_moments: list[str] = field(default_factory=list)
    confidence: str | None = None
    skip_reason: str | None = None

    @property
    def video_id(self) -> str:
        return self.score.record.video_id

    @property
    def title(self) -> str:
        return self.score.record.title or "(untitled)"

    @property
    def channel(self) -> str:
        return self.score.record.channel_title or self.score.record.channel_id or "(unknown)"

    @property
    def url(self) -> str:
        return self.score.record.url

    @property
    def duration_min(self) -> int:
        return int((self.score.record.duration_s or 0) / 60)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.score.to_dict(),
            "rights_status": self.rights_status,
            "rights_note": self.rights_note,
            "why": self.why,
            "probable_moments": self.probable_moments,
            "confidence": self.confidence,
            "skip_reason": self.skip_reason,
        }


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def render_candidates(candidates: list[Candidate]) -> str:
    """The candidate block for the prompt.

    One compact record per video. The velocity number is included because the
    model is explicitly told not to re-rank on it - seeing it stops the blurb
    claiming a video is a breakout when it is scoring 1.1.
    """
    blocks: list[str] = []
    for candidate in candidates:
        record = candidate.score.record
        blocks.append(
            f"### {candidate.video_id}\n"
            f"title: {candidate.title}\n"
            f"channel: {candidate.channel}\n"
            f"duration: {candidate.duration_min} min | "
            f"velocity: {candidate.score.velocity:.2f}x | "
            f"age: {candidate.score.age_hours:.0f}h\n"
            f"description: {_truncate(record.description, DESCRIPTION_CHARS) or '(none)'}\n"
        )
    return "\n".join(blocks)


def apply_blurbs(candidates: list[Candidate], payload: dict[str, Any]) -> list[Candidate]:
    """Fold the model's response into the candidates.

    A candidate the model skipped keeps ``why=None`` and still appears in the
    output. The velocity score is the reason it is on the list; the blurb is
    commentary, and missing commentary is not grounds for hiding a candidate
    from the person doing the picking.
    """
    by_id = {c.video_id: c for c in candidates}

    for raw in payload.get("shortlist") or []:
        if not isinstance(raw, dict):
            continue
        candidate = by_id.get(str(raw.get("video_id") or ""))
        if candidate is None:
            continue

        candidate.why = _truncate(str(raw.get("why") or ""), 200) or None

        # Anything outside the fixed vocabulary is dropped rather than passed
        # through: these labels are meant to match the pass-1 `type` values, and
        # a one-off invented label would read as a category that does not exist.
        candidate.probable_moments = [
            str(m).strip().lower()
            for m in raw.get("probable_moments") or []
            if str(m).strip().lower() in MOMENT_TYPES
        ][:4]

        confidence = str(raw.get("confidence") or "").strip().lower()
        candidate.confidence = confidence if confidence in CONFIDENCE_LEVELS else None

        skip = str(raw.get("skip_reason") or "").strip()
        candidate.skip_reason = skip or None

    return candidates


def run(
    candidates: list[Candidate],
    router: Router,
    *,
    channel_count: int = 0,
) -> list[Candidate]:
    """Annotate the shortlist with one ``recall``-tier call.

    Returns the candidates unchanged if the list is empty - there is nothing to
    ask about, and an empty call still costs a request against the daily budget.
    """
    if not candidates:
        return candidates

    prompt = prompts.render(
        "discover.user",
        channel_count=channel_count,
        candidate_count=len(candidates),
        candidates=render_candidates(candidates),
    )
    payload = router.complete(
        "recall",
        prompt,
        system=prompts.load("discover.system"),
        json_schema=RESPONSE_SCHEMA,
        temperature=0.2,
    )
    return apply_blurbs(candidates, payload)


# -- output ---------------------------------------------------------------


def _cell(text: str | None, limit: int) -> str:
    """Markdown table cells cannot contain a raw pipe or a newline."""
    return _truncate((text or "-").replace("|", "/"), limit) or "-"


def markdown_table(candidates: list[Candidate]) -> str:
    """The table from section 3: title, channel, link, why, moments, popularity."""
    if not candidates:
        return "_no candidates cleared the velocity filter._"

    lines = [
        "| # | title | channel | rights | velocity | len | probable moments | why | link |",
        "|--:|---|---|---|--:|--:|---|---|---|",
    ]
    for index, candidate in enumerate(candidates, start=1):
        moments = ", ".join(candidate.probable_moments) or "-"
        why = candidate.why or "-"
        if candidate.skip_reason:
            why = f"SKIP: {candidate.skip_reason}"
        lines.append(
            f"| {index} "
            f"| {_cell(candidate.title, 70)} "
            f"| {_cell(candidate.channel, 28)} "
            f"| {candidate.rights_status} "
            f"| {candidate.score.velocity:.2f}x "
            f"| {candidate.duration_min}m "
            f"| {_cell(moments, 40)} "
            f"| {_cell(why, 120)} "
            f"| {candidate.url} |"
        )
    return "\n".join(lines)


def report_path(directory: Path | str, *, now: datetime | None = None) -> Path:
    """``<dir>/discover-YYYY-MM-DD.json``.

    Dated rather than overwritten: yesterday's shortlist is what tells you
    whether the velocity threshold is picking the right things, and it costs
    a few kilobytes to keep.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
    return Path(directory) / f"discover-{stamp}.json"


def save(
    candidates: list[Candidate],
    directory: Path | str,
    *,
    quota: dict[str, Any] | None = None,
    skipped: dict[str, int] | None = None,
    errors: dict[str, str] | None = None,
    episodes: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> Path:
    """Write the JSON half of the output."""
    path = report_path(directory, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": (now or datetime.now(UTC)).isoformat(),
        "quota": quota or {},
        "skipped": skipped or {},
        "errors": errors or {},
        "podcast_episodes": episodes or [],
        "candidates": [c.to_dict() for c in candidates],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "CONFIDENCE_LEVELS",
    "DESCRIPTION_CHARS",
    "MOMENT_TYPES",
    "RESPONSE_SCHEMA",
    "Candidate",
    "apply_blurbs",
    "markdown_table",
    "render_candidates",
    "report_path",
    "run",
    "save",
]
