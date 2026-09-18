"""The one LLM call in discovery, and the report it produces.

The router is faked the way ``tests/test_router.py`` fakes it - no key, no
network. What is being checked is that the model's answer is folded in without
being trusted: an invented moment label is dropped, an unknown video id is
ignored, and a candidate the model skipped still appears in the table.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from clipper.discover.podcast_index import auth_headers
from clipper.discover.summarize import (
    Candidate,
    apply_blurbs,
    markdown_table,
    render_candidates,
    report_path,
    run,
    save,
)
from clipper.discover.velocity import ChannelBaseline, VelocityScore
from clipper.discover.youtube import VideoRecord

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def candidate(video_id: str, *, title: str = "A title", velocity: float = 2.0) -> Candidate:
    record = VideoRecord(
        video_id=video_id,
        channel_id="UC_A",
        channel_title="Example Channel",
        title=title,
        description="A long description. " * 60,
        published_at=NOW - timedelta(hours=24),
        duration_s=5400.0,
        view_count=10_000,
    )
    return Candidate(
        score=VelocityScore(
            record=record,
            vph=416.0,
            baseline=ChannelBaseline("UC_A", 208.0, 6, True),
            velocity=velocity,
            age_hours=24.0,
        ),
        rights_status="green",
    )


class FakeRouter:
    """Returns a canned payload and records what it was asked."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def complete(self, tier: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"tier": tier, "prompt": prompt, **kwargs})
        return self.payload


class TestPrompt:
    def test_descriptions_are_truncated(self) -> None:
        rendered = render_candidates([candidate("v1")])
        # The raw description is ~1200 characters; the sponsor block and
        # timestamp list that follow a real one are pure token cost.
        assert len(rendered) < 900
        assert "…" in rendered

    def test_velocity_is_shown_to_the_model(self) -> None:
        rendered = render_candidates([candidate("v1", velocity=3.5)])
        assert "3.50x" in rendered

    def test_runs_on_the_recall_tier(self) -> None:
        router = FakeRouter({"shortlist": []})
        run([candidate("v1")], router, channel_count=4)
        assert router.calls[0]["tier"] == "recall"
        assert "4" in router.calls[0]["prompt"]

    def test_an_empty_shortlist_makes_no_call(self) -> None:
        router = FakeRouter({"shortlist": []})
        assert run([], router) == []
        assert router.calls == []


class TestApplyBlurbs:
    def test_fields_are_folded_in(self) -> None:
        candidates = [candidate("v1")]
        apply_blurbs(
            candidates,
            {
                "shortlist": [
                    {
                        "video_id": "v1",
                        "why": "A specific claim in the title",
                        "probable_moments": ["contrarian", "insight"],
                        "confidence": "medium",
                    }
                ]
            },
        )
        assert candidates[0].why == "A specific claim in the title"
        assert candidates[0].probable_moments == ["contrarian", "insight"]
        assert candidates[0].confidence == "medium"

    def test_invented_moment_labels_are_dropped(self) -> None:
        candidates = [candidate("v1")]
        apply_blurbs(
            candidates,
            {
                "shortlist": [
                    {
                        "video_id": "v1",
                        "why": "-",
                        "probable_moments": ["insight", "vibes", "banger"],
                    }
                ]
            },
        )
        assert candidates[0].probable_moments == ["insight"]

    def test_an_unknown_confidence_becomes_none(self) -> None:
        candidates = [candidate("v1")]
        apply_blurbs(
            candidates,
            {"shortlist": [{"video_id": "v1", "why": "x", "confidence": "extremely"}]},
        )
        assert candidates[0].confidence is None

    def test_an_unknown_video_id_is_ignored(self) -> None:
        candidates = [candidate("v1")]
        apply_blurbs(candidates, {"shortlist": [{"video_id": "hallucinated", "why": "x"}]})
        assert candidates[0].why is None

    def test_a_skipped_candidate_still_appears(self) -> None:
        # The velocity score is the reason it is on the list; a missing blurb
        # is not grounds for hiding it from the person picking.
        candidates = [candidate("v1"), candidate("v2")]
        apply_blurbs(candidates, {"shortlist": [{"video_id": "v1", "why": "good"}]})

        table = markdown_table(candidates)
        assert "v1" in table and "v2" in table


class TestMarkdownTable:
    def test_pipes_in_a_title_do_not_break_the_table(self) -> None:
        table = markdown_table([candidate("v1", title="Ep 12 | The one about pipes")])
        rows = [line for line in table.splitlines() if line.startswith("| 1 ")]
        assert len(rows) == 1
        assert rows[0].count("|") == 10, "one pipe per column boundary, none from the title"

    def test_empty_is_a_sentence_not_a_broken_table(self) -> None:
        assert "no candidates" in markdown_table([])

    def test_rights_status_is_visible(self) -> None:
        assert "green" in markdown_table([candidate("v1")])

    def test_a_skip_reason_replaces_the_why(self) -> None:
        candidates = [candidate("v1")]
        apply_blurbs(
            candidates,
            {"shortlist": [{"video_id": "v1", "why": "x", "skip_reason": "highlight reel"}]},
        )
        assert "SKIP: highlight reel" in markdown_table(candidates)


class TestSave:
    def test_report_is_dated_and_complete(self, tmp_path) -> None:
        path = save(
            [candidate("v1")],
            tmp_path,
            quota={"spent_this_run": 5},
            skipped={"too_new": 2},
            now=NOW,
        )
        assert path == report_path(tmp_path, now=NOW)
        assert path.name == "discover-2026-09-13.json"

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["quota"]["spent_this_run"] == 5
        assert payload["skipped"]["too_new"] == 2
        assert payload["candidates"][0]["video_id"] == "v1"
        assert payload["candidates"][0]["velocity"] == 2.0
        assert payload["candidates"][0]["rights_status"] == "green"


class TestPodcastIndexAuth:
    def test_signature_is_sha1_of_key_secret_and_date(self) -> None:
        import hashlib

        headers = auth_headers("KEY", "SECRET", now=1_757_764_800)
        expected = hashlib.sha1(b"KEYSECRET1757764800").hexdigest()  # noqa: S324

        assert headers["Authorization"] == expected
        assert headers["X-Auth-Key"] == "KEY"
        assert headers["X-Auth-Date"] == "1757764800"
        assert headers["User-Agent"], "the API rejects a request with no User-Agent"

    def test_the_signature_changes_with_the_timestamp(self) -> None:
        # The date is signed, so a cached header block goes stale. Recomputing
        # per call is the reason, not caution.
        first = auth_headers("K", "S", now=1_000_000)
        second = auth_headers("K", "S", now=1_000_060)
        assert first["Authorization"] != second["Authorization"]
