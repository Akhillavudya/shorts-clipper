"""Upload metadata: the title, the credit block, and the gate.

Two things here are load-bearing rather than cosmetic. The gate must refuse a
red or unknown source, because a ready-made title and description for a source
that may not be clipped is one copy-paste from an upload. And the description
must never read as the creator's own account - impersonation is judged
separately from copyright, and a credit block that gets it wrong fails both.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from clipper.publish.metadata import (
    DESCRIPTION_LIMIT,
    DISCLAIMER,
    TITLE_LIMIT,
    build,
    build_description,
    build_title,
    credit_block,
    record_published,
    save,
    truncate_words,
)
from clipper.rights.registry import Watchlist
from clipper.state import JobState, RightsBlocked, RightsStatus


@pytest.fixture
def watchlist(tmp_path) -> Watchlist:
    path = tmp_path / "watchlist.yaml"
    path.write_text(
        textwrap.dedent(
            """
            defaults:
              max_per_day: 2
            channels:
              - id: UC_GREEN
                name: Example Creator
                rights_status: green
                note: Public clipping policy.
                topics: [startups, engineering]
              - id: UC_AMBER
                name: Permissioned Creator
                rights_status: amber
                note: Said yes by email.
                permission_file: rights/permissions/UC_AMBER.md
                credit_handle: "@thecreator"
            """
        ),
        encoding="utf-8",
    )
    return Watchlist.load(path)


def job(
    *,
    status: RightsStatus = RightsStatus.green,
    channel_id: str = "UC_GREEN",
    licence: str | None = None,
) -> JobState:
    state = JobState(
        video_id="vid123",
        source_url="https://www.youtube.com/watch?v=vid123",
        title="Episode 41: the one about compilers",
        channel_id=channel_id,
        channel_title="Example Creator",
        licence=licence,
    )
    state.rights_status = status
    state.rights_note = "test fixture"
    return state


def clip_on(state: JobState, **fields):
    defaults = {
        "suggested_title": "compilers are just very patient interpreters",
        "hook_rewrite": "compilers are patient interpreters",
        "type": "insight",
        "score": 7.4,
    }
    return state.add_clip(**{**defaults, **fields})


class TestTruncateWords:
    def test_short_text_is_untouched(self) -> None:
        assert truncate_words("hello world", 100) == ("hello world", False)

    def test_cuts_on_a_word_boundary(self) -> None:
        text, cut = truncate_words("the quick brown fox jumps", 13)
        assert text == "the quick"
        assert cut is True

    def test_whitespace_is_collapsed(self) -> None:
        assert truncate_words("a\n\n  b", 100)[0] == "a b"


class TestTitle:
    def test_uses_the_rerank_suggestion(self) -> None:
        state = job()
        title, _ = build_title(clip_on(state), state)
        assert title.startswith("compilers are just very patient interpreters")
        assert title.endswith("#Shorts")

    def test_stays_within_the_limit(self) -> None:
        state = job()
        clip = clip_on(state, suggested_title="word " * 60)
        title, truncated = build_title(clip, state)
        assert len(title) <= TITLE_LIMIT
        assert truncated is True
        assert title.endswith("#Shorts")

    def test_does_not_double_the_tag(self) -> None:
        state = job()
        clip = clip_on(state, suggested_title="a great clip #shorts")
        title, _ = build_title(clip, state)
        assert title.lower().count("#shorts") == 1

    def test_falls_back_through_hook_to_the_source_title(self) -> None:
        state = job()
        clip = clip_on(state, suggested_title=None, hook_rewrite=None)
        title, _ = build_title(clip, state)
        assert "compilers" in title


class TestCreditBlock:
    def test_names_the_creator_as_the_source_not_the_author(self) -> None:
        state = job()
        block = credit_block(state)
        assert "Clip from:" in block
        assert "Creator: Example Creator" in block

    def test_links_the_full_episode(self) -> None:
        assert "https://www.youtube.com/watch?v=vid123" in credit_block(job())

    def test_cc_by_adds_the_licence_line(self) -> None:
        state = job(licence="Creative Commons Attribution license (reuse allowed)")
        assert "CC BY" in credit_block(state)

    def test_a_non_commercial_licence_gets_no_licence_line(
        self,
    ) -> None:
        # MIT OCW and Khan Academy are CC BY-NC-SA; claiming CC BY for them
        # would be asserting a right that does not exist.
        state = job(licence="Creative Commons Attribution-NonCommercial-ShareAlike")
        assert "CC BY" not in credit_block(state)

    def test_a_required_handle_is_reproduced_verbatim(self, watchlist) -> None:
        state = job(channel_id="UC_AMBER", status=RightsStatus.amber)
        entry = watchlist.entry("UC_AMBER")
        assert "@thecreator" in credit_block(state, entry)


class TestDescription:
    def test_disclaimer_is_always_last(self) -> None:
        state = job()
        description = build_description(clip_on(state), state)
        assert description.rstrip().endswith(DISCLAIMER)

    def test_disclaimer_survives_truncation(self) -> None:
        state = job()
        clip = clip_on(state)
        description = build_description(clip, state, extra="x " * 4000)
        assert len(description) <= DESCRIPTION_LIMIT
        assert description.rstrip().endswith(DISCLAIMER)


class TestBuild:
    def test_green_source_builds(self, watchlist) -> None:
        state = job()
        meta = build(state, clip_on(state), watchlist=watchlist)
        assert meta.rights_status == "green"
        assert meta.privacy_status == "private", "uploads start private"
        assert "shorts" in meta.tags
        assert "startups" in meta.tags, "watchlist topics become tags"
        assert "insight" in meta.tags

    def test_red_source_is_refused(self, watchlist) -> None:
        state = job(status=RightsStatus.red)
        with pytest.raises(RightsBlocked):
            build(state, clip_on(state), watchlist=watchlist)

    def test_unknown_source_is_refused(self, watchlist) -> None:
        state = job(status=RightsStatus.unknown)
        with pytest.raises(RightsBlocked):
            build(state, clip_on(state), watchlist=watchlist)

    def test_an_override_is_recorded_on_the_state(self, watchlist) -> None:
        state = job(status=RightsStatus.red)
        build(
            state,
            clip_on(state),
            watchlist=watchlist,
            allow_red=True,
            override_reason="own footage, mis-tagged",
        )
        assert state.rights_override is True
        assert state.rights_override_reason == "own footage, mis-tagged"

    def test_amber_flags_attribution_as_required(self, watchlist) -> None:
        state = job(channel_id="UC_AMBER", status=RightsStatus.amber)
        meta = build(state, clip_on(state), watchlist=watchlist)
        assert meta.attribution_required is True

    def test_config_overrides_the_defaults(self, watchlist) -> None:
        state = job()
        meta = build(
            state,
            clip_on(state),
            config={"privacy_status": "unlisted", "category_id": "27", "add_shorts_tag": False},
            watchlist=watchlist,
        )
        assert meta.privacy_status == "unlisted"
        assert meta.category_id == "27"
        assert "#Shorts" not in meta.title


class TestSaveAndRecord:
    def test_upload_json_is_shaped_for_videos_insert(self, tmp_path, watchlist) -> None:
        state = job()
        meta = build(state, clip_on(state), watchlist=watchlist)
        path = save(meta, tmp_path / "clip_01")

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["snippet"]["title"] == meta.title
        assert payload["status"]["privacyStatus"] == "private"
        assert payload["snippet"]["categoryId"] == "22"

    def test_recording_a_published_id_is_the_calibration_join_key(self) -> None:
        state = job()
        clip = clip_on(state)
        assert clip.youtube_video_id is None

        record_published(clip, "abc123XYZ")

        assert clip.youtube_video_id == "abc123XYZ"
        assert clip.published_at is not None

    def test_the_published_id_round_trips_through_state_json(self, tmp_path) -> None:
        from clipper.state import JobStore

        store = JobStore(tmp_path)
        state = job()
        record_published(clip_on(state), "abc123XYZ")
        store.save(state)

        assert store.load("vid123").clips[0].youtube_video_id == "abc123XYZ"
