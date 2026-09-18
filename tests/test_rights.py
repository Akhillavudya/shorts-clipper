"""The rights gate.

This is the one part of the pipeline where a bug costs the whole channel
rather than one bad clip, so the tests lean towards "prove it refuses" over
"prove it allows". The licence-trap cases are not hypothetical: MIT OCW, Khan
Academy and TED all publish under a Creative Commons licence that reads open
and forbids exactly what this project does.
"""

from __future__ import annotations

import textwrap

import pytest

from clipper.rights.registry import (
    ChannelEntry,
    RightsError,
    Watchlist,
    check_export,
    licence_is_commercially_clear,
    resolve,
)
from clipper.state import JobState, RightsBlocked, RightsStatus


def write_watchlist(tmp_path, body: str):
    path = tmp_path / "watchlist.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


@pytest.fixture
def watchlist(tmp_path) -> Watchlist:
    path = write_watchlist(
        tmp_path,
        """
        defaults:
          max_per_day: 2
          min_duration_s: 900
        channels:
          - id: UC_GREEN
            name: Own Channel
            rights_status: green
            note: Own footage.
          - id: UC_AMBER
            name: Permissioned Creator
            rights_status: amber
            note: Said yes by email.
            permission_file: rights/permissions/UC_AMBER.md
          - id: UC_AMBER_NOFILE
            name: Sloppy Amber
            rights_status: amber
            note: Said yes, allegedly.
          - id: UC_RED
            name: Big Podcast
            rights_status: red
            note: Runs its own clips channel.
            max_per_day: 5
        """,
    )
    return Watchlist.load(path)


class TestLicence:
    @pytest.mark.parametrize(
        "licence",
        [
            "Creative Commons Attribution license (reuse allowed)",
            "cc-by",
            "CC BY",
        ],
    )
    def test_cc_by_is_clear(self, licence: str) -> None:
        assert licence_is_commercially_clear(licence) is True

    @pytest.mark.parametrize(
        "licence",
        [
            "CC BY-NC-SA",  # MIT OpenCourseWare, Khan Academy
            "CC BY-NC-ND",  # TED
            "Creative Commons BY-NC",
            "creative commons noncommercial",
            "CC BY-ND",
            "Standard YouTube License",
            "",
            None,
        ],
    )
    def test_everything_else_is_not(self, licence: str | None) -> None:
        assert licence_is_commercially_clear(licence) is False

    def test_nc_is_checked_before_by(self) -> None:
        """A BY-NC string contains "attribution"; it must not match CC BY."""
        assert not licence_is_commercially_clear(
            "Creative Commons Attribution-NonCommercial license"
        )


class TestWatchlistLoading:
    def test_indexes_by_channel_id(self, watchlist: Watchlist) -> None:
        assert len(watchlist) == 4
        assert "UC_RED" in watchlist
        assert watchlist.entry("UC_GREEN").rights_status is RightsStatus.green

    def test_unknown_channel_returns_none(self, watchlist: Watchlist) -> None:
        assert watchlist.entry("UC_NOPE") is None
        assert watchlist.entry(None) is None

    def test_per_channel_override_beats_the_default(self, watchlist: Watchlist) -> None:
        assert watchlist.max_per_day("UC_RED") == 5
        assert watchlist.max_per_day("UC_GREEN") == 2

    def test_defaults_apply_to_unlisted_channels(self, watchlist: Watchlist) -> None:
        assert watchlist.min_duration_s("UC_NOPE") == 900

    def test_missing_rights_status_is_an_error(self, tmp_path) -> None:
        path = write_watchlist(
            tmp_path,
            """
            channels:
              - id: UC_X
                name: Unclassified
            """,
        )
        with pytest.raises(RightsError, match="rights_status"):
            Watchlist.load(path)

    def test_bad_rights_status_is_an_error(self, tmp_path) -> None:
        path = write_watchlist(
            tmp_path,
            """
            channels:
              - id: UC_X
                rights_status: probably fine
            """,
        )
        with pytest.raises(RightsError, match="expected one of"):
            Watchlist.load(path)

    def test_duplicate_ids_are_an_error(self, tmp_path) -> None:
        """Two entries for one channel means file order decides the status."""
        path = write_watchlist(
            tmp_path,
            """
            channels:
              - id: UC_X
                rights_status: red
              - id: UC_X
                rights_status: green
            """,
        )
        with pytest.raises(RightsError, match="duplicate"):
            Watchlist.load(path)

    def test_missing_id_is_an_error(self, tmp_path) -> None:
        path = write_watchlist(
            tmp_path,
            """
            channels:
              - name: Nameless
                rights_status: green
            """,
        )
        with pytest.raises(RightsError, match="no 'id'"):
            Watchlist.load(path)

    def test_empty_channel_list_is_fine(self, tmp_path) -> None:
        assert len(Watchlist.load(write_watchlist(tmp_path, "channels: []"))) == 0

    def test_the_shipped_watchlist_parses(self) -> None:
        """The file the gate actually reads must always load."""
        assert isinstance(Watchlist.load(), Watchlist)


class TestResolve:
    def test_watchlist_entry_wins(self, watchlist: Watchlist) -> None:
        decision = resolve(channel_id="UC_GREEN", watchlist=watchlist)
        assert decision.status is RightsStatus.green
        assert decision.entry is not None

    def test_red_channel_stays_red_despite_a_cc_by_upload(self, watchlist: Watchlist) -> None:
        """A human decision about a channel outranks a per-video licence."""
        decision = resolve(
            channel_id="UC_RED",
            licence="Creative Commons Attribution license (reuse allowed)",
            watchlist=watchlist,
        )
        assert decision.status is RightsStatus.red

    def test_amber_without_a_permission_file_is_flagged(self, watchlist: Watchlist) -> None:
        decision = resolve(channel_id="UC_AMBER_NOFILE", watchlist=watchlist)
        assert decision.status is RightsStatus.amber
        assert "WARNING" in decision.note

    def test_cc_by_on_an_unlisted_channel_is_green(self, watchlist: Watchlist) -> None:
        decision = resolve(
            channel_id="UC_NEW",
            licence="Creative Commons Attribution license (reuse allowed)",
            watchlist=watchlist,
        )
        assert decision.status is RightsStatus.green
        assert "verify per video" in decision.note

    def test_unlisted_channel_is_unknown(self, watchlist: Watchlist) -> None:
        decision = resolve(channel_id="UC_NEW", watchlist=watchlist)
        assert decision.status is RightsStatus.unknown
        assert "not in config/watchlist.yaml" in decision.note

    def test_no_channel_id_is_unknown(self, watchlist: Watchlist) -> None:
        assert resolve(channel_id=None, watchlist=watchlist).status is RightsStatus.unknown

    def test_no_watchlist_at_all_is_unknown(self) -> None:
        assert resolve(channel_id="UC_X").status is RightsStatus.unknown

    def test_explicit_override_wins(self, watchlist: Watchlist) -> None:
        decision = resolve(channel_id="UC_RED", watchlist=watchlist, override="green")
        assert decision.status is RightsStatus.green
        assert "--rights" in decision.note

    def test_bad_override_is_an_error(self, watchlist: Watchlist) -> None:
        with pytest.raises(RightsError, match="not one of"):
            resolve(channel_id="UC_RED", watchlist=watchlist, override="probably")


class TestCheckExport:
    def state(self, status: RightsStatus) -> JobState:
        return JobState(video_id="vid", rights_status=status, rights_note="n")

    @pytest.mark.parametrize("status", [RightsStatus.green, RightsStatus.amber])
    def test_allows_cleared_sources(self, status: RightsStatus) -> None:
        check_export(self.state(status))

    @pytest.mark.parametrize("status", [RightsStatus.red, RightsStatus.unknown])
    def test_blocks_uncleared_sources(self, status: RightsStatus) -> None:
        with pytest.raises(RightsBlocked):
            check_export(self.state(status))

    def test_unknown_is_blocked_like_red(self) -> None:
        """Unclassified is not cleared. This is the default for a new source."""
        with pytest.raises(RightsBlocked):
            check_export(self.state(RightsStatus.unknown))

    def test_the_message_explains_the_channel_level_cost(self) -> None:
        with pytest.raises(RightsBlocked, match="CHANNEL level"):
            check_export(self.state(RightsStatus.red))

    def test_override_allows_and_is_recorded(self) -> None:
        state = self.state(RightsStatus.red)
        check_export(state, allow_red=True, reason="written permission in email")
        assert state.rights_override is True
        assert state.rights_override_reason == "written permission in email"

    def test_override_without_a_reason_records_a_default(self) -> None:
        state = self.state(RightsStatus.unknown)
        check_export(state, allow_red=True)
        assert "allow-red" in state.rights_override_reason

    def test_override_does_not_mark_an_already_clear_source(self) -> None:
        state = self.state(RightsStatus.green)
        check_export(state, allow_red=True)
        assert state.rights_override is False


class TestChannelEntry:
    def test_exportability_matches_the_tier(self) -> None:
        for status, expected in (
            (RightsStatus.green, True),
            (RightsStatus.amber, True),
            (RightsStatus.red, False),
            (RightsStatus.unknown, False),
        ):
            entry = ChannelEntry(id="x", name=None, rights_status=status)
            assert entry.is_exportable is expected
