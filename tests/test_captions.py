"""ASS caption generation.

The failure mode this guards against is not a crash - a malformed ASS file
renders as *nothing*, silently, in the final encode. So these tests assert on
the file's structure and on the two things that are easy to get wrong and
invisible until playback: the timestamp format, and karaoke durations that add
up to the line they belong to.
"""

from __future__ import annotations

import pytest

from clipper.edit.captions import (
    CAPTION_STYLE,
    HOOK_STYLE,
    build,
    escape,
    format_time,
    group_words,
    normalize_word,
    render_line,
    write,
)
from clipper.edit.render import _fontsdir_argument
from clipper.transcribe.schema import Word

STYLE = {
    "font": {"name": "Montserrat ExtraBold", "size": 78, "bold": True, "spacing": 0.4},
    "colours": {
        "primary": "&H00FFFFFF&",
        "highlight": "&H0000E5FF&",
        "emphasis": "&H0000D7FF&",
        "outline": "&H00000000&",
        "shadow": "&H80000000&",
    },
    "outline": {"width": 6, "shadow": 3, "border_style": 1},
    "layout": {
        "words_per_line": 4,
        "alignment": 2,
        "margin_v": 460,
        "margin_l": 80,
        "margin_r": 80,
    },
    "emphasis": {"scale_percent": 112, "enabled": True},
    "karaoke": {"enabled": True, "mode": "k", "min_word_ms": 90},
    "hook_overlay": {
        "enabled": True,
        "duration_s": 2.0,
        "size": 92,
        "margin_v": 1150,
        "colour": "&H00FFFFFF&",
    },
}


def words(*specs: tuple[str, float, float]) -> list[Word]:
    return [Word(w=w, start=s, end=e, speaker="S0") for w, s, e in specs]


@pytest.fixture
def sample() -> list[Word]:
    return words(
        ("Nobody", 10.0, 10.4),
        ("actually", 10.4, 10.9),
        ("reads", 10.9, 11.2),
        ("the", 11.2, 11.3),
        ("terms.", 11.3, 11.9),
    )


class TestFormatTime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0:00:00.00"),
            (1.5, "0:00:01.50"),
            (61.23, "0:01:01.23"),
            (3661.0, "1:01:01.00"),
            (-5.0, "0:00:00.00"),
        ],
    )
    def test_formats_as_ass_expects(self, seconds: float, expected: str) -> None:
        assert format_time(seconds) == expected

    def test_hour_is_not_zero_padded(self) -> None:
        """A two-digit hour parses, and the subtitles never appear."""
        assert format_time(3600.0).startswith("1:")


class TestEscape:
    def test_escapes_braces(self) -> None:
        # An unescaped brace opens an override block and swallows the word.
        assert escape("{oops}") == "\\{oops\\}"

    def test_escapes_backslashes(self) -> None:
        assert escape("a\\b") == "a\\\\b"

    def test_flattens_newlines(self) -> None:
        assert "\n" not in escape("one\ntwo")


class TestNormalizeWord:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("Terms.", "terms"), ("  Cost,  ", "cost"), ('"quoted"', "quoted")],
    )
    def test_strips_case_and_punctuation(self, text: str, expected: str) -> None:
        assert normalize_word(text) == expected


class TestGroupWords:
    def test_respects_the_word_count(self, sample: list[Word]) -> None:
        lines = group_words(sample, words_per_line=2)
        assert [len(line.words) for line in lines] == [2, 2, 1]

    def test_breaks_on_terminal_punctuation(self) -> None:
        grouped = group_words(
            words(("Done.", 0.0, 0.5), ("Next", 0.6, 0.9), ("thing", 0.9, 1.2)),
            words_per_line=4,
        )
        assert [line.text for line in grouped] == ["Done.", "Next thing"]

    def test_breaks_on_a_long_pause(self) -> None:
        grouped = group_words(
            words(("one", 0.0, 0.3), ("two", 5.0, 5.3)), words_per_line=4, max_gap_s=0.8
        )
        assert len(grouped) == 2

    def test_empty_input(self) -> None:
        assert group_words([]) == []


class TestRenderLine:
    def test_emits_a_dialogue_row(self, sample: list[Word]) -> None:
        [line] = group_words(sample[:3], words_per_line=4)
        row = render_line(line, STYLE, emphasis=set())
        assert row.startswith("Dialogue: 0,0:00:10.00,0:00:11.20,")
        assert f",{CAPTION_STYLE}," in row

    def test_karaoke_durations_sum_to_the_line_length(self, sample: list[Word]) -> None:
        """A line whose \\k values do not sum to its duration drifts."""
        import re

        [line] = group_words(sample, words_per_line=99)
        row = render_line(line, STYLE, emphasis=set())
        durations = [int(m) for m in re.findall(r"\\k(\d+)", row)]
        expected = int(round((line.end - line.start) * 100))
        assert sum(durations) == expected

    def test_every_word_gets_a_karaoke_tag(self, sample: list[Word]) -> None:
        [line] = group_words(sample, words_per_line=99)
        row = render_line(line, STYLE, emphasis=set())
        assert row.count("\\k") == len(sample)

    def test_karaoke_can_be_disabled(self, sample: list[Word]) -> None:
        [line] = group_words(sample, words_per_line=99)
        style = {**STYLE, "karaoke": {"enabled": False}}
        assert "\\k" not in render_line(line, style, emphasis=set())

    def test_emphasis_word_gets_colour_and_scale(self, sample: list[Word]) -> None:
        [line] = group_words(sample, words_per_line=99)
        row = render_line(line, STYLE, emphasis={"terms"})
        assert "\\fscx112" in row
        assert "H0000D7FF" in row

    def test_emphasis_override_is_closed(self, sample: list[Word]) -> None:
        """An unclosed override bleeds into the rest of the line."""
        [line] = group_words(sample, words_per_line=99)
        row = render_line(line, STYLE, emphasis={"actually"})
        assert f"{{\\r{CAPTION_STYLE}}}" in row

    def test_emphasis_matching_ignores_punctuation(self, sample: list[Word]) -> None:
        # The word in the transcript is "terms." with a full stop.
        [line] = group_words(sample, words_per_line=99)
        assert "\\fscx112" in render_line(line, STYLE, emphasis={"terms"})

    def test_emphasis_can_be_disabled(self, sample: list[Word]) -> None:
        [line] = group_words(sample, words_per_line=99)
        style = {**STYLE, "emphasis": {"enabled": False, "scale_percent": 112}}
        assert "\\fscx" not in render_line(line, style, emphasis={"terms"})


class TestBuild:
    def test_has_the_three_required_sections(self, sample: list[Word]) -> None:
        ass = build(sample, STYLE)
        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass

    def test_declares_the_export_resolution(self, sample: list[Word]) -> None:
        ass = build(sample, STYLE, width=1080, height=1920)
        assert "PlayResX: 1080" in ass
        assert "PlayResY: 1920" in ass

    def test_style_values_come_from_config(self, sample: list[Word]) -> None:
        ass = build(sample, STYLE)
        assert "Montserrat ExtraBold,78" in ass
        assert ",460,1" in ass  # margin_v from layout

    def test_secondary_colour_is_the_karaoke_source(self, sample: list[Word]) -> None:
        """Swapping primary and secondary inverts the karaoke reveal."""
        ass = build(sample, STYLE)
        style_row = next(r for r in ass.splitlines() if r.startswith(f"Style: {CAPTION_STYLE}"))
        fields = style_row.split(",")
        assert fields[3] == "&H00FFFFFF&"  # PrimaryColour
        assert fields[4] == "&H0000E5FF&"  # SecondaryColour = highlight

    def test_offset_shifts_captions_onto_the_clip_timeline(self, sample: list[Word]) -> None:
        """Without the offset every caption lands 40 minutes into a 40s clip."""
        ass = build(sample, STYLE, offset_s=10.0)
        assert "Dialogue: 0,0:00:00.00," in ass

    def test_offset_never_produces_a_negative_time(self, sample: list[Word]) -> None:
        ass = build(sample, STYLE, offset_s=999.0)
        assert "-" not in ass.split("[Events]")[1]

    def test_hook_overlay_is_rendered_on_its_own_layer(self, sample: list[Word]) -> None:
        ass = build(
            sample, STYLE, hook_text="Nobody reads them", offset_s=10.0, clip_duration_s=40.0
        )
        hook_row = next(r for r in ass.splitlines() if f",{HOOK_STYLE}," in r)
        assert hook_row.startswith("Dialogue: 1,0:00:00.00,0:00:02.00,")

    def test_hook_overlay_can_be_disabled(self, sample: list[Word]) -> None:
        style = {**STYLE, "hook_overlay": {"enabled": False}}
        ass = build(sample, style, hook_text="ignored", offset_s=10.0)
        assert "ignored" not in ass

    def test_hook_is_clamped_to_a_very_short_clip(self, sample: list[Word]) -> None:
        ass = build(sample, STYLE, hook_text="hi", offset_s=10.0, clip_duration_s=1.0)
        hook_row = next(r for r in ass.splitlines() if f",{HOOK_STYLE}," in r)
        assert "0:00:01.00" in hook_row

    def test_empty_word_list_still_produces_a_valid_file(self) -> None:
        ass = build([], STYLE)
        assert "[Events]" in ass
        assert "Dialogue:" not in ass

    def test_every_word_appears_in_the_output(self, sample: list[Word]) -> None:
        ass = build(sample, STYLE)
        for word in sample:
            assert word.w in ass


class TestWrite:
    def test_writes_utf8_without_a_bom(self, tmp_path, sample: list[Word]) -> None:
        """libass reads a BOM as part of the first style name."""
        path = write(sample, STYLE, tmp_path / "captions.ass")
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_creates_the_parent_directory(self, tmp_path, sample: list[Word]) -> None:
        path = write(sample, STYLE, tmp_path / "clips" / "clip_01" / "captions.ass")
        assert path.is_file()


class TestFontsDirArgument:
    """The caption font ships with the repo; libass has to be pointed at it.

    A font that libass cannot find is not an error - it substitutes one and the
    encode succeeds - so the argument being well-formed is the only thing
    standing between the configured style and a typeface nobody chose.
    """

    def test_is_relative_to_the_clip_directory(self, tmp_path) -> None:
        fonts = tmp_path / "assets" / "fonts"
        fonts.mkdir(parents=True)
        clip_dir = tmp_path / "jobs" / "vid" / "clips" / "clip_01"
        clip_dir.mkdir(parents=True)
        assert _fontsdir_argument(fonts, clip_dir) == "../../../../assets/fonts"

    def test_has_no_drive_letter_to_confuse_the_filter_parser(self, tmp_path) -> None:
        """A bare ``C:`` in a filter option value reads as an option separator."""
        fonts = tmp_path / "fonts"
        fonts.mkdir()
        argument = _fontsdir_argument(fonts, tmp_path / "clip")
        assert argument is not None
        assert ":" not in argument

    def test_missing_directory_is_dropped_rather_than_passed(self, tmp_path) -> None:
        assert _fontsdir_argument(tmp_path / "nope", tmp_path) is None
