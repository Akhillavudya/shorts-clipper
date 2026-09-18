"""ffmpeg output parsing and crop geometry.

Neither needs ffmpeg installed, which is the point - CLAUDE.md says the suite
runs without it. What is tested is the parsing of ffmpeg's *output* and the
arithmetic that produces its *input*; the subprocess call between them is not
this project's code.
"""

from __future__ import annotations

import pytest

from clipper import media
from clipper.edit.audio_mix import build_filter, parse_loudnorm_stats
from clipper.edit.reframe import plan_crop

SILENCEDETECT_STDERR = """\
[silencedetect @ 0x55f] silence_start: 1.234
[silencedetect @ 0x55f] silence_end: 1.987 | silence_duration: 0.753
[silencedetect @ 0x55f] silence_start: 10.5
[silencedetect @ 0x55f] silence_end: 11.25 | silence_duration: 0.75
size=N/A time=00:00:30.00 bitrate=N/A speed= 120x
"""

LOUDNORM_STDERR = """\
[Parsed_loudnorm_0 @ 0x55f]
{
    "input_i" : "-23.45",
    "input_tp" : "-4.12",
    "input_lra" : "7.30",
    "input_thresh" : "-33.90",
    "output_i" : "-14.02",
    "target_offset" : "0.31"
}
"""


class TestParseSilences:
    def test_pairs_start_and_end(self) -> None:
        assert media.parse_silences(SILENCEDETECT_STDERR) == [
            (1.234, 1.987),
            (10.5, 11.25),
        ]

    def test_closes_a_trailing_silence_at_the_file_duration(self) -> None:
        """The gap at the end is often the most useful boundary in the file."""
        stderr = SILENCEDETECT_STDERR + "[silencedetect @ 0x55f] silence_start: 28.0\n"
        assert media.parse_silences(stderr, total_s=30.0)[-1] == (28.0, 30.0)

    def test_drops_a_trailing_silence_with_no_known_duration(self) -> None:
        stderr = SILENCEDETECT_STDERR + "[silencedetect @ 0x55f] silence_start: 28.0\n"
        assert len(media.parse_silences(stderr)) == 2

    def test_ignores_a_zero_length_span(self) -> None:
        stderr = "silence_start: 5.0\nsilence_end: 5.0\n"
        assert media.parse_silences(stderr) == []

    def test_output_is_sorted(self) -> None:
        stderr = "silence_start: 20.0\nsilence_end: 21.0\nsilence_start: 1.0\nsilence_end: 2.0\n"
        assert media.parse_silences(stderr) == [(1.0, 2.0), (20.0, 21.0)]

    def test_no_silences(self) -> None:
        assert media.parse_silences("nothing here") == []

    def test_negative_start_is_clamped(self) -> None:
        stderr = "silence_start: -0.5\nsilence_end: 1.0\n"
        assert media.parse_silences(stderr) == [(0.0, 1.0)]


class TestParseFps:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("30/1", 30.0),
            ("30000/1001", pytest.approx(29.97, abs=0.01)),
            ("25", 25.0),
            ("0/0", 0.0),
            ("N/A", 0.0),
            (None, 0.0),
        ],
    )
    def test_parses_ffprobe_rate_strings(self, raw: str | None, expected) -> None:
        assert media._parse_fps(raw) == expected


class TestRequire:
    def test_missing_binary_explains_how_to_install_it(self) -> None:
        with pytest.raises(media.FFmpegMissing, match="winget install"):
            media.require("definitely-not-a-real-binary")


class TestPlanCrop:
    def test_16_9_source_crops_the_width(self) -> None:
        plan = plan_crop(1920, 1080)
        assert plan.crop_height == 1080
        assert plan.crop_width == 608  # 1080 * 9/16 = 607.5, rounded to even
        assert plan.pad is False

    def test_crop_is_horizontally_centred(self) -> None:
        plan = plan_crop(1920, 1080)
        assert plan.crop_x == (1920 - plan.crop_width) // 2 - ((1920 - plan.crop_width) // 2) % 2

    def test_dimensions_are_even(self) -> None:
        """Odd dimensions are rejected outright by yuv420p."""
        for width, height in ((1920, 1080), (1919, 1079), (1280, 721), (3840, 2160)):
            plan = plan_crop(width, height)
            assert plan.crop_width % 2 == 0
            assert plan.crop_height % 2 == 0
            assert plan.crop_x % 2 == 0
            assert plan.crop_y % 2 == 0

    def test_square_source_is_still_wider_than_9_16(self) -> None:
        plan = plan_crop(1080, 1080)
        assert plan.pad is False
        assert plan.crop_width == 608

    def test_source_narrower_than_9_16_is_padded_not_cropped(self) -> None:
        """Cropping a 9:21 source to 9:16 would cut the top off a face."""
        plan = plan_crop(1080, 2400)
        assert plan.pad is True
        assert "pad=" in plan.filter_chain

    def test_pad_preserves_the_whole_frame(self) -> None:
        plan = plan_crop(1080, 2400)
        assert (plan.crop_width, plan.crop_height) == (1080, 2400)

    def test_crop_never_exceeds_the_source(self) -> None:
        plan = plan_crop(640, 480)
        assert plan.crop_width <= 640
        assert plan.crop_height <= 480

    def test_detects_an_upscale(self) -> None:
        """A 1080p source is the practical floor; below it the output is soft."""
        assert plan_crop(1280, 720).is_upscaling is True
        assert plan_crop(3840, 2160).is_upscaling is False

    def test_1080p_source_is_still_an_upscale(self) -> None:
        # 1080 * 9/16 = 607 wide, scaled up to 1080. This is why the format
        # selector prefers better than 1080p.
        assert plan_crop(1920, 1080).is_upscaling is True

    def test_centre_x_shifts_the_crop(self) -> None:
        left = plan_crop(1920, 1080, centre_x=0.2)
        right = plan_crop(1920, 1080, centre_x=0.8)
        assert left.crop_x < right.crop_x

    def test_centre_x_is_clamped_inside_the_frame(self) -> None:
        for centre in (-1.0, 0.0, 1.0, 2.0):
            plan = plan_crop(1920, 1080, centre_x=centre)
            assert 0 <= plan.crop_x <= 1920 - plan.crop_width

    def test_filter_chain_ends_with_the_export_size(self) -> None:
        chain = plan_crop(1920, 1080).filter_chain
        assert chain.startswith("crop=608:1080:")
        assert "scale=1080:1920" in chain
        assert chain.endswith("setsar=1")

    def test_rejects_a_degenerate_source(self) -> None:
        with pytest.raises(ValueError, match="bad source dimensions"):
            plan_crop(0, 1080)


class TestLoudnorm:
    def test_parses_the_measurement_block(self) -> None:
        stats = parse_loudnorm_stats(LOUDNORM_STDERR)
        assert stats is not None
        assert stats.input_i == pytest.approx(-23.45)
        assert stats.target_offset == pytest.approx(0.31)

    def test_returns_none_when_there_is_no_block(self) -> None:
        assert parse_loudnorm_stats("just some ffmpeg noise") is None

    def test_first_pass_filter_asks_for_json(self) -> None:
        assert "print_format=json" in build_filter(None)

    def test_second_pass_filter_feeds_the_measurements_back(self) -> None:
        stats = parse_loudnorm_stats(LOUDNORM_STDERR)
        chain = build_filter(stats)
        assert "measured_I=-23.45" in chain
        assert "measured_TP=-4.12" in chain
        assert "linear=true" in chain

    def test_highpass_runs_before_loudnorm(self) -> None:
        """Measuring with rumble still in the signal biases the correction."""
        chain = build_filter(None)
        assert chain.index("highpass") < chain.index("loudnorm")

    def test_denoise_is_off_by_default(self) -> None:
        assert "afftdn" not in build_filter(None)
        assert "afftdn" in build_filter(None, denoise=True)

    def test_resamples_after_loudnorm(self) -> None:
        # loudnorm outputs 192kHz internally; AAC will not take that quietly.
        assert build_filter(None).endswith("aresample=48000")
