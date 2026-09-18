"""Speaker tracking: picking a face, smoothing the path, keying the crop.

No MediaPipe, no OpenCV, no video. Everything here is the arithmetic between
a list of detections and a sendcmd script, which is where all the behaviour
that can be wrong lives - the detector either finds a face or it does not, and
that is not something an assertion can hold to account.

The tests that matter most are the three smoothing rules. Each one is a thing
that looks fine in a still frame and is unwatchable in motion: no deadzone and
the frame drifts constantly, no scene reset and every cut becomes a whip pan,
no interpolation and the crop moves in 250ms lurches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipper.edit import reframe, track
from clipper.edit.track import CropPath, Face, FaceSample, TrackingOptions

WIDTH = 2560
CROP_W = 810  # what plan_crop gives for a 2560x1440 source


def face(centre_x: float, size: float = 250.0, score: float = 0.9) -> Face:
    """A square face of ``size`` centred on ``centre_x``, vertically mid-frame."""
    return Face(x=centre_x - size / 2, y=600.0, width=size, height=size, score=score)


def samples(*centres: float | None, interval_s: float = 0.25) -> list[FaceSample]:
    """One sample per argument; ``None`` means nothing was detected."""
    return [
        FaceSample(t=round(i * interval_s, 3), faces=() if c is None else (face(c),))
        for i, c in enumerate(centres)
    ]


class TestPickFace:
    def test_the_biggest_face_wins(self) -> None:
        small = face(400, size=120)
        big = face(1800, size=300)
        assert track.pick_face([small, big], frame_width=WIDTH) is big

    def test_centre_breaks_a_tie_between_equal_faces(self) -> None:
        """A wide two-shot puts the interviewee nearer the middle."""
        edge = face(200, size=250)
        middle = face(1200, size=250)
        assert track.pick_face([edge, middle], frame_width=WIDTH) is middle

    def test_the_previous_pick_is_sticky(self) -> None:
        """Two faces of near-equal size otherwise trade the frame every sample."""
        held = face(1800, size=250)
        rival = face(700, size=262)  # ~5% bigger; enough to win without stickiness
        assert track.pick_face([held, rival], frame_width=WIDTH) is rival
        assert track.pick_face([held, rival], frame_width=WIDTH, previous_centre_x=1800.0) is held

    def test_stickiness_does_not_hold_against_a_much_bigger_face(self) -> None:
        held = face(1800, size=250)
        closer = face(700, size=600)
        assert (
            track.pick_face([held, closer], frame_width=WIDTH, previous_centre_x=1800.0) is closer
        )

    def test_no_faces_is_none_not_a_guess(self) -> None:
        assert track.pick_face([], frame_width=WIDTH) is None


class TestSmoothCentres:
    def test_a_target_inside_the_deadzone_moves_nothing(self) -> None:
        """Normal head sway must not produce a permanently drifting frame."""
        path = track.smooth_centres(
            samples(1280, 1300, 1260, 1295), frame_width=WIDTH, deadzone_px=40
        )
        assert [c for _, c in path] == [1280.0] * 4

    def test_outside_the_deadzone_the_crop_eases_toward_the_target(self) -> None:
        path = track.smooth_centres(samples(1280, 1780), frame_width=WIDTH, alpha=0.2)
        # 1280 + 0.2 * (1780 - 1280)
        assert path[1][1] == pytest.approx(1380.0)

    def test_a_lost_face_holds_position_rather_than_recentring(self) -> None:
        """A turn of the head is not a reason to pan back to the middle."""
        path = track.smooth_centres(
            samples(1900, None, None, 1900), frame_width=WIDTH, alpha=0.5, deadzone_px=10
        )
        assert path[1][1] == path[2][1] == path[0][1]

    def test_a_scene_cut_snaps_instead_of_panning(self) -> None:
        path = track.smooth_centres(
            samples(700, 700, 2000, 2000),
            frame_width=WIDTH,
            scene_cuts=[0.5],
            alpha=0.15,
            deadzone_px=40,
            start_centre_x=700,
        )
        assert path[2][1] == 2000.0  # adopted outright, not eased toward

    def test_without_the_cut_the_same_jump_is_eased(self) -> None:
        path = track.smooth_centres(
            samples(700, 700, 2000, 2000),
            frame_width=WIDTH,
            alpha=0.15,
            deadzone_px=40,
            start_centre_x=700,
        )
        assert path[2][1] == pytest.approx(700 + 0.15 * 1300)

    def test_a_cut_into_an_empty_shot_recentres(self) -> None:
        """Nobody in the new shot; the middle is the only defensible guess."""
        path = track.smooth_centres(
            samples(2100, None, None), frame_width=WIDTH, scene_cuts=[0.2], deadzone_px=10
        )
        assert path[1][1] == WIDTH / 2

    def test_a_cut_at_the_start_of_the_file_is_not_a_cut(self) -> None:
        """PySceneDetect reports the first scene's start; it is not a boundary."""
        with_cut = track.smooth_centres(
            samples(700, 2000), frame_width=WIDTH, scene_cuts=[0.0], alpha=0.15
        )
        without = track.smooth_centres(samples(700, 2000), frame_width=WIDTH, alpha=0.15)
        assert with_cut == without

    def test_the_path_starts_centred_when_the_first_frame_has_no_face(self) -> None:
        path = track.smooth_centres(samples(None, None), frame_width=WIDTH)
        assert path[0][1] == WIDTH / 2

    def test_the_first_face_is_adopted_not_eased_toward(self) -> None:
        """Otherwise every Short opens with a two-second drift onto the speaker."""
        path = track.smooth_centres(samples(1900, 1900), frame_width=WIDTH, alpha=0.15)
        assert path[0][1] == 1900.0

    def test_the_first_face_is_adopted_even_when_it_appears_late(self) -> None:
        """Nothing was being framed before it, so there is nothing to ease from."""
        path = track.smooth_centres(samples(None, None, 1900), frame_width=WIDTH, alpha=0.15)
        assert path[2][1] == 1900.0

    def test_a_seeded_path_eases_from_the_seed(self) -> None:
        path = track.smooth_centres(
            samples(1900), frame_width=WIDTH, alpha=0.5, start_centre_x=1000
        )
        assert path[0][1] == pytest.approx(1450.0)

    def test_no_samples_is_an_empty_path(self) -> None:
        assert track.smooth_centres([], frame_width=WIDTH) == []


class TestCropX:
    def test_the_offset_is_always_even(self) -> None:
        """yuv420p rejects an odd crop outright, mid-encode, as a bad command."""
        for centre in (1000.3, 1001.0, 1234.9, 999.5):
            x = track.crop_x_for(centre, crop_width=CROP_W, src_width=WIDTH)
            assert x % 2 == 0

    def test_a_face_at_the_edge_clamps_inside_the_frame(self) -> None:
        assert track.crop_x_for(10, crop_width=CROP_W, src_width=WIDTH) == 0
        assert track.crop_x_for(WIDTH - 10, crop_width=CROP_W, src_width=WIDTH) == WIDTH - CROP_W

    def test_a_centred_face_gives_a_centred_crop(self) -> None:
        x = track.crop_x_for(WIDTH / 2, crop_width=CROP_W, src_width=WIDTH)
        assert x == pytest.approx((WIDTH - CROP_W) / 2, abs=2)


class TestBuildKeys:
    def test_samples_are_interpolated_onto_the_frame_grid(self) -> None:
        """One command per detection would move the crop in 250ms lurches."""
        keys = track.build_keys(
            [(0.0, 1280.0), (0.25, 1480.0)],
            crop_width=CROP_W,
            src_width=WIDTH,
            fps=30,
            duration_s=0.25,
        )
        times = [t for t, _ in keys]
        assert len(keys) > 2
        assert max(times) <= 0.25 + 1 / 30

    def test_a_cut_steps_instead_of_ramping(self) -> None:
        """Interpolating through a cut smears it into a 250ms whip pan."""
        keys = track.build_keys(
            [(0.0, 700.0), (0.25, 2000.0)],
            crop_width=CROP_W,
            src_width=WIDTH,
            fps=30,
            duration_s=0.25,
            scene_cuts=[0.1],
        )
        assert len(keys) == 2
        assert keys[0] == (0.0, track.crop_x_for(700.0, crop_width=CROP_W, src_width=WIDTH))

    def test_the_step_lands_on_the_cut_not_the_sample_grid(self) -> None:
        """Detections are 250ms apart; the shot changes on a specific frame."""
        keys = track.build_keys(
            [(0.0, 700.0), (0.25, 2000.0)],
            crop_width=CROP_W,
            src_width=WIDTH,
            fps=30,
            duration_s=0.25,
            scene_cuts=[0.2],
        )
        assert keys[1][0] == pytest.approx(0.2, abs=1 / 30)

    def test_without_a_cut_the_same_move_is_a_ramp(self) -> None:
        keys = track.build_keys(
            [(0.0, 700.0), (0.25, 2000.0)],
            crop_width=CROP_W,
            src_width=WIDTH,
            fps=30,
            duration_s=0.25,
        )
        assert len(keys) > 2

    def test_a_static_shot_costs_one_line(self) -> None:
        keys = track.build_keys(
            [(0.0, 1280.0), (5.0, 1280.0)],
            crop_width=CROP_W,
            src_width=WIDTH,
            fps=30,
            duration_s=5.0,
        )
        assert len(keys) == 1
        assert keys[0][0] == 0.0

    def test_the_first_key_is_at_time_zero(self) -> None:
        """The crop has to be right on frame one, not from the first movement."""
        keys = track.build_keys(
            [(0.0, 400.0), (1.0, 2000.0)],
            crop_width=CROP_W,
            src_width=WIDTH,
            fps=30,
            duration_s=1.0,
        )
        assert keys[0][0] == 0.0

    def test_every_offset_is_legal(self) -> None:
        keys = track.build_keys(
            [(0.0, 0.0), (1.0, float(WIDTH))],
            crop_width=CROP_W,
            src_width=WIDTH,
            fps=30,
            duration_s=1.0,
        )
        assert all(0 <= x <= WIDTH - CROP_W and x % 2 == 0 for _, x in keys)

    def test_an_empty_path_produces_no_keys(self) -> None:
        assert track.build_keys([], crop_width=CROP_W, src_width=WIDTH, fps=30) == ()


class TestSendcmd:
    def test_the_script_addresses_the_named_crop_filter(self) -> None:
        script = track.to_sendcmd([(0.0, 875), (1.5, 900)])
        assert script.splitlines() == [
            "0.000 crop@dyn x 875;",
            "1.500 crop@dyn x 900;",
        ]

    def test_arguments_are_integers_never_expressions(self) -> None:
        """crop's `eval` option is gone in ffmpeg 8 and was `init` in 7, so an
        expression in `t` evaluates once on one build and per frame on another."""
        script = track.to_sendcmd([(0.0, 875), (0.033, 878)])
        for line in script.splitlines():
            assert line.split()[-1].rstrip(";").isdigit()

    def test_an_empty_path_writes_an_empty_script(self) -> None:
        assert track.to_sendcmd([]) == ""


class TestCropPath:
    def test_coverage_is_the_fraction_of_samples_with_a_face(self) -> None:
        assert CropPath(keys=(), samples=100, detections=42).coverage == pytest.approx(0.42)

    def test_coverage_of_nothing_is_zero_not_an_error(self) -> None:
        assert CropPath(keys=()).coverage == 0.0

    def test_travel_is_zero_for_a_static_crop(self) -> None:
        assert CropPath(keys=((0.0, 875),)).travel_px == 0

    def test_travel_sums_absolute_movement(self) -> None:
        assert CropPath(keys=((0.0, 100), (1.0, 300), (2.0, 200))).travel_px == 300


class TestConfig:
    def test_only_the_word_centre_turns_tracking_off(self) -> None:
        assert track.tracking_enabled({}) is True
        assert track.tracking_enabled({"reframe": "track"}) is True
        assert track.tracking_enabled({"reframe": "Centre"}) is False
        # A typo must not silently downgrade every render.
        assert track.tracking_enabled({"reframe": "centr"}) is True

    def test_the_cli_flag_beats_the_config(self) -> None:
        assert track.plan_from_config({"reframe": "track"}, disabled=True) is None

    def test_settings_are_read_from_the_edit_block(self) -> None:
        options = track.options_from_config(
            {
                "face_model": "models/other.tflite",
                "detect_interval_s": 0.5,
                "smoothing_alpha": 0.3,
                "deadzone_px": 10,
                "min_face_coverage": 0.6,
            }
        )
        assert options.model_path == Path("models/other.tflite")
        assert options.interval_s == 0.5
        assert options.alpha == 0.3
        assert options.deadzone_px == 10
        assert options.min_coverage == 0.6

    def test_missing_keys_fall_back_to_the_documented_defaults(self) -> None:
        options = track.options_from_config({})
        assert options.interval_s == track.DEFAULT_INTERVAL_S
        assert options.alpha == track.DEFAULT_ALPHA
        assert options.deadzone_px == track.DEFAULT_DEADZONE_PX


class TestFilterChain:
    def test_the_static_chain_has_no_sendcmd(self) -> None:
        plan = reframe.plan_crop(2560, 1440)
        assert "sendcmd" not in plan.filter_chain
        assert plan.filter_chain.startswith("crop=810:1440:")

    def test_the_tracked_chain_sends_commands_to_a_named_crop(self) -> None:
        plan = reframe.plan_crop(2560, 1440)
        chain = plan.build_filter_chain("crop_path.cmd")
        assert chain.startswith("sendcmd=f=crop_path.cmd,crop@dyn=")
        # sendcmd must precede its target, or the commands arrive nowhere.
        assert chain.index("sendcmd") < chain.index(track.FILTER_NAME)

    def test_the_tracked_chain_still_scales_and_squares_the_pixels(self) -> None:
        plan = reframe.plan_crop(2560, 1440)
        chain = plan.build_filter_chain("crop_path.cmd")
        assert "scale=1080:1920:flags=lanczos" in chain
        assert chain.endswith("setsar=1")


class TestBuildPathFallback:
    """``reframe.build_path`` decides whether a tracked path is trustworthy."""

    def _plan(self) -> reframe.CropPlan:
        return reframe.plan_crop(2560, 1440)

    def test_a_missing_dependency_falls_back_with_its_reason(self, monkeypatch) -> None:
        def explode(*args, **kwargs):
            raise track.TrackingUnavailable("mediapipe is not importable")

        monkeypatch.setattr(track, "plan_path", explode)
        path, note = reframe.build_path(
            Path("clip.mp4"), self._plan(), options=TrackingOptions(), fps=30, duration_s=40
        )
        assert path is None
        assert "mediapipe" in note

    def test_too_few_faces_is_noise_not_a_camera(self, monkeypatch) -> None:
        monkeypatch.setattr(
            track,
            "plan_path",
            lambda *a, **k: CropPath(keys=((0.0, 875),), samples=160, detections=8),
        )
        path, note = reframe.build_path(
            Path("clip.mp4"),
            self._plan(),
            options=TrackingOptions(min_coverage=0.35),
            fps=30,
            duration_s=40,
        )
        assert path is None
        assert "5%" in note

    def test_a_well_covered_path_is_used_without_comment(self, monkeypatch) -> None:
        good = CropPath(keys=((0.0, 875), (1.0, 900)), samples=160, detections=150)
        monkeypatch.setattr(track, "plan_path", lambda *a, **k: good)
        path, note = reframe.build_path(
            Path("clip.mp4"), self._plan(), options=TrackingOptions(), fps=30, duration_s=40
        )
        assert path is good
        assert note == ""
