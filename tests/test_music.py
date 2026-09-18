"""The music bed: filter construction, track resolution, config reading.

No ffmpeg here either. What is tested is the filter string handed to ffmpeg
and the decisions taken before the subprocess exists - which track, how quiet,
and whether a bed is laid at all. The mix itself is one ffmpeg invocation and
is verified by ear, not by assertion.

The two assertions worth keeping honest are ``normalize=0`` on ``amix`` and
the direction of the sidechain. Both are silent failures: the first pulls the
dialogue 6 dB under the loudnorm target the pass before it just hit, and the
second ducks the voice out of the way of the music.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipper.edit import audio_mix
from clipper.edit.audio_mix import MusicMissing, MusicPlan


@pytest.fixture
def music_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "music"
    directory.mkdir()
    for name in ("b-second.mp3", "a-first.wav", "notes.txt"):
        (directory / name).write_bytes(b"")
    return directory


class TestBuildMusicFilter:
    def test_mix_does_not_renormalize(self) -> None:
        """amix's default divides by the input count; the speech would drop 6 dB."""
        assert "normalize=0" in audio_mix.build_music_filter(40.0)

    def test_speech_keys_the_compressor_and_music_is_compressed(self) -> None:
        """[bed][key]sidechaincompress - main input first, key second."""
        chain = audio_mix.build_music_filter(40.0)
        assert "[bed][key]sidechaincompress=" in chain
        assert "[key][bed]sidechaincompress=" not in chain

    def test_bed_is_normalized_to_the_requested_loudness(self) -> None:
        """A gain would mean something different for every track dropped in."""
        assert "loudnorm=I=-26:" in audio_mix.build_music_filter(40.0)
        assert "loudnorm=I=-18:" in audio_mix.build_music_filter(40.0, bed_lufs=-18.0)

    def test_fade_out_starts_before_the_end(self) -> None:
        chain = audio_mix.build_music_filter(40.0, fade_out_s=1.5)
        assert "afade=t=out:st=38.500:d=1.5" in chain

    def test_fade_out_never_starts_before_zero(self) -> None:
        """A clip shorter than the fade would otherwise get a negative start."""
        chain = audio_mix.build_music_filter(1.0, fade_out_s=1.5)
        assert "afade=t=out:st=0.000:d=1.5" in chain

    def test_undacked_chain_has_no_compressor(self) -> None:
        chain = audio_mix.build_music_filter(40.0, duck=False)
        assert "sidechaincompress" not in chain
        assert "asplit" not in chain
        assert "normalize=0" in chain

    def test_music_is_trimmed_to_the_clip(self) -> None:
        """The track is looped on the way in, so the trim is what bounds it."""
        assert "atrim=0:40.000" in audio_mix.build_music_filter(40.0)


class TestChooseTrack:
    def test_no_name_takes_the_first_in_sorted_order(self, music_dir: Path) -> None:
        assert audio_mix.choose_track(music_dir) == music_dir / "a-first.wav"

    def test_ignores_files_that_are_not_audio(self, music_dir: Path) -> None:
        (music_dir / "a-first.wav").unlink()
        assert audio_mix.choose_track(music_dir) == music_dir / "b-second.mp3"

    def test_name_resolves_inside_the_directory(self, music_dir: Path) -> None:
        assert audio_mix.choose_track(music_dir, "b-second.mp3") == music_dir / "b-second.mp3"

    def test_name_resolves_by_stem(self, music_dir: Path) -> None:
        assert audio_mix.choose_track(music_dir, "b-second") == music_dir / "b-second.mp3"

    def test_name_may_be_a_path_outside_the_directory(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere.mp3"
        elsewhere.write_bytes(b"")
        assert audio_mix.choose_track(tmp_path / "music", str(elsewhere)) == elsewhere

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert audio_mix.choose_track(tmp_path / "nope") is None

    def test_empty_directory_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert audio_mix.choose_track(tmp_path / "empty") is None


class TestPlanFromConfig:
    def test_disabled_by_default(self, music_dir: Path) -> None:
        assert audio_mix.plan_from_config({}, music_dir=music_dir) is None

    def test_enabled_in_config_picks_a_track(self, music_dir: Path) -> None:
        plan = audio_mix.plan_from_config({"music_enabled": True}, music_dir=music_dir)
        assert plan is not None
        assert plan.track == music_dir / "a-first.wav"

    def test_override_enables_without_touching_config(self, music_dir: Path) -> None:
        """--music is only ever typed to get a bed, so it implies music_enabled."""
        plan = audio_mix.plan_from_config({}, music_dir=music_dir, override="b-second.mp3")
        assert plan is not None
        assert plan.track.name == "b-second.mp3"

    def test_no_music_beats_both(self, music_dir: Path) -> None:
        plan = audio_mix.plan_from_config(
            {"music_enabled": True},
            music_dir=music_dir,
            override="b-second.mp3",
            disabled=True,
        )
        assert plan is None

    def test_levels_come_from_the_edit_block(self, music_dir: Path) -> None:
        plan = audio_mix.plan_from_config(
            {"music_enabled": True, "music_bed_lufs": -20, "music_duck": False},
            music_dir=music_dir,
        )
        assert plan is not None
        assert plan.bed_lufs == -20.0
        assert plan.duck is False

    def test_the_old_music_bed_db_key_still_reads(self, music_dir: Path) -> None:
        plan = audio_mix.plan_from_config(
            {"music_enabled": True, "music_bed_db": -22},
            music_dir=music_dir,
        )
        assert plan is not None
        assert plan.bed_lufs == -22.0

    def test_named_track_that_does_not_exist_is_an_error(self, music_dir: Path) -> None:
        """Silently rendering without the bed the user asked for is worse."""
        with pytest.raises(MusicMissing):
            audio_mix.plan_from_config({}, music_dir=music_dir, override="ghost.mp3")

    def test_enabled_with_an_empty_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(MusicMissing):
            audio_mix.plan_from_config({"music_enabled": True}, music_dir=tmp_path / "none")


class TestMixMusic:
    def test_missing_track_is_reported_before_ffmpeg_runs(self, tmp_path: Path) -> None:
        speech = tmp_path / "mixed.wav"
        speech.write_bytes(b"")
        plan = MusicPlan(track=tmp_path / "ghost.mp3")
        with pytest.raises(MusicMissing):
            audio_mix.mix_music(speech, tmp_path / "out.wav", plan, duration_s=10.0)

    def test_missing_speech_is_reported_before_ffmpeg_runs(self, tmp_path: Path) -> None:
        track = tmp_path / "bed.mp3"
        track.write_bytes(b"")
        plan = MusicPlan(track=track)
        with pytest.raises(FileNotFoundError):
            audio_mix.mix_music(tmp_path / "gone.wav", tmp_path / "out.wav", plan, duration_s=10.0)
