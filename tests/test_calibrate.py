"""The calibration loop.

The fit is the project's stated edge, so the tests lean on the guards rather
than the arithmetic: that a thin sample refuses to re-fit, that a negative
correlation never becomes a negative weight, that no axis is ever zeroed out,
and that writing the weights back does not eat the comments in
``scoring_weights.yaml`` - those comments are what make the numbers mean
anything.

Analytics runs against an ``httpx.MockTransport``; nothing reaches a network
and no OAuth flow is exercised.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, date, datetime

import httpx
import pytest

from clipper.calibrate.analytics import (
    AnalyticsClient,
    CalibrationError,
    RetentionRow,
    Sample,
    access_token,
    collect_samples,
    correlations,
    evaluate,
    fit_weights,
    parse_rows,
    pearson,
    published_video_ids,
    run,
    update_weights_file,
)
from clipper.moments.score import RUBRIC_AXES
from clipper.state import JobStore

CURRENT = {
    "hook": 0.25,
    "self_containment": 0.20,
    "payoff": 0.15,
    "surprise_emotion": 0.15,
    "visual_independence": 0.05,
    "quotability": 0.10,
    "length_fit": 0.10,
}


def rubric(**overrides: float) -> dict[str, float]:
    base = dict.fromkeys(RUBRIC_AXES, 5.0)
    base.update(overrides)
    return base


def sample(index: int, *, hook: float, actual: float, views: int = 1000) -> Sample:
    return Sample(
        job_id=f"job{index}",
        clip_id=f"clip_{index:02d}",
        youtube_video_id=f"yt{index}",
        rubric=rubric(hook=hook),
        predicted=hook,
        actual=actual,
        views=views,
    )


class TestPearson:
    def test_perfect_positive(self) -> None:
        assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_no_variance_is_undefined_not_zero(self) -> None:
        # A model that scored every clip 7 on one axis has said nothing.
        assert pearson([7, 7, 7, 7], [1, 2, 3, 4]) is None

    def test_too_few_pairs_is_undefined(self) -> None:
        assert pearson([1, 2], [3, 4]) is None


class TestFit:
    def test_the_predictive_axis_gains_weight(self) -> None:
        samples = [
            sample(i, hook=float(h), actual=float(h) * 5.0)
            for i, h in enumerate([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
        ]
        fitted, correlation = fit_weights(samples, CURRENT, learning_rate=1.0)

        assert correlation["hook"] == pytest.approx(1.0)
        assert fitted["hook"] > CURRENT["hook"]

    def test_a_negative_correlation_never_becomes_a_negative_weight(self) -> None:
        # At n=50 a negative correlation is far more likely to be noise or
        # selection bias than a finding, and acting on it would start selecting
        # for bad clips.
        samples = [
            sample(i, hook=float(h), actual=100.0 - float(h) * 5.0)
            for i, h in enumerate([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
        ]
        fitted, correlation = fit_weights(samples, CURRENT, learning_rate=1.0)

        assert correlation["hook"] < 0
        assert all(w >= 0 for w in fitted.values())

    def test_no_axis_is_ever_zeroed_out(self) -> None:
        # An axis at zero stops influencing which clips get rendered, so nothing
        # scoring well on it is ever published, so no evidence about it is ever
        # collected again. The floor keeps that door open.
        samples = [
            sample(i, hook=float(h), actual=float(h) * 5.0)
            for i, h in enumerate([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
        ]
        fitted, _ = fit_weights(samples, CURRENT, learning_rate=1.0, floor=0.02)
        assert all(w >= 0.02 for w in fitted.values())

    def test_weights_sum_to_one(self) -> None:
        samples = [
            sample(i, hook=float(h), actual=float(h) * 5.0)
            for i, h in enumerate([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
        ]
        fitted, _ = fit_weights(samples, CURRENT, learning_rate=1.0)
        assert sum(fitted.values()) == pytest.approx(1.0, abs=0.001)

    def test_learning_rate_of_zero_changes_nothing(self) -> None:
        samples = [
            sample(i, hook=float(h), actual=float(h) * 5.0)
            for i, h in enumerate([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
        ]
        fitted, _ = fit_weights(samples, CURRENT, learning_rate=0.0, floor=0.0)
        for axis in RUBRIC_AXES:
            assert fitted[axis] == pytest.approx(CURRENT[axis], abs=0.001)

    def test_nothing_correlating_keeps_the_current_weights(self) -> None:
        # "The rubric is not predictive yet" is not the same claim as "some
        # other weighting is better".
        samples = [sample(i, hook=5.0, actual=float(i)) for i in range(10)]
        fitted, _ = fit_weights(samples, CURRENT)
        assert fitted == CURRENT

    def test_evaluate_compares_two_weightings(self) -> None:
        samples = [
            sample(i, hook=float(h), actual=float(h) * 5.0)
            for i, h in enumerate([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
        ]
        hook_heavy = {**dict.fromkeys(RUBRIC_AXES, 0.0), "hook": 1.0}
        hook_free = {**dict.fromkeys(RUBRIC_AXES, 0.2), "hook": 0.0}

        assert evaluate(samples, hook_heavy) == pytest.approx(1.0)
        assert evaluate(samples, hook_free) is None, "no variance left to correlate"

    def test_correlations_skip_an_axis_with_too_little_data(self) -> None:
        samples = [sample(i, hook=float(i), actual=float(i)) for i in range(2)]
        assert correlations(samples)["hook"] is None


class TestParseRows:
    def test_columns_are_read_from_the_response(self) -> None:
        # Deliberately not in METRICS order: assuming the order would silently
        # train the fit on the likes column the day the API reorders them.
        payload = {
            "columnHeaders": [
                {"name": "video"},
                {"name": "likes"},
                {"name": "averageViewPercentage"},
                {"name": "views"},
            ],
            "rows": [["yt1", 12, 68.5, 5000]],
        }
        rows = parse_rows(payload)
        assert rows["yt1"].average_view_percentage == 68.5
        assert rows["yt1"].views == 5000
        assert rows["yt1"].likes == 12

    def test_a_response_with_no_headers_is_empty_not_an_error(self) -> None:
        assert parse_rows({}) == {}

    def test_a_missing_metric_defaults_to_zero(self) -> None:
        payload = {"columnHeaders": [{"name": "video"}], "rows": [["yt1"]]}
        assert parse_rows(payload)["yt1"].views == 0


class TestAnalyticsClient:
    def test_retention_is_keyed_by_video_id(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "video==" in request.url.params["filters"]
            return httpx.Response(
                200,
                json={
                    "columnHeaders": [
                        {"name": "video"},
                        {"name": "views"},
                        {"name": "averageViewPercentage"},
                    ],
                    "rows": [["yt1", 5000, 71.0], ["yt2", 300, 44.0]],
                },
            )

        client = AnalyticsClient(
            client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://a.test")
        )
        rows = client.retention(["yt1", "yt2"], start_date=date(2026, 6, 1))

        assert set(rows) == {"yt1", "yt2"}
        assert rows["yt1"].average_view_percentage == 71.0

    def test_a_401_explains_the_scope(self) -> None:
        client = AnalyticsClient(
            client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(401, text="nope")),
                base_url="https://a.test",
            )
        )
        with pytest.raises(CalibrationError, match="yt-analytics.readonly"):
            client.retention(["yt1"], start_date=date(2026, 6, 1))

    def test_no_credentials_says_what_to_set(self, monkeypatch) -> None:
        for var in (
            "YOUTUBE_OAUTH_TOKEN",
            "YOUTUBE_REFRESH_TOKEN",
            "YOUTUBE_CLIENT_ID",
            "YOUTUBE_CLIENT_SECRET",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(CalibrationError, match="YOUTUBE_REFRESH_TOKEN"):
            access_token()

    def test_a_refresh_token_is_exchanged(self, monkeypatch) -> None:
        monkeypatch.delenv("YOUTUBE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
        monkeypatch.setenv("YOUTUBE_CLIENT_ID", "cid")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "secret")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "minted"})

        assert access_token(client=httpx.Client(transport=httpx.MockTransport(handler))) == "minted"


# --------------------------------------------------------------------------
# the join
# --------------------------------------------------------------------------


def build_job(store: JobStore, job_id: str, *, published: str | None, score: float = 7.0):
    from clipper.state import JobState, RightsStatus

    state = JobState(video_id=job_id)
    state.rights_status = RightsStatus.green
    clip = state.add_clip(rubric=rubric(hook=8.0), score=score)
    if published:
        clip.youtube_video_id = published
    store.save(state)
    return state


class TestCollectSamples:
    def test_joins_on_the_youtube_video_id(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        build_job(store, "job_a", published="yt1")
        build_job(store, "job_b", published=None)

        retention = {"yt1": RetentionRow("yt1", views=5000, average_view_percentage=70.0)}
        samples, dropped = collect_samples(store, retention)

        assert len(samples) == 1
        assert samples[0].actual == 70.0
        assert dropped["unpublished"] == 1

    def test_low_traffic_rows_are_dropped(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        build_job(store, "job_a", published="yt1")

        retention = {"yt1": RetentionRow("yt1", views=40, average_view_percentage=95.0)}
        samples, dropped = collect_samples(store, retention)

        assert samples == []
        assert dropped["low_views"] == 1

    def test_the_archive_wins_over_the_state(self, tmp_path) -> None:
        # scored.json is the archive the fit is supposed to correlate against
        # (hard rule 8); the state's copy can have been edited since.
        store = JobStore(tmp_path)
        build_job(store, "job_a", published="yt1")
        (store.job_dir("job_a") / "scored.json").write_text(
            '{"clips": [{"clip_id": "clip_01", "rubric": {"hook": 3.0}}]}',
            encoding="utf-8",
        )

        retention = {"yt1": RetentionRow("yt1", views=5000, average_view_percentage=70.0)}
        samples, _ = collect_samples(store, retention)

        assert samples[0].rubric["hook"] == 3.0

    def test_published_ids_are_collected_across_jobs(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        build_job(store, "job_a", published="yt1")
        build_job(store, "job_b", published="yt2")
        assert set(published_video_ids(store)) == {"yt1", "yt2"}


# --------------------------------------------------------------------------
# writing the weights back
# --------------------------------------------------------------------------


WEIGHTS_FILE = textwrap.dedent(
    """\
    # Rubric weights for the rerank pass.
    #
    # These starting values are guesses.

    weights:
      # Weighted hardest because on Shorts nothing else matters first.
      hook: 0.25
      self_containment: 0.20
      payoff: 0.15
      surprise_emotion: 0.15
      visual_independence: 0.05
      quotability: 0.10
      length_fit: 0.10

    modifiers:
      length_penalty:
        enabled: true
        under_min_s: 0.85

    min_score: 5.5
    max_clips_per_video: 3

    calibration:
      last_fit: null
      sample_size: 0
      target_metric: average_view_percentage
      min_sample: 30
    """
)


class TestUpdateWeightsFile:
    def test_comments_survive(self, tmp_path) -> None:
        path = tmp_path / "scoring_weights.yaml"
        path.write_text(WEIGHTS_FILE, encoding="utf-8")

        update_weights_file(
            path,
            {**CURRENT, "hook": 0.4},
            sample_size=52,
            fitted_at=datetime(2026, 9, 13, tzinfo=UTC),
        )
        text = path.read_text(encoding="utf-8")

        assert "# These starting values are guesses." in text
        assert "# Weighted hardest because" in text

    def test_the_numbers_change(self, tmp_path) -> None:
        import yaml

        path = tmp_path / "scoring_weights.yaml"
        path.write_text(WEIGHTS_FILE, encoding="utf-8")

        update_weights_file(
            path,
            {**CURRENT, "hook": 0.4, "payoff": 0.05},
            sample_size=52,
            fitted_at=datetime(2026, 9, 13, tzinfo=UTC),
        )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert data["weights"]["hook"] == 0.4
        assert data["weights"]["payoff"] == 0.05
        assert data["calibration"]["sample_size"] == 52
        assert str(data["calibration"]["last_fit"]) == "2026-09-13"

    def test_unrelated_blocks_are_untouched(self, tmp_path) -> None:
        import yaml

        path = tmp_path / "scoring_weights.yaml"
        path.write_text(WEIGHTS_FILE, encoding="utf-8")

        update_weights_file(path, CURRENT, sample_size=30)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert data["min_score"] == 5.5
        assert data["max_clips_per_video"] == 3
        assert data["modifiers"]["length_penalty"]["under_min_s"] == 0.85
        assert data["modifiers"]["length_penalty"]["enabled"] is True

    def test_the_result_still_parses_as_yaml(self, tmp_path) -> None:
        import yaml

        path = tmp_path / "scoring_weights.yaml"
        path.write_text(WEIGHTS_FILE, encoding="utf-8")
        update_weights_file(path, CURRENT, sample_size=1)
        assert set(yaml.safe_load(path.read_text(encoding="utf-8"))["weights"]) == set(RUBRIC_AXES)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


class FakeAnalytics:
    def __init__(self, rows: dict[str, RetentionRow]) -> None:
        self.rows = rows

    def retention(self, video_ids, **kwargs):
        return {k: v for k, v in self.rows.items() if k in set(video_ids)}


def weights_config(**calibration) -> dict:
    return {
        "weights": dict(CURRENT),
        "calibration": {
            "target_metric": "average_view_percentage",
            "min_sample": 30,
            "learning_rate": 0.5,
            "weight_floor": 0.02,
            **calibration,
        },
    }


class TestRun:
    def test_no_published_clips_says_what_to_do(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        build_job(store, "job_a", published=None)

        report = run(store, weights_config(), client=FakeAnalytics({}))

        assert report.sample_size == 0
        assert "youtube_video_id" in (report.note or "")

    def test_a_thin_sample_refuses_to_refit(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        rows = {}
        for i in range(5):
            build_job(store, f"job_{i}", published=f"yt{i}")
            rows[f"yt{i}"] = RetentionRow(f"yt{i}", views=5000, average_view_percentage=50.0 + i)

        report = run(store, weights_config(min_sample=30), client=FakeAnalytics(rows))

        assert report.sample_size == 5
        assert "below calibration.min_sample" in (report.note or "")
        assert report.fitted_weights == report.current_weights
        assert report.applied is False

    def test_a_dry_run_writes_nothing(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        path = tmp_path / "scoring_weights.yaml"
        path.write_text(WEIGHTS_FILE, encoding="utf-8")
        before = path.read_text(encoding="utf-8")

        rows = {}
        for i in range(40):
            build_job(store, f"job_{i:02d}", published=f"yt{i}")
            rows[f"yt{i}"] = RetentionRow(f"yt{i}", views=5000, average_view_percentage=50.0 + i)

        report = run(
            store, weights_config(min_sample=10), client=FakeAnalytics(rows), weights_path=path
        )

        assert report.applied is False
        assert path.read_text(encoding="utf-8") == before
        assert "dry run" in report.render()

    def test_a_fit_that_does_not_improve_is_not_written(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        path = tmp_path / "scoring_weights.yaml"
        path.write_text(WEIGHTS_FILE, encoding="utf-8")
        before = path.read_text(encoding="utf-8")

        # Every clip has an identical rubric, so no weighting predicts better
        # than any other and there is nothing to learn.
        rows = {}
        for i in range(40):
            build_job(store, f"job_{i:02d}", published=f"yt{i}")
            rows[f"yt{i}"] = RetentionRow(f"yt{i}", views=5000, average_view_percentage=50.0 + i)

        report = run(
            store,
            weights_config(min_sample=10),
            client=FakeAnalytics(rows),
            weights_path=path,
            apply=True,
        )

        assert report.applied is False
        assert path.read_text(encoding="utf-8") == before

    def test_the_report_renders_without_a_fit(self, tmp_path) -> None:
        store = JobStore(tmp_path)
        report = run(store, weights_config(), client=FakeAnalytics({}))
        assert "samples: 0" in report.render()
