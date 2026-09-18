"""The calibration loop (FULL_PLAN.md section 5, and section 12 Phase 5).

Responsibility: pull retention data from the YouTube Analytics API for clips
already published, join it against the rubric scores saved in scored.json, and
re-fit the weights in config/scoring_weights.yaml.

This is the part no off-the-shelf tool has. After ~50 published Shorts the
correlation between predicted score and actual retention is a real signal, and
tuning against it is the project's actual differentiator - not the crop
smoother, not the caption renderer.

How the fit works, and why it is this and not a regression:

At the sample sizes this project will ever see - 50 Shorts, maybe 200 - fitting
seven free parameters by least squares overfits badly, and the seven rubric
axes are correlated with each other (a clip with a strong hook usually has a
payoff), which makes the individual coefficients unstable and occasionally
negative for no real reason. So the fit is deliberately blunt: correlate each
axis against retention on its own, keep the positive correlations, and set the
weights proportional to them. It cannot express "hook matters only when
self-containment is high", and it does not pretend to.

Three guards that matter more than the arithmetic:

- **A floor under every weight.** An axis weighted to zero stops influencing
  which clips get rendered, so no clip that scores well on it is ever
  published, so no data about it is ever collected, so it can never come back.
  That is selection bias closing a door permanently. The floor keeps every axis
  in the sample.
- **A blend with the current weights.** One fit on fifty noisy samples should
  nudge the weights, not replace them. ``learning_rate`` controls how far.
- **A minimum sample size.** Below ``calibration.min_sample`` the correlation is
  noise, and the honest output is a report saying so rather than new weights.

Writing the result back is a surgical line edit rather than a YAML dump.
``config/scoring_weights.yaml`` is a hand-maintained file whose comments explain
what each axis means, and round-tripping it through ``yaml.safe_dump`` would
delete all of them to save twenty lines of code.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from clipper import google_auth
from clipper.moments.score import RUBRIC_AXES, SCORED_FILENAME, weighted_total
from clipper.state import JobStore

ANALYTICS_BASE = "https://youtubeanalytics.googleapis.com/v2"
TOKEN_URL = google_auth.TOKEN_URL

#: The Analytics API is read-only for our purposes, and this is the narrowest
#: scope that returns retention for one's own channel.
SCOPE = google_auth.SCOPES["analytics"]

#: Metrics pulled per video. ``averageViewPercentage`` is the retention figure
#: the fit targets; the others are context for the report, and ``views`` is what
#: decides whether a row has enough traffic to mean anything.
METRICS = ("views", "averageViewDuration", "averageViewPercentage", "likes", "comments")

#: Rows below this many views are dropped. A Short with 40 views has a retention
#: figure dominated by who happened to see it, and averaging it with a Short
#: that got 40,000 treats the two as equally informative.
MIN_VIEWS = 250

DEFAULT_TARGET_METRIC = "average_view_percentage"


class CalibrationError(RuntimeError):
    """Analytics could not be fetched, or the fit cannot be trusted."""


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


def access_token(
    *,
    token: str | None = None,
    token_env: str = "YOUTUBE_OAUTH_TOKEN",
    refresh_token: str | None = None,
    refresh_token_env: str = "YOUTUBE_REFRESH_TOKEN",
    client_id_env: str = "YOUTUBE_CLIENT_ID",
    client_secret_env: str = "YOUTUBE_CLIENT_SECRET",
    client: httpx.Client | None = None,
) -> str:
    """An OAuth access token for the Analytics API.

    Two paths, neither interactive. Either an access token is handed in (short
    lived, fine for one run), or a refresh token is exchanged for one. The
    consent flow that produces the refresh token in the first place is a
    one-time browser step (``clipper auth``), deliberately outside the stages -
    a CLI stage that opens a browser is a CLI stage that cannot run from a
    scheduler.

    An API key is not an option here: unlike the Data API, Analytics only serves
    data for the authenticated channel, so it requires OAuth by construction.

    The exchange itself lives in ``clipper.google_auth``, shared with
    ``publish/upload.py``. This wrapper exists to keep the failure a
    ``CalibrationError``, because that is what the rest of this module raises
    and what the CLI reports against.
    """
    try:
        return google_auth.mint_access_token(
            token=token,
            token_env=token_env,
            refresh_token=refresh_token,
            refresh_token_env=refresh_token_env,
            client_id_env=client_id_env,
            client_secret_env=client_secret_env,
            client=client,
            scope_hint=SCOPE,
        )
    except google_auth.AuthError as exc:
        raise CalibrationError(str(exc)) from exc


# --------------------------------------------------------------------------
# the API
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionRow:
    """One published Short's performance."""

    video_id: str
    views: int = 0
    average_view_duration_s: float = 0.0
    average_view_percentage: float = 0.0
    likes: int = 0
    comments: int = 0

    def metric(self, name: str) -> float | None:
        return {
            "average_view_percentage": self.average_view_percentage,
            "average_view_duration_s": self.average_view_duration_s,
            "views": float(self.views),
            "likes": float(self.likes),
            "comments": float(self.comments),
        }.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "views": self.views,
            "average_view_duration_s": self.average_view_duration_s,
            "average_view_percentage": self.average_view_percentage,
            "likes": self.likes,
            "comments": self.comments,
        }


class AnalyticsClient:
    """Read-only YouTube Analytics API v2 client."""

    def __init__(
        self,
        *,
        token: str | None = None,
        client: httpx.Client | None = None,
        base_url: str = ANALYTICS_BASE,
        timeout_s: float = 60.0,
    ) -> None:
        self._token = token
        self._client = client
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _ensure_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        token = self._token or access_token()
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=self.timeout_s,
        )
        return self._client

    def query(self, params: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            resp = client.get("/reports", params=params, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise CalibrationError(f"analytics query failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise CalibrationError(
                f"analytics rejected the credentials ({resp.status_code}). The "
                f"token must carry {SCOPE} and belong to the channel that owns "
                f"the videos: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise CalibrationError(f"analytics {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def retention(
        self,
        video_ids: list[str],
        *,
        start_date: date | str,
        end_date: date | str | None = None,
        ids: str = "channel==MINE",
    ) -> dict[str, RetentionRow]:
        """Per-video retention, keyed by video id.

        The API caps ``filters`` at 500 video ids per request, so this batches.
        Videos with no rows - too new, or no traffic at all - are simply absent
        from the result rather than present with zeros, because "no data yet"
        and "nobody watched" are different facts and only one of them belongs
        in a correlation.
        """
        unique = list(dict.fromkeys(v for v in video_ids if v))
        rows: dict[str, RetentionRow] = {}

        start = start_date.isoformat() if isinstance(start_date, date) else str(start_date)
        finish = (
            end_date.isoformat() if isinstance(end_date, date) else str(end_date or date.today())
        )

        for offset in range(0, len(unique), 500):
            batch = unique[offset : offset + 500]
            payload = self.query(
                {
                    "ids": ids,
                    "startDate": start,
                    "endDate": finish,
                    "metrics": ",".join(METRICS),
                    "dimensions": "video",
                    "filters": "video==" + ",".join(batch),
                    "maxResults": 500,
                }
            )
            rows.update(parse_rows(payload))

        return rows

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def parse_rows(payload: dict[str, Any]) -> dict[str, RetentionRow]:
    """Turn the API's column-headers-plus-rows shape into records.

    The response describes its own columns, so the order is read from
    ``columnHeaders`` rather than assumed to match ``METRICS``. Assuming the
    order would silently swap two metrics the day the API reorders them, and a
    retention fit trained on the likes column fails quietly.
    """
    headers = [str(h.get("name") or "") for h in payload.get("columnHeaders") or []]
    if not headers:
        return {}

    index = {name: position for position, name in enumerate(headers)}
    video_col = index.get("video")
    if video_col is None:
        return {}

    def value(row: list[Any], name: str, default: float = 0.0) -> float:
        position = index.get(name)
        if position is None or position >= len(row):
            return default
        try:
            return float(row[position] or 0)
        except (TypeError, ValueError):
            return default

    rows: dict[str, RetentionRow] = {}
    for row in payload.get("rows") or []:
        if not isinstance(row, list) or video_col >= len(row):
            continue
        video_id = str(row[video_col])
        rows[video_id] = RetentionRow(
            video_id=video_id,
            views=int(value(row, "views")),
            average_view_duration_s=value(row, "averageViewDuration"),
            average_view_percentage=value(row, "averageViewPercentage"),
            likes=int(value(row, "likes")),
            comments=int(value(row, "comments")),
        )
    return rows


# --------------------------------------------------------------------------
# the join
# --------------------------------------------------------------------------


@dataclass
class Sample:
    """One published clip: what the rubric predicted, and what happened."""

    job_id: str
    clip_id: str
    youtube_video_id: str
    rubric: dict[str, float]
    predicted: float
    actual: float
    views: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "clip_id": self.clip_id,
            "youtube_video_id": self.youtube_video_id,
            "rubric": self.rubric,
            "predicted": round(self.predicted, 4),
            "actual": round(self.actual, 4),
            "views": self.views,
        }


def load_archived_rubrics(job_dir: Path | str) -> dict[str, dict[str, float]]:
    """Raw rubric scores from ``scored.json``, keyed by clip id.

    ``scored.json`` is the archive, and it holds every candidate that was
    scored, not just the ones that shipped. The state file holds the rubric too,
    but only for shortlisted clips - reading the archive means a clip whose
    state was edited after the fact still correlates against what the model
    actually said (CLAUDE.md hard rule 8).
    """
    path = Path(job_dir) / SCORED_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    rubrics: dict[str, dict[str, float]] = {}
    for clip in payload.get("clips") or []:
        clip_id = str(clip.get("clip_id") or "")
        rubric = clip.get("rubric") or {}
        if clip_id and rubric:
            rubrics[clip_id] = {axis: float(rubric[axis]) for axis in RUBRIC_AXES if axis in rubric}
    return rubrics


def collect_samples(
    store: JobStore,
    retention: dict[str, RetentionRow],
    *,
    target_metric: str = DEFAULT_TARGET_METRIC,
    min_views: int = MIN_VIEWS,
) -> tuple[list[Sample], dict[str, int]]:
    """Join published clips to their retention figures.

    Walks every job, not a published index: the state files *are* the index, and
    a separate one could go stale. Returns the samples plus counts of what was
    dropped, so a fit that finds three samples can say why rather than just
    reporting three.
    """
    samples: list[Sample] = []
    dropped = {"unpublished": 0, "no_analytics": 0, "low_views": 0, "no_rubric": 0}

    for job_id in store.list_jobs():
        try:
            state = store.load(job_id)
        except (OSError, ValueError):
            continue

        archived = load_archived_rubrics(store.job_dir(job_id))

        for clip in state.clips:
            if not clip.youtube_video_id:
                dropped["unpublished"] += 1
                continue

            row = retention.get(clip.youtube_video_id)
            if row is None:
                dropped["no_analytics"] += 1
                continue
            if row.views < min_views:
                dropped["low_views"] += 1
                continue

            rubric = archived.get(clip.clip_id) or dict(clip.rubric or {})
            rubric = {axis: float(v) for axis, v in rubric.items() if axis in RUBRIC_AXES}
            if not rubric:
                dropped["no_rubric"] += 1
                continue

            actual = row.metric(target_metric)
            if actual is None:
                dropped["no_analytics"] += 1
                continue

            samples.append(
                Sample(
                    job_id=job_id,
                    clip_id=clip.clip_id,
                    youtube_video_id=clip.youtube_video_id,
                    rubric=rubric,
                    predicted=float(clip.score or 0.0),
                    actual=float(actual),
                    views=row.views,
                )
            )

    return samples, dropped


def published_video_ids(store: JobStore) -> list[str]:
    """Every YouTube video id recorded across all jobs."""
    ids: list[str] = []
    for job_id in store.list_jobs():
        try:
            state = store.load(job_id)
        except (OSError, ValueError):
            continue
        ids.extend(c.youtube_video_id for c in state.clips if c.youtube_video_id)
    return list(dict.fromkeys(ids))


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson's r, or ``None`` when it is undefined.

    Undefined means fewer than three pairs, or one series with no variance at
    all - which happens for real: a model that scored ``visual_independence``
    at 7 for every clip in the sample has said nothing to correlate.
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]

    numerator = sum(a * b for a, b in zip(dx, dy, strict=True))
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denominator == 0:
        return None
    return numerator / denominator


def correlations(samples: list[Sample]) -> dict[str, float | None]:
    """Each rubric axis against the observed metric, independently."""
    actual = [s.actual for s in samples]
    results: dict[str, float | None] = {}
    for axis in RUBRIC_AXES:
        pairs = [(s.rubric[axis], s.actual) for s in samples if axis in s.rubric]
        if len(pairs) < 3:
            results[axis] = None
            continue
        results[axis] = pearson([p[0] for p in pairs], [p[1] for p in pairs])
    del actual
    return results


def fit_weights(
    samples: list[Sample],
    current: dict[str, float],
    *,
    learning_rate: float = 0.5,
    floor: float = 0.02,
) -> tuple[dict[str, float], dict[str, float | None]]:
    """Re-fit the rubric weights from observed retention.

    Returns the new weights and the per-axis correlations behind them.

    An axis that correlates negatively contributes nothing to the target rather
    than a negative weight. A negative weight would say "clips that score worse
    on payoff do better", which at n=50 is far more likely to be noise, or an
    artefact of what got published, than a real finding - and acting on it would
    start selecting for bad clips.
    """
    correlation = correlations(samples)

    positives = {axis: max(0.0, r) for axis, r in correlation.items() if r is not None and r > 0}
    total = sum(positives.values())

    if total <= 0:
        # Nothing correlated positively. Keeping the current weights is the
        # honest response: the data says the rubric is not predictive yet, not
        # that some other weighting is better.
        return dict(current), correlation

    target = {axis: positives.get(axis, 0.0) / total for axis in RUBRIC_AXES}

    rate = max(0.0, min(1.0, float(learning_rate)))
    blended = {
        axis: (1.0 - rate) * float(current.get(axis, 0.0)) + rate * target[axis]
        for axis in RUBRIC_AXES
    }

    # The floor is *reserved* before the rest is shared out, not clamped after.
    # Clamping and then renormalizing divides the floored axes back down below
    # the floor, which is the one thing the floor exists to prevent.
    floor = max(0.0, float(floor))
    reserved = floor * len(RUBRIC_AXES)
    mass = sum(blended.values())
    if reserved >= 1.0 or mass <= 0:
        even = 1.0 / len(RUBRIC_AXES)
        return {axis: round(even, 4) for axis in RUBRIC_AXES}, correlation

    share = 1.0 - reserved
    return {
        axis: round(floor + share * value / mass, 4) for axis, value in blended.items()
    }, correlation


def evaluate(samples: list[Sample], weights: dict[str, float]) -> float | None:
    """How well a weighting predicts the observed metric, as Pearson's r.

    Used to compare the proposed weights against the ones in force. A fit that
    does not improve this number is not an improvement, whatever the individual
    correlations say.
    """
    if len(samples) < 3:
        return None
    predicted = [weighted_total(s.rubric, weights) for s in samples]
    return pearson(predicted, [s.actual for s in samples])


# --------------------------------------------------------------------------
# writing the result back
# --------------------------------------------------------------------------

_WEIGHT_LINE = re.compile(r"^(?P<indent>\s+)(?P<axis>[a-z_]+):\s*(?P<value>[0-9.]+)\s*$")
_CAL_LINE = re.compile(r"^(?P<indent>\s+)(?P<key>last_fit|sample_size):\s*(?P<value>.*)$")


def update_weights_file(
    path: Path | str,
    weights: dict[str, float],
    *,
    sample_size: int = 0,
    fitted_at: datetime | None = None,
) -> Path:
    """Rewrite the numbers in ``scoring_weights.yaml`` and nothing else.

    A line edit rather than a YAML round trip. The file's comments are the
    documentation for what each axis means and why it is weighted as it is;
    ``yaml.safe_dump`` would delete every one of them, and the next person to
    open the file would find seven bare numbers.

    Only lines inside the top-level ``weights:`` block whose key is a known
    rubric axis are touched, plus ``last_fit`` and ``sample_size`` under
    ``calibration:``. Anything else in the file is passed through byte for byte.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    stamp = (fitted_at or datetime.now(UTC)).date().isoformat()

    section: str | None = None
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not line[0].isspace() and stripped.endswith(":"):
            section = stripped[:-1]
            out.append(line)
            continue

        if section == "weights":
            match = _WEIGHT_LINE.match(line)
            if match and match.group("axis") in weights:
                axis = match.group("axis")
                out.append(f"{match.group('indent')}{axis}: {weights[axis]:.4f}")
                continue

        if section == "calibration":
            match = _CAL_LINE.match(line)
            if match:
                key = match.group("key")
                value = stamp if key == "last_fit" else str(sample_size)
                out.append(f"{match.group('indent')}{key}: {value}")
                continue

        out.append(line)

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """What one calibration run found. Printed, and saved next to the weights."""

    sample_size: int
    target_metric: str
    dropped: dict[str, int] = field(default_factory=dict)
    correlations: dict[str, float | None] = field(default_factory=dict)
    current_weights: dict[str, float] = field(default_factory=dict)
    fitted_weights: dict[str, float] = field(default_factory=dict)
    current_r: float | None = None
    fitted_r: float | None = None
    applied: bool = False
    note: str | None = None
    samples: list[Sample] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """Whether the new weighting predicts better than the one in force."""
        if self.fitted_r is None or self.current_r is None:
            return False
        return self.fitted_r > self.current_r

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "target_metric": self.target_metric,
            "dropped": self.dropped,
            "correlations": self.correlations,
            "current_weights": self.current_weights,
            "fitted_weights": self.fitted_weights,
            "current_r": self.current_r,
            "fitted_r": self.fitted_r,
            "improved": self.improved,
            "applied": self.applied,
            "note": self.note,
            "samples": [s.to_dict() for s in self.samples],
        }

    def render(self) -> str:
        """A plain-text report. This is the output when nothing is applied."""
        lines = [
            f"samples: {self.sample_size}  metric: {self.target_metric}",
        ]
        if self.dropped:
            dropped = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items()) if v)
            if dropped:
                lines.append(f"dropped: {dropped}")
        if self.note:
            lines.append(self.note)

        if self.correlations:
            lines.append("")
            lines.append(f"{'axis':<22}{'r':>8}{'weight':>10}{'fitted':>10}")
            for axis in RUBRIC_AXES:
                r = self.correlations.get(axis)
                r_text = f"{r:+.3f}" if r is not None else "     -"
                lines.append(
                    f"{axis:<22}{r_text:>8}"
                    f"{self.current_weights.get(axis, 0.0):>10.4f}"
                    f"{self.fitted_weights.get(axis, 0.0):>10.4f}"
                )

        lines.append("")
        current = f"{self.current_r:+.3f}" if self.current_r is not None else "-"
        fitted = f"{self.fitted_r:+.3f}" if self.fitted_r is not None else "-"
        lines.append(f"predicted-vs-actual r: current {current} -> fitted {fitted}")
        lines.append(
            "weights written to config/scoring_weights.yaml"
            if self.applied
            else "dry run - nothing written. Pass --apply to write the fitted weights."
        )
        return "\n".join(lines)


def run(
    store: JobStore,
    weights_config: dict[str, Any],
    *,
    client: AnalyticsClient | None = None,
    weights_path: Path | str | None = None,
    apply: bool = False,
    days: int = 90,
    now: Callable[[], datetime] | None = None,
) -> CalibrationReport:
    """Pull retention, join, fit, and optionally write the weights back.

    ``apply`` defaults to false. Rewriting the file that decides which clips get
    made is not something to do as a side effect of looking at a report.
    """
    now_fn = now or (lambda: datetime.now(UTC))
    calibration = weights_config.get("calibration") or {}
    target_metric = str(calibration.get("target_metric") or DEFAULT_TARGET_METRIC)
    min_sample = int(calibration.get("min_sample", 30))
    configured = weights_config.get("weights") or {}
    current = {axis: float(configured.get(axis, 0.0)) for axis in RUBRIC_AXES}

    video_ids = published_video_ids(store)
    if not video_ids:
        return CalibrationReport(
            sample_size=0,
            target_metric=target_metric,
            current_weights=current,
            fitted_weights=current,
            note=(
                "no clip has a youtube_video_id recorded. Publish a Short, then "
                "run `clipper publish <job> --clip N --published <video id>` so "
                "the retention figure has something to join against."
            ),
        )

    analytics = client or AnalyticsClient()
    start = (now_fn() - timedelta(days=max(1, int(days)))).date()
    retention = analytics.retention(video_ids, start_date=start, end_date=now_fn().date())

    samples, dropped = collect_samples(store, retention, target_metric=target_metric)

    report = CalibrationReport(
        sample_size=len(samples),
        target_metric=target_metric,
        dropped=dropped,
        current_weights=current,
        fitted_weights=current,
        samples=samples,
    )

    if len(samples) < min_sample:
        report.note = (
            f"{len(samples)} samples is below calibration.min_sample "
            f"({min_sample}); correlations are reported but the weights are not "
            f"re-fit. At this size the noise dominates."
        )
        report.correlations = correlations(samples)
        report.current_r = evaluate(samples, current)
        return report

    fitted, correlation = fit_weights(
        samples,
        current,
        learning_rate=float(calibration.get("learning_rate", 0.5)),
        floor=float(calibration.get("weight_floor", 0.02)),
    )
    report.correlations = correlation
    report.fitted_weights = fitted
    report.current_r = evaluate(samples, current)
    report.fitted_r = evaluate(samples, fitted)

    if not apply:
        return report

    if not report.improved:
        report.note = (
            "the fitted weights do not predict better than the ones in force; "
            "refusing to write them. Collect more samples."
        )
        return report

    update_weights_file(
        weights_path or Path("config/scoring_weights.yaml"),
        fitted,
        sample_size=len(samples),
        fitted_at=now_fn(),
    )
    report.applied = True
    return report


__all__ = [
    "ANALYTICS_BASE",
    "DEFAULT_TARGET_METRIC",
    "METRICS",
    "MIN_VIEWS",
    "SCOPE",
    "TOKEN_URL",
    "AnalyticsClient",
    "CalibrationError",
    "CalibrationReport",
    "RetentionRow",
    "Sample",
    "access_token",
    "collect_samples",
    "correlations",
    "evaluate",
    "fit_weights",
    "load_archived_rubrics",
    "parse_rows",
    "pearson",
    "published_video_ids",
    "run",
    "update_weights_file",
]
