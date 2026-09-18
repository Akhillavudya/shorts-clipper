"""Shared fixtures. Nothing here touches a network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


class FakeWallClock:
    """Manually advanced UTC wall clock, for the daily-quota reset boundary."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class FakeClock:
    """Manually advanced monotonic clock.

    Time-dependent tests drive this instead of sleeping, so the suite is
    deterministic and runs in milliseconds regardless of the delays involved.

    ``sleep`` advances the wall clock too. Real sleeping moves both, and the
    daily quota resets on the wall clock while the buckets refill on the
    monotonic one - decoupling them here would make a passing test impossible
    for reasons that do not exist in production.
    """

    def __init__(self, start: float = 1000.0, wall: FakeWallClock | None = None) -> None:
        self.now = start
        self.wall = wall
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self.now += seconds
        if self.wall is not None:
            self.wall.advance(seconds=seconds)

    @property
    def total_slept(self) -> float:
        return sum(self.slept)


@pytest.fixture
def wall_clock() -> FakeWallClock:
    return FakeWallClock()


@pytest.fixture
def clock(wall_clock: FakeWallClock) -> FakeClock:
    return FakeClock(wall=wall_clock)


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "jobs" / "vid123"
    d.mkdir(parents=True)
    return d
