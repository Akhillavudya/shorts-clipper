"""The web server layer (WEB_UI_PLAN.md).

``clipper`` is eight CLI stages driven by hand, each of which blocks a terminal
for minutes to tens of minutes. This package is the other half: a JSON API and
a job runner, so a browser can drive the pipeline and *you* stop being the
scheduler.

The layering, which the module docstrings repeat where it matters:

- ``review/presenter.py`` stays the read model. It is Flask-free, well tested,
  and was written for exactly this.
- ``schemas.py`` turns presenter dataclasses into JSON, by hand, because their
  useful parts are ``@property`` accessors that ``asdict`` silently drops.
- ``api.py`` is the blueprint: routing and nothing else.
- ``doctor.py`` holds the preflight checks as data, so ``clipper doctor`` and
  ``GET /api/doctor`` cannot drift apart.
- ``text.py`` strips ANSI off anything that crosses into JSON.
- ``queue.py`` is the durable queue: one JSON file under ``jobs/_queue/``,
  written only by the Flask process, read by everything.
- ``events.py`` is the per-entry progress log: one JSONL file, written only by
  the worker child, polled by the browser.

- ``worker.py`` is the child process one job runs in, and ``runner.py`` is
  the supervisor thread that spawns and reaps it.

Still to come, in plan order: ``logging.py`` and ``disk.py`` (phase 6).

**This module re-exports only the Flask-free half**, and that is deliberate.
``cli.py`` imports ``server.doctor``, and ``clipper moments`` should not pay to
import a web framework - the same reason ``flask`` is imported inside the
``review`` command rather than at CLI import time. Importing ``schemas`` here
would pull in ``clipper.review``, and ``api`` would pull in Flask itself, so
both are left to be imported explicitly by the code that actually serves HTTP.

Nothing in this package imports ``pipeline`` at module scope either -
``worker.py`` imports it inside the function that runs a job, and ``worker.py``
runs in a child process. That is what makes an ``edit/track.py`` access
violation take down one job instead of the page you are watching.

``runner`` is not re-exported here for the same reason ``api`` is not: it is
the half that spawns processes, and ``cli.py`` imports this package for
``doctor`` alone. Import ``clipper.server.runner`` explicitly.
"""

from __future__ import annotations

from clipper.server.doctor import Check, diagnose, overall_status, run_checks
from clipper.server.events import Event, EventLog
from clipper.server.queue import QueueEntry, QueueStatus, QueueStore, RunOptions
from clipper.server.text import clean_error, strip_ansi

__all__ = [
    "Check",
    "Event",
    "EventLog",
    "QueueEntry",
    "QueueStatus",
    "QueueStore",
    "RunOptions",
    "clean_error",
    "diagnose",
    "overall_status",
    "run_checks",
    "strip_ansi",
]
