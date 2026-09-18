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

Still to come, in plan order: ``queue.py`` and ``events.py`` (phase 1),
``worker.py`` and ``runner.py`` (phase 2), ``logging.py`` and ``disk.py``
(phase 6).

**This module re-exports only the Flask-free half**, and that is deliberate.
``cli.py`` imports ``server.doctor``, and ``clipper moments`` should not pay to
import a web framework - the same reason ``flask`` is imported inside the
``review`` command rather than at CLI import time. Importing ``schemas`` here
would pull in ``clipper.review``, and ``api`` would pull in Flask itself, so
both are left to be imported explicitly by the code that actually serves HTTP.

Nothing in this package imports ``pipeline`` either. The pipeline runs in a
child process (``worker.py``), and keeping that import out of the server means
an ``edit/track.py`` crash cannot take down the page you are watching.
"""

from __future__ import annotations

from clipper.server.doctor import Check, diagnose, overall_status, run_checks
from clipper.server.text import clean_error, strip_ansi

__all__ = [
    "Check",
    "clean_error",
    "diagnose",
    "overall_status",
    "run_checks",
    "strip_ansi",
]
