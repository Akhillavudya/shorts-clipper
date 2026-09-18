"""Phase 4 - the review loop (FULL_PLAN.md section 12).

A local Flask page listing the scored candidates for a job, with an inline
preview and keep/cut buttons. It is the only part of the pipeline with a human
in it, and it exists for two reasons:

1. Render is the expensive stage. Watching a forty-second candidate costs forty
   seconds; encoding one that was never going to be used costs minutes of CPU
   and, for a music bed, a second loudnorm pass.
2. **Rejections are training data.** ``scored.json`` keeps the rubric that
   predicted a clip was good (CLAUDE.md hard rule 8); ``approved`` and
   ``review_note`` are the only record of whether it actually was. The
   calibration loop correlates the rubric against retention for the clips that
   shipped - the rejected ones are the other half of that sample, and they only
   exist if someone wrote down the verdict.

The page writes nothing but ``approved``, ``review_note`` and ``reviewed_at``
on a clip. It runs no stage: ``clipper render <job> --approved-only`` is the
next command, and keeping the encode on the CLI means closing the browser tab
cannot leave a half-written mp4 behind.
"""

from __future__ import annotations

from clipper.review.app import create_app, serve
from clipper.review.presenter import (
    ClipCard,
    ClipNotFound,
    Counts,
    JobSummary,
    JobView,
    apply_review,
    build_index,
    build_job,
)

__all__ = [
    "ClipCard",
    "ClipNotFound",
    "Counts",
    "JobSummary",
    "JobView",
    "apply_review",
    "build_index",
    "build_job",
    "create_app",
    "serve",
]
