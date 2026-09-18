"""Making subprocess output safe to put in JSON (WEB_UI_PLAN.md section 9).

ffmpeg and yt-dlp colour their output. Those escape sequences are meaningless
in a browser and ugly in a JSON string, so every error that crosses into the
API is stripped and truncated on the way out.

Two rules, both deliberate:

**The raw log file is never stripped.** ``jobs/_queue/logs/worker-<id>.log``
keeps the original bytes, because the point of a raw log is that it is raw.
Stripping happens only where text is *surfaced* - the ``error`` event the
worker writes, and ``StageState.error`` as ``schemas.py`` serialises it.

**Surfaced errors are truncated, not summarised.** An ffmpeg stderr tail is
fifteen lines of banner and two lines of cause, and guessing which two is how
you lose the cause. Keep the head, mark the cut, and point at the raw log.
"""

from __future__ import annotations

import re

#: CSI sequences (``\x1b[0m``) and the two-character escapes around them.
#: Deliberately narrow: this runs over error text, not arbitrary input, and a
#: greedy pattern here would eat real characters out of a stack trace.
ANSI_RE = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")

#: How much of an error survives into a JSON payload. Two thousand characters
#: is roughly forty lines - enough for a traceback, short of a whole encode log.
MAX_ERROR_CHARS = 2000

TRUNCATION_MARKER = "\n… truncated, see the raw log"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences. ``None``-safe callers pass ``or ''``."""
    return ANSI_RE.sub("", text)


def clean_error(text: str | None, *, limit: int = MAX_ERROR_CHARS) -> str | None:
    """Strip, trim and truncate an error for display. ``None`` stays ``None``.

    An empty or whitespace-only error becomes ``None`` rather than ``""``: a
    stage that failed with no message should read as "no message", not as a
    blank one the UI then renders an empty box for.
    """
    if text is None:
        return None
    cleaned = strip_ansi(text).strip()
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + TRUNCATION_MARKER


__all__ = ["ANSI_RE", "MAX_ERROR_CHARS", "TRUNCATION_MARKER", "clean_error", "strip_ansi"]
