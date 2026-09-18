"""Rights status lookup and the export gate (FULL_PLAN.md section 0).

Responsibility: answer "may this source be clipped for a monetized channel?"
and hard-block the renderer when the answer is no.

Tiers, read from config/watchlist.yaml:

- green  - own footage, public clipping policy, written permission, YouTube
           CC-BY licence, US federal works, public domain.
- amber  - mid-tier creator who said yes by email; the email is saved under
           rights/permissions/ and referenced from the watchlist entry.
- red    - creators who run their own official clip channels. This is the
           canonical reused-content pattern; keep it off the monetized channel.

Two traps this module exists to catch:

- Licences that look open but are not. MIT OpenCourseWare and Khan Academy are
  CC BY-NC-SA (no commercial use, and a monetized channel is commercial); TED
  is CC BY-NC-ND (no derivatives at all, and a clip is a derivative). On
  YouTube only the Creative Commons - Attribution (CC BY) filter permits
  commercial derivatives.
- Retrofitting. The gate must exist before the first upload, not after 200.
  Enforcement is at the channel level, so one bad batch costs the whole asset.

The gate refuses red and unknown unless an explicit override is passed, and
every override is recorded in the job state.

Resolution order, most specific first: a watchlist entry always wins, because
it is a human decision about a channel; a per-video CC-BY licence is only
consulted when the channel is unlisted. That order matters - a red channel
that happens to mark one upload CC BY is still a channel running its own clips
operation, and the watchlist is where that judgement lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper.config import WATCHLIST_PATH, load
from clipper.state import JobState, RightsBlocked, RightsStatus

#: yt-dlp reports the YouTube licence as free text. Only this one permits a
#: commercial derivative; every other "creative commons" string does not.
YOUTUBE_CC_BY = "creative commons attribution license (reuse allowed)"

#: Substrings that mark a licence as commercially unusable even though it
#: contains the words "creative commons". Checked before the CC-BY match so a
#: BY-NC-SA string can never be mistaken for BY.
NON_COMMERCIAL_MARKERS = ("-nc", "noncommercial", "non-commercial", "-nd", "noderiv")


class RightsError(RuntimeError):
    """The watchlist itself is malformed - a bad entry, not a blocked export."""


@dataclass(frozen=True)
class ChannelEntry:
    """One channel from ``config/watchlist.yaml``."""

    id: str
    name: str | None
    rights_status: RightsStatus
    note: str | None = None
    source: str | None = None
    licence: str | None = None
    permission_file: str | None = None
    topics: tuple[str, ...] = ()
    max_per_day: int | None = None
    min_duration_s: int | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelEntry:
        channel_id = data.get("id")
        if not channel_id:
            raise RightsError(f"watchlist entry has no 'id': {data!r}")

        raw_status = data.get("rights_status")
        if raw_status is None:
            raise RightsError(
                f"watchlist entry {channel_id!r} has no 'rights_status'. Every "
                f"entry needs one; the gate treats a missing status as blocked, "
                f"but an unclassified entry in the file is a mistake worth "
                f"failing on."
            )
        try:
            status = RightsStatus(str(raw_status).strip().lower())
        except ValueError as exc:
            raise RightsError(
                f"watchlist entry {channel_id!r} has rights_status="
                f"{raw_status!r}; expected one of "
                f"{', '.join(s.value for s in RightsStatus)}"
            ) from exc

        topics = data.get("topics") or []
        return cls(
            id=str(channel_id),
            name=data.get("name"),
            rights_status=status,
            note=data.get("note"),
            source=data.get("source"),
            licence=data.get("licence"),
            permission_file=data.get("permission_file"),
            topics=tuple(str(t) for t in topics),
            max_per_day=data.get("max_per_day"),
            min_duration_s=data.get("min_duration_s"),
            raw=data,
        )

    @property
    def is_exportable(self) -> bool:
        return self.rights_status in {RightsStatus.green, RightsStatus.amber}


@dataclass(frozen=True)
class RightsDecision:
    """Why a job ended up with the status it has. Recorded on the job state."""

    status: RightsStatus
    note: str
    entry: ChannelEntry | None = None


class Watchlist:
    """``config/watchlist.yaml``, indexed by channel id."""

    def __init__(
        self,
        channels: list[ChannelEntry] | None = None,
        defaults: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> None:
        self.defaults = defaults or {}
        self.path = path
        self._by_id: dict[str, ChannelEntry] = {c.id: c for c in (channels or [])}

    @classmethod
    def load(cls, path: Path | str = WATCHLIST_PATH) -> Watchlist:
        data = load(path)
        raw_channels = data.get("channels") or []
        if not isinstance(raw_channels, list):
            raise RightsError(
                f"{path}: 'channels' must be a list, got {type(raw_channels).__name__}"
            )

        entries = [ChannelEntry.from_dict(c) for c in raw_channels]

        duplicates = {e.id for e in entries if sum(1 for x in entries if x.id == e.id) > 1}
        if duplicates:
            raise RightsError(
                f"{path}: duplicate channel ids {sorted(duplicates)}. Two entries "
                f"for one channel means the effective rights status depends on "
                f"file order, which is not a thing to leave to chance."
            )

        return cls(channels=entries, defaults=data.get("defaults") or {}, path=Path(path))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, channel_id: str) -> bool:
        return channel_id in self._by_id

    @property
    def channels(self) -> list[ChannelEntry]:
        return list(self._by_id.values())

    def entry(self, channel_id: str | None) -> ChannelEntry | None:
        if not channel_id:
            return None
        return self._by_id.get(channel_id)

    def min_duration_s(self, channel_id: str | None = None) -> int:
        entry = self.entry(channel_id)
        if entry is not None and entry.min_duration_s is not None:
            return int(entry.min_duration_s)
        return int(self.defaults.get("min_duration_s", 900))

    def max_per_day(self, channel_id: str | None = None) -> int:
        entry = self.entry(channel_id)
        if entry is not None and entry.max_per_day is not None:
            return int(entry.max_per_day)
        return int(self.defaults.get("max_per_day", 2))

    def permission_path(self, channel_id: str) -> Path | None:
        """Absolute path to the saved permission thread for an amber entry."""
        entry = self.entry(channel_id)
        if entry is None or not entry.permission_file:
            return None
        candidate = Path(entry.permission_file)
        if candidate.is_absolute():
            return candidate
        # Watchlist paths are written relative to src/clipper/.
        return (Path(__file__).resolve().parent.parent / candidate).resolve()


def licence_is_commercially_clear(licence: str | None) -> bool:
    """Whether a YouTube licence string permits a monetized derivative.

    Only "Creative Commons Attribution license (reuse allowed)" does. Anything
    carrying NC or ND does not, however open it reads - that is the trap this
    function exists for.
    """
    if not licence:
        return False
    text = licence.strip().lower()
    if any(marker in text for marker in NON_COMMERCIAL_MARKERS):
        return False
    return YOUTUBE_CC_BY in text or text in {"cc-by", "cc by"}


def resolve(
    *,
    channel_id: str | None,
    licence: str | None = None,
    watchlist: Watchlist | None = None,
    override: str | None = None,
) -> RightsDecision:
    """Decide the rights status for one source video.

    ``override`` is the ``--rights`` CLI flag: an explicit human classification
    for a source that is not on the watchlist yet. It wins over everything,
    including a red watchlist entry, because the person typing it is making the
    same call the watchlist encodes - but it is recorded as an override so the
    decision is auditable afterwards.
    """
    if override:
        try:
            status = RightsStatus(override.strip().lower())
        except ValueError as exc:
            raise RightsError(
                f"--rights={override!r} is not one of {', '.join(s.value for s in RightsStatus)}"
            ) from exc
        return RightsDecision(status=status, note=f"set explicitly via --rights={status.value}")

    watchlist = watchlist if watchlist is not None else Watchlist()
    entry = watchlist.entry(channel_id)
    if entry is not None:
        note = entry.note or f"watchlist entry {entry.id}"
        if entry.rights_status is RightsStatus.amber and not entry.permission_file:
            note += " (WARNING: amber with no permission_file recorded)"
        return RightsDecision(status=entry.rights_status, note=note, entry=entry)

    if licence_is_commercially_clear(licence):
        return RightsDecision(
            status=RightsStatus.green,
            note=(
                f"YouTube licence {licence!r} is CC BY, which permits a commercial "
                f"derivative. Channel is not on the watchlist - verify per video, "
                f"a channel default is not a guarantee for any single upload."
            ),
        )

    return RightsDecision(
        status=RightsStatus.unknown,
        note=(
            f"channel {channel_id or '<unknown>'} is not in config/watchlist.yaml"
            + (f" and licence {licence!r} is not CC BY" if licence else "")
            + ". Unclassified is not cleared; classify it before rendering."
        ),
    )


def apply(state: JobState, decision: RightsDecision) -> JobState:
    """Write a decision onto the job state. Called by ingest."""
    state.rights_status = decision.status
    state.rights_note = decision.note
    return state


def check_export(
    state: JobState,
    *,
    allow_red: bool = False,
    reason: str | None = None,
) -> None:
    """The gate. Called by ``edit/render.py`` before it writes anything.

    ``allow_red`` records the override on the job state before asserting, so a
    forced export is permanently visible in ``state.json`` rather than being a
    flag that existed only in one shell invocation.
    """
    if allow_red and not state.is_exportable:
        state.rights_override = True
        state.rights_override_reason = reason or (
            f"--allow-red passed for rights_status={state.rights_status.value}"
        )

    try:
        state.assert_exportable()
    except RightsBlocked as exc:
        raise RightsBlocked(
            f"{exc}\n"
            f"  status note: {state.rights_note or '(none)'}\n"
            f"  Reused-content enforcement is at the CHANNEL level: one bad "
            f"batch suspends monetization for everything."
        ) from exc


__all__ = [
    "NON_COMMERCIAL_MARKERS",
    "YOUTUBE_CC_BY",
    "ChannelEntry",
    "RightsDecision",
    "RightsError",
    "Watchlist",
    "apply",
    "check_export",
    "licence_is_commercially_clear",
    "resolve",
]
