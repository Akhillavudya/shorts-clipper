"""Title, description, and creator credit (FULL_PLAN.md sections 0 and 11).

Responsibility: build the upload metadata for a finished clip.

The credit block is not decoration - crediting the creator on-screen and in the
description, and linking the full episode, is part of the transformation
argument that separates a clip from a copy. Equally, the metadata must never be
shaped so the channel reads as the creator's own official account.

Pulls suggested_title from the rerank pass, and the source link and rights note
from the job state.

Three things this module refuses to do, each for a concrete reason:

- **Build metadata for a source the gate blocks.** The gate fires in
  ``edit/render.py`` because that is where an uploadable file appears; it fires
  again here because this is where the *text that accompanies an upload*
  appears, and a red source with a ready-made title and description is an
  upload waiting to happen at 1am. Same override flag, same audit trail.
- **Name the creator as the author.** The description says "Clip from" and names
  the channel as the source; it never uses the creator's name as the byline, and
  the disclaimer line says plainly that this is an unofficial clip. A channel
  that reads as somebody's official account is the impersonation half of the
  reused-content problem, distinct from the copyright half and judged
  separately.
- **Silently truncate mid-word.** YouTube's limits are hard (100 characters of
  title, 5000 of description, 500 across all tags) and a title cut mid-word
  looks like a broken bot. Truncation happens on a word boundary and is
  reported in the returned object so a caller can see it happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipper.rights import registry
from clipper.state import ClipState, JobState, RightsStatus

UPLOAD_FILENAME = "upload.json"

#: YouTube's hard limits. Exceeding any of them is a rejected upload.
TITLE_LIMIT = 100
DESCRIPTION_LIMIT = 5000
TAGS_TOTAL_LIMIT = 500

#: "People & Blogs". The safe default for a talking-head clip; set
#: ``publish.category_id`` in settings.yaml for a channel with a real niche.
DEFAULT_CATEGORY_ID = "22"

#: Uploads start private. An automated pipeline that publishes straight to
#: public removes the last place a bad clip can be caught, and the cost of
#: flipping a good one to public by hand is a single click.
DEFAULT_PRIVACY = "private"

#: Appended to the title when it fits. Shorts are detected by aspect ratio and
#: length, not by this tag, so it is a convention rather than a requirement -
#: which is why it is dropped rather than truncating the title to fit it.
SHORTS_SUFFIX = "#Shorts"

#: The line that keeps the channel from reading as the creator's own. Short,
#: plain, and last, so it survives the description being collapsed in the app.
DISCLAIMER = "Unofficial fan clip. Not affiliated with or endorsed by the original creator."


class PublishBlocked(RuntimeError):
    """Metadata was requested for a clip that is not ready to be published."""


def truncate_words(text: str, limit: int) -> tuple[str, bool]:
    """Trim to ``limit`` characters on a word boundary.

    Returns the text and whether anything was cut, so the caller can record it
    rather than discovering a half-sentence title after upload.
    """
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat, False
    cut = flat[:limit].rsplit(" ", 1)[0].rstrip(" ,.-:;")
    return (cut or flat[:limit]).rstrip(), True


@dataclass
class UploadMetadata:
    """Everything ``videos.insert`` needs, plus why it says what it says."""

    clip_id: str
    video_id: str
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    category_id: str = DEFAULT_CATEGORY_ID
    privacy_status: str = DEFAULT_PRIVACY
    made_for_kids: bool = False
    source_url: str | None = None
    rights_status: str = RightsStatus.unknown.value
    rights_note: str | None = None
    attribution_required: bool = False
    title_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "video_id": self.video_id,
            "snippet": {
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "categoryId": self.category_id,
            },
            "status": {
                "privacyStatus": self.privacy_status,
                "selfDeclaredMadeForKids": self.made_for_kids,
            },
            "source_url": self.source_url,
            "rights_status": self.rights_status,
            "rights_note": self.rights_note,
            "attribution_required": self.attribution_required,
            "title_truncated": self.title_truncated,
        }


def build_title(
    clip: ClipState, state: JobState, *, add_shorts_tag: bool = True
) -> tuple[str, bool]:
    """The rerank pass's suggested title, made upload-safe.

    Falls back through hook_rewrite to the source title. A clip with no title of
    its own is better labelled with the episode's title than with ``clip_03``.
    """
    raw = (clip.suggested_title or clip.hook_rewrite or state.title or clip.clip_id).strip()

    if add_shorts_tag and SHORTS_SUFFIX.lower() not in raw.lower():
        room = TITLE_LIMIT - len(SHORTS_SUFFIX) - 1
        body, truncated = truncate_words(raw, room)
        return f"{body} {SHORTS_SUFFIX}", truncated

    return truncate_words(raw, TITLE_LIMIT)


def credit_block(state: JobState, entry: Any = None) -> str:
    """The attribution lines. This is the transformation argument in text form.

    Ordered so the two facts that matter - who made it, where the whole thing
    is - are the first two lines, because the description is collapsed by
    default and those are the lines a viewer sees without tapping.
    """
    creator = state.channel_title or state.channel_id or "the original creator"
    lines = [f"Clip from: {state.title or 'the full episode'}", f"Creator: {creator}"]

    if state.source_url:
        lines.append(f"Watch the full episode: {state.source_url}")

    handle = getattr(entry, "raw", None) or {}
    credit_handle = handle.get("credit_handle") if isinstance(handle, dict) else None
    if credit_handle:
        # Some permissions are granted on condition of a specific on-screen or
        # in-description handle. That condition is part of the permission, so
        # dropping it would invalidate the thing that made the clip legal.
        lines.append(f"Credit: {credit_handle}")

    if registry.licence_is_commercially_clear(state.licence):
        lines.append(
            "Source licensed under Creative Commons Attribution (CC BY 3.0). "
            "https://creativecommons.org/licenses/by/3.0/"
        )

    return "\n".join(lines)


def build_description(
    clip: ClipState,
    state: JobState,
    *,
    entry: Any = None,
    extra: str | None = None,
) -> str:
    """Hook, credit, disclaimer - in that order, then trimmed to the limit.

    The disclaimer goes last and is protected from truncation: if the
    description is long enough to be cut, the line that stops the channel
    reading as the creator's own account is the one line that must survive.
    """
    parts: list[str] = []

    opening = (clip.hook_rewrite or clip.suggested_title or "").strip()
    if opening:
        parts.append(opening)

    parts.append(credit_block(state, entry))

    if extra:
        parts.append(extra.strip())

    parts.append(DISCLAIMER)

    body = "\n\n".join(p for p in parts if p)
    if len(body) <= DESCRIPTION_LIMIT:
        return body

    room = DESCRIPTION_LIMIT - len(DISCLAIMER) - 2
    head, _ = truncate_words("\n\n".join(parts[:-1]), max(0, room))
    return f"{head}\n\n{DISCLAIMER}"


def build_tags(clip: ClipState, state: JobState, *, entry: Any = None) -> list[str]:
    """Topic tags from the watchlist, plus the clip's own type.

    Kept short on purpose. Tags are a weak ranking signal and a strong way to
    look like a content farm; the watchlist's ``topics`` are the ones a human
    already decided this channel is about.
    """
    tags: list[str] = ["shorts"]

    for topic in getattr(entry, "topics", ()) or ():
        cleaned = str(topic).strip().lower()
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    if clip.type and clip.type.lower() not in tags:
        tags.append(clip.type.lower())

    channel = (state.channel_title or "").strip().lower()
    if channel and channel not in tags:
        tags.append(channel)

    # YouTube counts the total characters across every tag, not per tag.
    kept: list[str] = []
    used = 0
    for tag in tags:
        cost = len(tag) + 1
        if used + cost > TAGS_TOTAL_LIMIT:
            break
        kept.append(tag)
        used += cost
    return kept


def build(
    state: JobState,
    clip: ClipState,
    *,
    config: dict[str, Any] | None = None,
    watchlist: registry.Watchlist | None = None,
    allow_red: bool = False,
    override_reason: str | None = None,
) -> UploadMetadata:
    """Assemble the upload metadata for one rendered clip.

    Runs the rights gate first. Producing the text that accompanies an upload
    for a source the gate blocks is one copy-paste away from the upload itself,
    and the enforcement that follows is channel-level.
    """
    registry.check_export(state, allow_red=allow_red, reason=override_reason)

    config = config or {}
    entry = watchlist.entry(state.channel_id) if watchlist is not None else None

    title, truncated = build_title(
        clip, state, add_shorts_tag=bool(config.get("add_shorts_tag", True))
    )

    return UploadMetadata(
        clip_id=clip.clip_id,
        video_id=state.video_id,
        title=title,
        description=build_description(clip, state, entry=entry, extra=config.get("extra_text")),
        tags=build_tags(clip, state, entry=entry),
        category_id=str(config.get("category_id", DEFAULT_CATEGORY_ID)),
        privacy_status=str(config.get("privacy_status", DEFAULT_PRIVACY)),
        made_for_kids=bool(config.get("made_for_kids", False)),
        source_url=state.source_url,
        rights_status=state.rights_status.value,
        rights_note=state.rights_note,
        attribution_required=registry.licence_is_commercially_clear(state.licence)
        or state.rights_status is RightsStatus.amber,
        title_truncated=truncated,
    )


def upload_path(clip_dir: Path | str) -> Path:
    return Path(clip_dir) / UPLOAD_FILENAME


def save(meta: UploadMetadata, clip_dir: Path | str) -> Path:
    """Write ``clips/clip_NN/upload.json`` next to ``final.mp4``."""
    path = upload_path(clip_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return path


def record_published(clip: ClipState, youtube_video_id: str, *, when: Any = None) -> ClipState:
    """Note the live video id on the clip. This is the calibration join key.

    Recorded by hand after an upload (``clipper publish <job> --clip 1
    --published <id>``) because upload itself is optional in the plan - section
    12 Phase 5 lists ``videos.insert`` as a "maybe". The calibration loop only
    needs the id, not the mechanism that produced it.
    """
    from clipper.state import utcnow

    clip.youtube_video_id = youtube_video_id.strip()
    clip.published_at = when or utcnow()
    return clip


__all__ = [
    "DEFAULT_CATEGORY_ID",
    "DEFAULT_PRIVACY",
    "DESCRIPTION_LIMIT",
    "DISCLAIMER",
    "SHORTS_SUFFIX",
    "TAGS_TOTAL_LIMIT",
    "TITLE_LIMIT",
    "UPLOAD_FILENAME",
    "PublishBlocked",
    "UploadMetadata",
    "build",
    "build_description",
    "build_tags",
    "build_title",
    "credit_block",
    "record_published",
    "save",
    "truncate_words",
    "upload_path",
]
