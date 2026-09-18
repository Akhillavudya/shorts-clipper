"""``videos.insert`` - the upload itself (FULL_PLAN.md section 12, Phase 5).

``metadata.py`` builds the text that accompanies an upload; this sends the
bytes. They are separate modules because they fail for entirely different
reasons: a bad title is a formatting problem caught instantly, and a half-sent
1080x1920 file is a network problem that must be recoverable.

Four decisions worth stating, because each one is a bug if reversed:

- **Resumable, never single-request.** A simple POST of the whole file has no
  way to answer "how much of that did you actually get?", so any failure at
  90% costs the whole upload. The resumable protocol makes the server
  authoritative about the offset: after any failure we *ask* rather than
  assume, and re-send only what is missing.
- **The session URI is persisted next to the clip.** Google keeps a resumable
  session alive for about a week. A crashed run that had sent 80% should
  resume into the same session rather than start a second upload of the same
  Short, which would leave a duplicate video on the channel.
- **An upload is charged to the same daily quota as discovery.**
  ``videos.insert`` costs 1600 units against the *project's* 10,000/day - the
  same budget ``discover`` spends one unit at a time. They share the ledger
  under ``jobs/.discover/quota.json`` because they share the budget; three
  uploads a day is 4800 units, and finding out about that by having discovery
  start failing tomorrow morning is the failure this prevents.
- **The rights gate fires a third time.** ``render`` gates the file,
  ``publish`` gates the text, and this gates the irreversible part. The first
  two produce something on disk that a human still has to act on; this one
  puts it on a channel whose monetization is enforced at the channel level.

Uploads land ``private`` unless ``config/settings.yaml`` says otherwise. The
last place a bad clip can be caught is a human opening it in YouTube Studio,
and flipping a good one to public is one click.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

UPLOAD_ENDPOINT = "https://www.googleapis.com/upload/youtube/v3/videos"
WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
STUDIO_URL = "https://studio.youtube.com/video/{video_id}/edit"

#: The scope ``videos.insert`` needs. Narrow on purpose - see ``google_auth``.
SCOPE = "https://www.googleapis.com/auth/youtube.upload"

#: What one ``videos.insert`` costs against the project's 10,000 units/day.
UPLOAD_QUOTA_COST = 1600

#: Every chunk except the last must be a multiple of 256 KiB. That is the
#: protocol, not a preference: a chunk of any other size is rejected outright.
CHUNK_MULTIPLE = 256 * 1024
DEFAULT_CHUNK_BYTES = 32 * CHUNK_MULTIPLE  # 8 MiB

#: Where the resumable session is remembered between runs.
SESSION_FILENAME = ".upload_session.json"

#: Google expires a resumable session after about a week. Six days, so a
#: session is abandoned while it is still definitely valid rather than
#: discovered dead at byte zero.
SESSION_TTL_S = 6 * 24 * 3600

MIME_TYPE = "video/mp4"

#: Retried with backoff. 5xx is the API being unwell; 429 is rate limiting.
#: 403 is *not* here: on this endpoint it means quota or permissions, and
#: retrying it burns nothing but time.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

#: The session is gone - it expired, or it was already completed and cleaned
#: up. Either way the offset cannot be queried and a fresh session is needed.
DEAD_SESSION_STATUS = frozenset({404, 410})

_RANGE = re.compile(r"bytes=(\d+)-(\d+)")

ProgressFn = Callable[[int, int], None] | None


class UploadError(RuntimeError):
    """The upload failed, or was refused before it started."""


class SessionExpired(UploadError):
    """The resumable session is dead; a new one must be started."""


@dataclass
class UploadResult:
    """What came back, plus what it cost to get there."""

    video_id: str
    clip_id: str = ""
    privacy_status: str = ""
    title: str = ""
    bytes_sent: int = 0
    file_size: int = 0
    resumed_from: int = 0
    attempts: int = 0
    response: dict[str, Any] = field(default_factory=dict)

    @property
    def watch_url(self) -> str:
        return WATCH_URL.format(video_id=self.video_id)

    @property
    def studio_url(self) -> str:
        return STUDIO_URL.format(video_id=self.video_id)


# --------------------------------------------------------------------------
# the request body
# --------------------------------------------------------------------------


def insert_body(meta: Any) -> dict[str, Any]:
    """The ``videos.insert`` resource, from an ``UploadMetadata`` or its dict.

    Only ``snippet`` and ``status`` are sent. ``upload.json`` also carries the
    rights status and the source URL - those are our bookkeeping and mean
    nothing to the API, and posting unknown keys is how a request gets
    rejected for a reason that has nothing to do with the video.
    """
    raw = meta.to_dict() if hasattr(meta, "to_dict") else dict(meta)

    snippet = dict(raw.get("snippet") or {})
    status = dict(raw.get("status") or {})

    if not snippet.get("title"):
        raise UploadError("refusing to upload a video with no title")
    if "selfDeclaredMadeForKids" not in status:
        # Required on insert since 2020. Omitting it is a 400, and defaulting
        # it to True would restrict comments and personalized ads on a clip
        # that is not made for kids.
        status["selfDeclaredMadeForKids"] = False

    return {"snippet": snippet, "status": status}


def read_upload_json(clip_dir: Path | str) -> dict[str, Any]:
    """Load ``clips/clip_NN/upload.json``, written by ``publish``."""
    path = Path(clip_dir) / "upload.json"
    if not path.is_file():
        raise UploadError(
            f"no upload.json in {clip_dir}. Run `clipper publish <job> --clip N` "
            f"first - the title, description and credit are built there."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UploadError(f"{path} is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------
# the resumable session, remembered on disk
# --------------------------------------------------------------------------


def file_digest(path: Path | str) -> str:
    """SHA-256 of the whole file.

    Whole file, not a sample of it: this is what decides whether a half-sent
    upload may be resumed, and the encode that produced the bytes already on
    the server can differ from the one on disk anywhere. Hashing 16 MB costs
    milliseconds against an upload that costs seconds.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Session:
    """A resumable upload session, tied to one exact file."""

    uri: str
    file_size: int
    file_digest: str
    started_at: float

    def matches(self, path: Path, *, now: float) -> bool:
        """Whether this session can still be resumed for ``path``.

        Identified by content, not by timestamp. A re-render produces a
        different file of a very similar length, and resuming a session opened
        for the old bytes splices two encodes into something that decodes as
        neither - an mtime is too weak a guard for a mistake that silent.
        """
        if now - self.started_at > SESSION_TTL_S:
            return False
        if path.stat().st_size != self.file_size:
            return False
        return file_digest(path) == self.file_digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "file_size": self.file_size,
            "file_digest": self.file_digest,
            "started_at": self.started_at,
        }

    @classmethod
    def opening(cls, uri: str, path: Path | str, *, now: float) -> Session:
        """A session for the file as it is right now."""
        path = Path(path)
        return cls(
            uri=uri,
            file_size=path.stat().st_size,
            file_digest=file_digest(path),
            started_at=now,
        )


def session_path(clip_dir: Path | str) -> Path:
    return Path(clip_dir) / SESSION_FILENAME


def load_session(clip_dir: Path | str) -> Session | None:
    path = session_path(clip_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session(
            uri=str(data["uri"]),
            file_size=int(data["file_size"]),
            file_digest=str(data["file_digest"]),
            started_at=float(data["started_at"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        # An unreadable session file means one fact is unknown; the upload can
        # still proceed from zero. Deleting it here keeps the next run clean.
        path.unlink(missing_ok=True)
        return None


def save_session(clip_dir: Path | str, session: Session) -> Path:
    path = session_path(clip_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    return path


def clear_session(clip_dir: Path | str) -> None:
    session_path(clip_dir).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


class ResumableUpload:
    """The resumable ``videos.insert`` protocol, and nothing else.

    Knows about HTTP, offsets and retries. Knows nothing about jobs, clips or
    the rights gate - ``pipeline.upload`` owns that half, which is what makes
    this testable against an ``httpx.MockTransport`` with no state file in
    sight.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        client: httpx.Client | None = None,
        endpoint: str = UPLOAD_ENDPOINT,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        max_attempts: int = 5,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        timeout_s: float = 300.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._token = token
        self._client = client
        self.endpoint = endpoint
        self.chunk_bytes = normalize_chunk_size(chunk_bytes)
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._now = now
        self.attempts = 0

    # -- plumbing -------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        from clipper.google_auth import AuthError, mint_access_token

        try:
            token = self._token or mint_access_token(scope_hint=SCOPE)
        except AuthError as exc:
            raise UploadError(str(exc)) from exc
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            timeout=self.timeout_s,
        )
        return self._client

    def _wait(self, attempt: int, resp: httpx.Response | None = None) -> None:
        """Back off, honouring ``Retry-After`` when the server sent one."""
        delay = min(self.backoff_max_s, self.backoff_base_s * (2 ** max(0, attempt - 1)))
        if resp is not None:
            header = resp.headers.get("retry-after")
            if header:
                # Retry-After may be an HTTP date rather than seconds. Our own
                # backoff is a fine answer to one we cannot parse.
                with contextlib.suppress(ValueError):
                    delay = max(delay, float(header))
        self._sleep(delay)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    # -- the protocol ---------------------------------------------------

    def start(
        self,
        body: dict[str, Any],
        *,
        file_size: int,
        notify_subscribers: bool = False,
        mime_type: str = MIME_TYPE,
    ) -> str:
        """Open a session. Returns the session URI from the ``Location`` header.

        The metadata travels in this request; the bytes never do. That is why
        a rejected title fails here, in one cheap round trip, instead of after
        sending 12 MB.
        """
        http = self._http()
        params = {
            "uploadType": "resumable",
            "part": "snippet,status",
            "notifySubscribers": "true" if notify_subscribers else "false",
        }
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": mime_type,
            "X-Upload-Content-Length": str(file_size),
        }

        last: httpx.Response | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.attempts += 1
            try:
                resp = http.post(
                    self.endpoint,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=self.timeout_s,
                )
            except httpx.HTTPError as exc:
                if attempt >= self.max_attempts:
                    raise UploadError(f"could not open an upload session: {exc}") from exc
                self._wait(attempt)
                continue

            if resp.status_code in (200, 201):
                location = resp.headers.get("location")
                if not location:
                    raise UploadError(
                        "the API accepted the session but sent no Location header, "
                        "so there is nowhere to send the file"
                    )
                return location

            if resp.status_code in RETRY_STATUS and attempt < self.max_attempts:
                last = resp
                self._wait(attempt, resp)
                continue

            raise _api_error("opening the upload session", resp)

        raise _api_error("opening the upload session", last)

    def offset(self, session_uri: str, file_size: int) -> int | dict[str, Any]:
        """Ask the server how much it has. The only trustworthy answer.

        Returns the next byte to send, or the finished video resource if the
        upload turns out to have completed - which is exactly what a run that
        died between the last chunk and writing the state file looks like.
        """
        http = self._http()
        try:
            resp = http.put(
                session_uri,
                headers={"Content-Range": f"bytes */{file_size}", "Content-Length": "0"},
                content=b"",
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise UploadError(f"could not query the upload offset: {exc}") from exc

        if resp.status_code in (200, 201):
            return _video_resource(resp)
        if resp.status_code in DEAD_SESSION_STATUS:
            raise SessionExpired(f"the upload session is gone ({resp.status_code})")
        if resp.status_code == 308:
            return _next_offset(resp)
        raise _api_error("querying the upload offset", resp)

    def send(
        self,
        path: Path | str,
        session_uri: str,
        *,
        start_at: int = 0,
        on_progress: ProgressFn = None,
    ) -> dict[str, Any]:
        """Send the file from ``start_at``, and return the video resource.

        Every failure re-asks the server for the offset rather than assuming
        the chunk it just sent was lost: a timeout on the response says
        nothing about whether the request arrived, and re-sending bytes the
        server already has is the one thing the protocol lets us avoid.
        """
        path = Path(path)
        file_size = path.stat().st_size
        offset = start_at
        attempt = 0
        http = self._http()

        with path.open("rb") as fh:
            while offset < file_size:
                fh.seek(offset)
                chunk = fh.read(self.chunk_bytes)
                if not chunk:
                    raise UploadError(f"read no bytes at offset {offset} of {file_size} in {path}")

                last_byte = offset + len(chunk) - 1
                attempt += 1
                self.attempts += 1
                resp: httpx.Response | None = None
                try:
                    resp = http.put(
                        session_uri,
                        headers={
                            "Content-Type": MIME_TYPE,
                            "Content-Length": str(len(chunk)),
                            "Content-Range": f"bytes {offset}-{last_byte}/{file_size}",
                        },
                        content=chunk,
                        timeout=self.timeout_s,
                    )
                except httpx.HTTPError as exc:
                    if attempt >= self.max_attempts:
                        raise UploadError(
                            f"upload failed at byte {offset} of {file_size}: {exc}"
                        ) from exc
                    self._wait(attempt)
                    offset = self._resync(session_uri, file_size, fallback=offset)
                    continue

                if resp.status_code in (200, 201):
                    if on_progress is not None:
                        on_progress(file_size, file_size)
                    return _video_resource(resp)

                if resp.status_code == 308:
                    offset = _next_offset(resp, fallback=offset + len(chunk))
                    attempt = 0
                    if on_progress is not None:
                        on_progress(offset, file_size)
                    continue

                if resp.status_code in DEAD_SESSION_STATUS:
                    raise SessionExpired(
                        f"the upload session expired mid-transfer ({resp.status_code})"
                    )

                if resp.status_code in RETRY_STATUS and attempt < self.max_attempts:
                    self._wait(attempt, resp)
                    offset = self._resync(session_uri, file_size, fallback=offset)
                    continue

                raise _api_error(f"uploading at byte {offset}", resp)

        # The loop ended without a 2xx: every byte was acknowledged by a 308
        # but the final response never came. Asking settles it.
        final = self.offset(session_uri, file_size)
        if isinstance(final, dict):
            return final
        raise UploadError(
            f"the server acknowledged {final} of {file_size} bytes but never "
            f"returned the finished video"
        )

    def _resync(self, session_uri: str, file_size: int, *, fallback: int) -> int:
        """Re-ask for the offset after a failure, keeping ``fallback`` if the
        query itself fails - a bad answer here is worse than an old one."""
        try:
            where = self.offset(session_uri, file_size)
        except SessionExpired:
            raise
        except UploadError:
            return fallback
        if isinstance(where, dict):
            # Completed while we were not looking. Sending zero more bytes and
            # letting the caller's final query pick it up keeps one exit path.
            return file_size
        return where


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def normalize_chunk_size(value: int) -> int:
    """Round a chunk size down to a multiple of 256 KiB, minimum one multiple."""
    units = max(1, int(value) // CHUNK_MULTIPLE)
    return units * CHUNK_MULTIPLE


def _next_offset(resp: httpx.Response, *, fallback: int = 0) -> int:
    """The byte after the last one the server acknowledged.

    A 308 with no ``Range`` header means the server has stored nothing, which
    is byte 0 - not "carry on where you were".
    """
    header = resp.headers.get("range")
    if not header:
        return 0
    match = _RANGE.search(header)
    if not match:
        return fallback
    return int(match.group(2)) + 1


def _video_resource(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError as exc:
        raise UploadError(f"the upload finished but the response was not JSON: {exc}") from exc
    if not isinstance(data, dict) or not data.get("id"):
        raise UploadError(f"the upload finished but no video id came back: {str(data)[:300]}")
    return data


def _api_error(what: str, resp: httpx.Response | None) -> UploadError:
    """Turn an API refusal into a message that says what to do about it."""
    if resp is None:
        return UploadError(f"{what} failed")

    body = resp.text[:400]
    hint = ""
    if resp.status_code in (401, 403):
        if "uploadLimitExceeded" in body:
            hint = (
                "\n  the channel hit its daily upload limit - this is YouTube's "
                "per-channel cap, not the API quota, and it resets on its own"
            )
        elif "quotaExceeded" in body:
            hint = (
                f"\n  the project is out of Data API quota for the day. One "
                f"upload costs {UPLOAD_QUOTA_COST} of 10,000 units, so "
                f"discovery will be failing too until the UTC-midnight reset"
            )
        elif "youtubeSignupRequired" in body:
            hint = "\n  the authorized Google account has no YouTube channel"
        else:
            hint = (
                f"\n  the token must carry {SCOPE} and belong to the channel "
                f"being uploaded to. `clipper auth` mints one"
            )
    elif resp.status_code == 400:
        hint = "\n  the metadata was rejected - check title, description and categoryId"

    return UploadError(f"{what}: HTTP {resp.status_code} {body}{hint}")


__all__ = [
    "CHUNK_MULTIPLE",
    "DEFAULT_CHUNK_BYTES",
    "SCOPE",
    "SESSION_FILENAME",
    "SESSION_TTL_S",
    "UPLOAD_ENDPOINT",
    "UPLOAD_QUOTA_COST",
    "WATCH_URL",
    "ResumableUpload",
    "Session",
    "SessionExpired",
    "UploadError",
    "UploadResult",
    "clear_session",
    "file_digest",
    "insert_body",
    "load_session",
    "normalize_chunk_size",
    "read_upload_json",
    "save_session",
    "session_path",
]
