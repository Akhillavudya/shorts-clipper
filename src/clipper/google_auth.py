"""Google OAuth, shared by the two stages that need a real account.

Every other network call in this project authenticates with an API key. Two do
not, and cannot:

- ``calibrate/`` reads the YouTube **Analytics** API, which only ever serves
  the authenticated channel's own data.
- ``publish/upload.py`` calls ``videos.insert``, which writes to a channel.

Both need OAuth, and both need it *non-interactively*, because a stage that
opens a browser is a stage that cannot run from a scheduler. So the split is:

1. **Once, by hand:** ``clipper auth`` runs the consent flow against a loopback
   redirect and prints a refresh token.
2. **Every run after that:** the refresh token is exchanged for a short-lived
   access token with no user present.

The consent half lives here rather than in a stage because both consumers want
it and neither owns it. ``calibrate/analytics.py`` keeps its own
``access_token`` wrapper so its failures stay ``CalibrationError``; it delegates
the actual exchange here rather than carrying a second copy of it.

**The client must be an OAuth "Desktop app" client**, not a Web application
one. Only the desktop type accepts a loopback redirect URI on an arbitrary
port, and the port here is whatever the OS hands out.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import urllib.parse
from dataclasses import dataclass, field

import httpx

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

#: Logical scope names, so nothing outside this module pastes a Google URL.
#: ``youtube.upload`` is the narrowest scope that can insert a video - the
#: broader ``youtube`` scope can also edit and delete everything on the
#: channel, which an automated pipeline has no business being able to do.
SCOPES: dict[str, str] = {
    "upload": "https://www.googleapis.com/auth/youtube.upload",
    "analytics": "https://www.googleapis.com/auth/yt-analytics.readonly",
}

TOKEN_ENV = "YOUTUBE_OAUTH_TOKEN"
REFRESH_TOKEN_ENV = "YOUTUBE_REFRESH_TOKEN"
CLIENT_ID_ENV = "YOUTUBE_CLIENT_ID"
CLIENT_SECRET_ENV = "YOUTUBE_CLIENT_SECRET"


class AuthError(RuntimeError):
    """No usable credentials, or Google refused the ones we had."""


def resolve_scopes(names: list[str] | tuple[str, ...] | str) -> list[str]:
    """Map ``["upload", "analytics"]`` onto the Google scope URLs.

    A full URL passes through untouched, so using a scope this project has not
    named yet does not require editing this module first.
    """
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]

    out: list[str] = []
    for name in names:
        scope = SCOPES.get(name, name)
        if not scope.startswith("https://"):
            known = ", ".join(sorted(SCOPES))
            raise AuthError(f"unknown scope {name!r}; known names: {known}")
        if scope not in out:
            out.append(scope)
    return out


# --------------------------------------------------------------------------
# the non-interactive half: refresh token -> access token
# --------------------------------------------------------------------------


def mint_access_token(
    *,
    token: str | None = None,
    token_env: str = TOKEN_ENV,
    refresh_token: str | None = None,
    refresh_token_env: str = REFRESH_TOKEN_ENV,
    client_id_env: str = CLIENT_ID_ENV,
    client_secret_env: str = CLIENT_SECRET_ENV,
    client: httpx.Client | None = None,
    scope_hint: str = "",
) -> str:
    """An access token, from an env var or from the refresh grant.

    Neither path is interactive. A pasted access token is fine for one run
    (they last an hour); the refresh token is the one to set in ``.env``.

    ``scope_hint`` only ever shapes the error message. The refresh token
    already carries the scopes it was granted, and asking for one it does not
    have fails at the API, not here.
    """
    direct = token or os.environ.get(token_env)
    if direct:
        return direct.strip()

    refresh = refresh_token or os.environ.get(refresh_token_env)
    client_id = os.environ.get(client_id_env)
    client_secret = os.environ.get(client_secret_env)

    if not (refresh and client_id and client_secret):
        raise AuthError(
            f"no OAuth credentials. Set {token_env} to a short-lived access "
            f"token, or set {refresh_token_env}, {client_id_env} and "
            f"{client_secret_env} so one can be minted per run. "
            f"`clipper auth` runs the consent flow that produces them."
            + (f" Scope needed: {scope_hint}" if scope_hint else "")
        )

    return refresh_access_token(
        refresh_token=refresh,
        client_id=client_id,
        client_secret=client_secret,
        client=client,
    )


def refresh_access_token(
    *,
    refresh_token: str,
    client_id: str,
    client_secret: str,
    client: httpx.Client | None = None,
) -> str:
    """Exchange a refresh token for an access token."""
    payload = _token_request(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        client=client,
        what="token refresh",
    )
    minted = payload.get("access_token")
    if not minted:
        raise AuthError(f"token endpoint returned no access_token: {payload}")
    return str(minted)


def _token_request(
    data: dict[str, str],
    *,
    client: httpx.Client | None,
    what: str,
) -> dict[str, object]:
    owned = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        resp = http.post(TOKEN_URL, data=data)
    except httpx.HTTPError as exc:
        raise AuthError(f"{what} failed: {exc}") from exc
    finally:
        if owned:
            http.close()

    if resp.status_code >= 400:
        # `invalid_grant` is the one worth naming: it is what a revoked or
        # expired refresh token returns, and the fix is to run `clipper auth`
        # again rather than to retry.
        body = resp.text[:300]
        hint = (
            "\n  the refresh token has been revoked or has expired - run `clipper auth` again"
            if "invalid_grant" in body
            else ""
        )
        raise AuthError(f"{what} {resp.status_code}: {body}{hint}")

    return resp.json()


# --------------------------------------------------------------------------
# the interactive half: `clipper auth`, run once
# --------------------------------------------------------------------------


@dataclass
class ConsentRequest:
    """One consent attempt: the URL to open, and the secrets it is bound to."""

    url: str
    redirect_uri: str
    state: str
    code_verifier: str
    scopes: list[str] = field(default_factory=list)


def _pkce_pair() -> tuple[str, str]:
    """A PKCE verifier and its S256 challenge.

    PKCE is what stops another process on this machine racing to the loopback
    port and redeeming the authorization code first: the code alone is useless
    without the verifier, which never leaves this process.
    """
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("ascii").rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def build_consent_request(
    *,
    client_id: str,
    redirect_uri: str,
    scopes: list[str],
) -> ConsentRequest:
    """The URL to send the user to.

    ``access_type=offline`` with ``prompt=consent`` is what makes Google return
    a refresh token. Without ``prompt=consent`` a *second* authorization of an
    already-approved client returns only an access token, and the flow appears
    to succeed while producing nothing worth keeping.
    """
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    return ConsentRequest(
        url=f"{AUTH_URL}?{query}",
        redirect_uri=redirect_uri,
        state=state,
        code_verifier=verifier,
        scopes=list(scopes),
    )


def exchange_code(
    code: str,
    request: ConsentRequest,
    *,
    client_id: str,
    client_secret: str,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    """Redeem an authorization code for tokens.

    Returns the whole token payload rather than just the refresh token: the
    granted ``scope`` is in there, and a flow that quietly granted less than it
    asked for is worth showing the user before they save the result.
    """
    payload = _token_request(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "code_verifier": request.code_verifier,
            "grant_type": "authorization_code",
            "redirect_uri": request.redirect_uri,
        },
        client=client,
        what="code exchange",
    )
    if not payload.get("refresh_token"):
        raise AuthError(
            "Google returned no refresh_token. That happens when the client "
            "was already authorized and consent was not re-prompted; revoke "
            "the app at https://myaccount.google.com/permissions and retry."
        )
    return payload


def run_consent_flow(
    *,
    client_id: str,
    client_secret: str,
    scopes: list[str],
    host: str = "127.0.0.1",
    port: int = 0,
    timeout_s: float = 300.0,
    open_browser: bool = True,
) -> dict[str, object]:  # pragma: no cover - needs a browser and a live Google
    """The one-time browser step. Serves a loopback redirect and waits.

    The server accepts exactly one request and shuts down, and it binds to
    loopback on an OS-assigned port: a consent redirect carries an
    authorization code in its query string, and anything that can read that URL
    can redeem it.
    """
    import http.server
    import threading
    import webbrowser

    captured: dict[str, str] = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - the stdlib names it
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            captured.update({k: v[0] for k, v in params.items() if v})
            ok = "code" in captured
            body = (
                b"<h1>Authorized.</h1><p>Close this tab and return to the terminal.</p>"
                if ok
                else b"<h1>Authorization failed.</h1><p>Check the terminal.</p>"
            )
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

        def log_message(self, *args: object) -> None:
            """Silence the stdlib access log on stderr."""

    server = http.server.HTTPServer((host, port), Handler)
    bound_port = server.server_address[1]
    request = build_consent_request(
        client_id=client_id,
        redirect_uri=f"http://{host}:{bound_port}",
        scopes=scopes,
    )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if open_browser:
            webbrowser.open(request.url)
        print(f"  open this if the browser did not:\n  {request.url}\n")
        if not done.wait(timeout_s):
            raise AuthError(f"no consent redirect arrived within {timeout_s:.0f}s")
    finally:
        server.shutdown()
        server.server_close()

    if "error" in captured:
        raise AuthError(f"consent was refused: {captured['error']}")
    if captured.get("state") != request.state:
        raise AuthError("the consent redirect carried the wrong state; discarded")

    return exchange_code(
        captured["code"],
        request,
        client_id=client_id,
        client_secret=client_secret,
    )


__all__ = [
    "AUTH_URL",
    "CLIENT_ID_ENV",
    "CLIENT_SECRET_ENV",
    "REFRESH_TOKEN_ENV",
    "SCOPES",
    "TOKEN_ENV",
    "TOKEN_URL",
    "AuthError",
    "ConsentRequest",
    "build_consent_request",
    "exchange_code",
    "mint_access_token",
    "refresh_access_token",
    "resolve_scopes",
    "run_consent_flow",
]
