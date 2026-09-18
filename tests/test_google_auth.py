"""The shared OAuth helper.

The consent flow itself needs a browser and a live Google, so what is tested
here is everything either side of it: the URL that is built, the code exchange,
the refresh grant, and the two failure modes that look like success if nobody
checks - a consent that returns no refresh token, and a revoked one.

Every request goes through an ``httpx.MockTransport``.
"""

from __future__ import annotations

import urllib.parse

import httpx
import pytest

from clipper.google_auth import (
    AuthError,
    ConsentRequest,
    build_consent_request,
    exchange_code,
    mint_access_token,
    resolve_scopes,
)

OAUTH_VARS = (
    "YOUTUBE_OAUTH_TOKEN",
    "YOUTUBE_REFRESH_TOKEN",
    "YOUTUBE_CLIENT_ID",
    "YOUTUBE_CLIENT_SECRET",
)


@pytest.fixture(autouse=True)
def no_ambient_credentials(monkeypatch):
    """The developer's own .env must never decide what a test does."""
    for var in OAUTH_VARS:
        monkeypatch.delenv(var, raising=False)


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestScopes:
    def test_names_map_to_urls(self) -> None:
        assert resolve_scopes(["upload", "analytics"]) == [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/yt-analytics.readonly",
        ]

    def test_a_comma_separated_string_is_accepted(self) -> None:
        assert len(resolve_scopes("upload, analytics")) == 2

    def test_duplicates_collapse(self) -> None:
        assert len(resolve_scopes("upload,upload")) == 1

    def test_a_full_url_passes_through(self) -> None:
        url = "https://www.googleapis.com/auth/youtube.readonly"
        assert resolve_scopes([url]) == [url]

    def test_a_typo_is_refused_rather_than_sent_to_google(self) -> None:
        with pytest.raises(AuthError, match="unknown scope"):
            resolve_scopes(["uplaod"])


class TestMintAccessToken:
    def test_an_access_token_in_the_environment_is_used_as_is(self, monkeypatch) -> None:
        monkeypatch.setenv("YOUTUBE_OAUTH_TOKEN", "  direct  ")
        assert mint_access_token() == "direct"

    def test_no_credentials_names_every_variable_and_the_command(self) -> None:
        with pytest.raises(AuthError) as exc:
            mint_access_token(scope_hint="https://example.test/scope")
        message = str(exc.value)
        assert "YOUTUBE_REFRESH_TOKEN" in message
        assert "clipper auth" in message
        assert "https://example.test/scope" in message

    def test_the_refresh_grant_is_exchanged(self, monkeypatch) -> None:
        monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
        monkeypatch.setenv("YOUTUBE_CLIENT_ID", "cid")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "secret")
        sent: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            sent.update(urllib.parse.parse_qsl(request.content.decode()))
            return httpx.Response(200, json={"access_token": "minted"})

        assert mint_access_token(client=client(handler)) == "minted"
        assert sent["grant_type"] == "refresh_token"
        assert sent["refresh_token"] == "r"

    def test_a_revoked_refresh_token_says_to_run_auth_again(self, monkeypatch) -> None:
        monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
        monkeypatch.setenv("YOUTUBE_CLIENT_ID", "cid")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "secret")

        handler = lambda r: httpx.Response(400, text='{"error": "invalid_grant"}')  # noqa: E731
        with pytest.raises(AuthError, match="clipper auth"):
            mint_access_token(client=client(handler))

    def test_a_200_with_no_token_is_not_a_success(self, monkeypatch) -> None:
        monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
        monkeypatch.setenv("YOUTUBE_CLIENT_ID", "cid")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "secret")

        with pytest.raises(AuthError, match="no access_token"):
            mint_access_token(client=client(lambda r: httpx.Response(200, json={"scope": "x"})))


class TestConsentRequest:
    def test_asks_for_offline_access_and_forces_the_consent_screen(self) -> None:
        request = build_consent_request(
            client_id="cid",
            redirect_uri="http://127.0.0.1:8123",
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
        )
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(request.url).query))

        # Without both of these Google returns an access token and no refresh
        # token, and the flow looks like it worked.
        assert params["access_type"] == "offline"
        assert params["prompt"] == "consent"
        assert params["code_challenge_method"] == "S256"
        assert params["redirect_uri"] == "http://127.0.0.1:8123"

    def test_the_verifier_is_not_the_challenge(self) -> None:
        request = build_consent_request(client_id="c", redirect_uri="http://x", scopes=["s"])
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(request.url).query))
        assert params["code_challenge"] != request.code_verifier

    def test_two_requests_do_not_share_state(self) -> None:
        one = build_consent_request(client_id="c", redirect_uri="http://x", scopes=["s"])
        two = build_consent_request(client_id="c", redirect_uri="http://x", scopes=["s"])
        assert one.state != two.state
        assert one.code_verifier != two.code_verifier


class TestExchangeCode:
    def request(self) -> ConsentRequest:
        return ConsentRequest(
            url="https://accounts.test/auth",
            redirect_uri="http://127.0.0.1:8123",
            state="st",
            code_verifier="verifier",
        )

    def test_sends_the_verifier_and_returns_the_whole_payload(self) -> None:
        sent: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            sent.update(urllib.parse.parse_qsl(request.content.decode()))
            return httpx.Response(
                200, json={"refresh_token": "1//rt", "scope": "a b", "access_token": "at"}
            )

        payload = exchange_code(
            "code123",
            self.request(),
            client_id="cid",
            client_secret="secret",
            client=client(handler),
        )
        assert sent["code_verifier"] == "verifier"
        assert sent["grant_type"] == "authorization_code"
        assert payload["refresh_token"] == "1//rt"
        # The granted scope comes back so the caller can notice it is short.
        assert payload["scope"] == "a b"

    def test_a_consent_with_no_refresh_token_is_an_error(self) -> None:
        with pytest.raises(AuthError, match="no refresh_token"):
            exchange_code(
                "code123",
                self.request(),
                client_id="cid",
                client_secret="secret",
                client=client(lambda r: httpx.Response(200, json={"access_token": "at"})),
            )
