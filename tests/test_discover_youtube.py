"""The quota-aware YouTube client.

The tests that matter here are about *not spending money*: that search.list is
refused outright, that the ledger survives a restart, and that ids are batched
50 to a call rather than one call each. Everything runs against an
``httpx.MockTransport``; nothing reaches a network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from clipper.discover.youtube import (
    MAX_IDS_PER_CALL,
    DiscoveryError,
    PlaylistCache,
    QuotaExceeded,
    QuotaLedger,
    VideoRecord,
    YouTubeClient,
    parse_duration,
    parse_timestamp,
)


def fake_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test")


def json_response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


class TestParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("PT40S", 40.0),
            ("PT2M30S", 150.0),
            ("PT1H2M3S", 3723.0),
            ("P1DT1H", 90000.0),
            ("P0D", 0.0),
            ("", None),
            (None, None),
            ("banana", None),
        ],
    )
    def test_duration(self, raw, expected) -> None:
        assert parse_duration(raw) == expected

    def test_timestamp_is_utc_aware(self) -> None:
        parsed = parse_timestamp("2026-09-07T12:00:00Z")
        assert parsed == datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

    def test_bad_timestamp_is_none_not_an_error(self) -> None:
        # One malformed field in a 50-video batch must not abort the scan.
        assert parse_timestamp("not a date") is None

    def test_hidden_view_count_is_none_not_zero(self) -> None:
        record = VideoRecord.from_api({"id": "v1", "statistics": {}})
        assert record.view_count is None


class TestQuotaLedger:
    def test_refuses_to_pass_the_ceiling(self) -> None:
        ledger = QuotaLedger(ceiling=3)
        ledger.spend(2, resource="videos")
        with pytest.raises(QuotaExceeded):
            ledger.spend(2, resource="videos")
        assert ledger.used == 2, "a refused spend must not be charged"

    def test_ceiling_of_zero_disables_the_check(self) -> None:
        ledger = QuotaLedger(ceiling=0)
        ledger.spend(10_000)
        assert ledger.remaining > 0

    def test_survives_a_restart_on_the_same_day(self, tmp_path) -> None:
        now = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
        path = tmp_path / "quota.json"

        first = QuotaLedger(path, ceiling=100, now_fn=lambda: now)
        first.spend(40, resource="channels")

        second = QuotaLedger(path, ceiling=100, now_fn=lambda: now)
        assert second.used == 40
        assert second.spent_this_run == 0, "this run has spent nothing yet"

    def test_resets_on_a_new_utc_day(self, tmp_path) -> None:
        path = tmp_path / "quota.json"
        QuotaLedger(
            path, ceiling=100, now_fn=lambda: datetime(2026, 9, 13, 23, 0, tzinfo=UTC)
        ).spend(90)

        tomorrow = QuotaLedger(
            path, ceiling=100, now_fn=lambda: datetime(2026, 9, 14, 1, 0, tzinfo=UTC)
        )
        assert tomorrow.used == 0

    def test_a_corrupt_ledger_starts_from_zero(self, tmp_path) -> None:
        path = tmp_path / "quota.json"
        path.write_text("{not json", encoding="utf-8")
        assert QuotaLedger(path, ceiling=100).used == 0


class TestSearchIsRefused:
    def test_search_raises_before_any_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("search.list must never reach the transport")

        client = YouTubeClient(api_key="k", client=fake_client(handler))
        with pytest.raises(DiscoveryError, match="search.list costs 100"):
            client.get("search", {"q": "podcast"})

    def test_search_is_not_charged_to_the_ledger(self) -> None:
        ledger = QuotaLedger(ceiling=1000)
        client = YouTubeClient(
            api_key="k", client=fake_client(lambda r: json_response({})), ledger=ledger
        )
        with pytest.raises(DiscoveryError):
            client.get("search", {})
        assert ledger.used == 0

    def test_an_unknown_resource_is_refused_rather_than_guessed(self) -> None:
        client = YouTubeClient(api_key="k", client=fake_client(lambda r: json_response({})))
        with pytest.raises(DiscoveryError, match="unknown YouTube resource"):
            client.get("captions", {})


class TestEndpoints:
    def test_uploads_playlist_is_resolved_once_then_cached(self, tmp_path) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return json_response(
                {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU_ABC"}}}]}
            )

        cache = PlaylistCache(tmp_path / "playlists.json")
        ledger = QuotaLedger(ceiling=100)
        client = YouTubeClient(
            api_key="k", client=fake_client(handler), ledger=ledger, playlist_cache=cache
        )

        assert client.uploads_playlist_id("UC_ABC") == "UU_ABC"
        assert client.uploads_playlist_id("UC_ABC") == "UU_ABC"
        assert len(calls) == 1, "the second lookup must come from the cache"
        assert ledger.used == 1

        # And it survives the process: a fresh cache reads the file.
        reloaded = PlaylistCache(tmp_path / "playlists.json")
        assert reloaded.get("UC_ABC") == "UU_ABC"

    def test_missing_channel_says_what_to_check(self) -> None:
        client = YouTubeClient(
            api_key="k", client=fake_client(lambda r: json_response({"items": []}))
        )
        with pytest.raises(DiscoveryError, match="UC... channel id"):
            client.uploads_playlist_id("@handle")

    def test_video_ids_are_batched_fifty_per_call(self) -> None:
        batches: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            ids = request.url.params["id"].split(",")
            batches.append(len(ids))
            return json_response({"items": [{"id": v} for v in ids]})

        ledger = QuotaLedger(ceiling=100)
        client = YouTubeClient(api_key="k", client=fake_client(handler), ledger=ledger)

        records = client.videos([f"v{i}" for i in range(120)])

        assert batches == [MAX_IDS_PER_CALL, MAX_IDS_PER_CALL, 20]
        assert len(records) == 120
        assert ledger.used == 3, "120 ids must cost 3 units, not 120"

    def test_duplicate_ids_are_collapsed(self) -> None:
        seen: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            ids = request.url.params["id"].split(",")
            seen.append(ids)
            return json_response({"items": [{"id": v} for v in ids]})

        client = YouTubeClient(api_key="k", client=fake_client(handler))
        client.videos(["a", "b", "a", "b", "c"])
        assert seen == [["a", "b", "c"]]

    def test_playlist_paging_stops_at_the_limit(self) -> None:
        pages = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal pages
            pages += 1
            start = (pages - 1) * 50
            return json_response(
                {
                    "items": [{"contentDetails": {"videoId": f"v{start + i}"}} for i in range(50)],
                    "nextPageToken": "more",
                }
            )

        client = YouTubeClient(api_key="k", client=fake_client(handler))
        ids = client.recent_upload_ids("UU_ABC", limit=60)

        assert len(ids) == 60
        assert pages == 2

    def test_a_403_explains_the_likely_cause(self) -> None:
        client = YouTubeClient(
            api_key="k",
            client=fake_client(lambda r: httpx.Response(403, text="quotaExceeded")),
        )
        with pytest.raises(DiscoveryError, match="daily quota"):
            client.uploads_playlist_id("UC_ABC")

    def test_a_failed_call_is_still_charged(self) -> None:
        ledger = QuotaLedger(ceiling=100)
        client = YouTubeClient(
            api_key="k",
            client=fake_client(lambda r: httpx.Response(404, text="nope")),
            ledger=ledger,
        )
        with pytest.raises(DiscoveryError):
            client.get("videos", {"id": "v1"})
        assert ledger.used == 1, "Google charges for a 4xx the same as a 200"


class TestScan:
    def handler_for(self, uploads: dict[str, list[str]], broken: set[str] | None = None):
        broken = broken or set()

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            params = request.url.params

            if path.endswith("/channels"):
                channel_id = params["id"]
                if channel_id in broken:
                    return json_response({"items": []})
                return json_response(
                    {
                        "items": [
                            {
                                "contentDetails": {
                                    "relatedPlaylists": {"uploads": f"UU{channel_id[2:]}"}
                                }
                            }
                        ]
                    }
                )

            if path.endswith("/playlistItems"):
                playlist = params["playlistId"]
                channel_id = "UC" + playlist[2:]
                return json_response(
                    {
                        "items": [
                            {"contentDetails": {"videoId": v}} for v in uploads.get(channel_id, [])
                        ]
                    }
                )

            ids = params["id"].split(",")
            return json_response(
                {
                    "items": [
                        {
                            "id": v,
                            "snippet": {"channelId": "UC_A", "title": v},
                            "contentDetails": {"duration": "PT1H"},
                            "statistics": {"viewCount": "1000"},
                        }
                        for v in ids
                    ]
                }
            )

        return handler

    def test_one_stats_batch_for_the_whole_watchlist(self) -> None:
        uploads = {"UC_A": ["a1", "a2"], "UC_B": ["b1"]}
        ledger = QuotaLedger(ceiling=100)
        client = YouTubeClient(
            api_key="k", client=fake_client(self.handler_for(uploads)), ledger=ledger
        )

        result = client.scan_channels(["UC_A", "UC_B"], per_channel=5)

        assert len(result.records) == 3
        # 2 channels.list + 2 playlistItems.list + 1 videos.list
        assert ledger.used == 5
        assert result.quota["spent_this_run"] == 5

    def test_one_broken_channel_does_not_abort_the_scan(self) -> None:
        uploads = {"UC_A": ["a1"], "UC_B": ["b1"]}
        client = YouTubeClient(
            api_key="k",
            client=fake_client(self.handler_for(uploads, broken={"UC_B"})),
        )

        result = client.scan_channels(["UC_A", "UC_B"])

        assert [r.video_id for r in result.records] == ["a1"]
        assert "UC_B" in result.errors

    def test_quota_exhaustion_stops_the_scan(self) -> None:
        uploads = {"UC_A": ["a1"], "UC_B": ["b1"]}
        client = YouTubeClient(
            api_key="k",
            client=fake_client(self.handler_for(uploads)),
            ledger=QuotaLedger(ceiling=2),
        )
        with pytest.raises(QuotaExceeded):
            client.scan_channels(["UC_A", "UC_B"])

    def test_ledger_file_is_written_as_it_goes(self, tmp_path) -> None:
        path = tmp_path / ".discover" / "quota.json"
        client = YouTubeClient(
            api_key="k",
            client=fake_client(self.handler_for({"UC_A": ["a1"]})),
            ledger=QuotaLedger(path, ceiling=100),
        )
        client.scan_channels(["UC_A"])

        assert json.loads(path.read_text(encoding="utf-8"))["used"] == 3


class TestMissingKey:
    def test_no_api_key_says_where_to_get_one(self) -> None:
        client = YouTubeClient(api_key=None, api_key_env="NOT_SET_ANYWHERE")
        with pytest.raises(DiscoveryError, match="NOT_SET_ANYWHERE is not set"):
            client.get("videos", {"id": "v1"})
