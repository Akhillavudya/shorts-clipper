from pathlib import Path

import pytest
import yaml

from clipper.rights.registry import RightsError, Watchlist
from clipper.rights.watchlist_edit import append_channel


@pytest.mark.parametrize(
    "channels", ["channels: [] # keep this\n", "channels:\n  - id: old\n    rights_status: red\n"]
)
def test_append_preserves_comments_and_other_sections(tmp_path, channels):
    path = tmp_path / "watchlist.yaml"
    original = (
        "# Licence traps: NC / ND\ndefaults:\n  max_per_day: 2 # quota\n"
        + channels
        + "# footer\nother: true\n"
    )
    path.write_text(original, encoding="utf-8")
    entry = {
        "id": "new",
        "name": "雪: #channel",
        "rights_status": "green",
        "note": "Permission: yes\nwith credit",
        "checked": "2026-09-20",
    }
    append_channel(path, entry)
    result = path.read_text(encoding="utf-8")
    assert "# Licence traps: NC / ND" in result
    assert "# quota" in result and "# footer" in result
    assert yaml.safe_load(result)["other"] is True
    assert Watchlist.load(path).entry("new").note == entry["note"]
    before = path.read_bytes()
    with pytest.raises(RightsError, match="already remembered"):
        append_channel(path, entry)
    assert path.read_bytes() == before


def test_real_watchlist_comments_survive_two_appends(tmp_path):
    source = Path("config/watchlist.yaml").read_text(encoding="utf-8")
    path = tmp_path / "watchlist.yaml"
    path.write_text(source, encoding="utf-8")
    for channel in ("one", "two"):
        append_channel(
            path, {"id": channel, "rights_status": "amber", "note": "Email permission saved"}
        )
    result = path.read_text(encoding="utf-8")
    assert [line for line in source.splitlines() if line.startswith("#")] == [
        line for line in result.splitlines() if line.startswith("#")
    ]
    assert len(Watchlist.load(path)) == 2


@pytest.mark.parametrize(
    "text",
    ["channels: [broken", "channels:\n - nope\n", "channels: [{id: old, rights_status: red}]\n"],
)
def test_unsafe_layout_is_refused_without_writing(tmp_path, text):
    path = tmp_path / "watchlist.yaml"
    path.write_text(text, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(RightsError):
        append_channel(path, {"id": "new", "rights_status": "green"})
    assert path.read_bytes() == before
