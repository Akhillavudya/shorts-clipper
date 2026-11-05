"""Append one channel without rewriting the watchlist's hand-written comments."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import yaml

from clipper.rights.registry import ChannelEntry, RightsError


def append_channel(path: Path, entry: dict) -> None:
    ChannelEntry.from_dict(entry)
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
        document = yaml.compose(text)
    except yaml.YAMLError as exc:
        raise RightsError(
            "The watchlist contains invalid YAML; fix it before remembering a channel."
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("channels"), list):
        raise RightsError("watchlist must contain a channels list")
    for channel in data["channels"]:
        if not isinstance(channel, dict):
            raise RightsError("Every watchlist channel must be a mapping")
        ChannelEntry.from_dict(channel)
    if any(c.get("id") == entry["id"] for c in data["channels"]):
        raise RightsError("This channel is already remembered; edit its watchlist entry directly.")
    # Locate the actual mapping node, not a commented example or nested key.
    nodes = [(k, v) for k, v in document.value if k.value == "channels"]
    if len(nodes) != 1:
        raise RightsError("watchlist must have exactly one channels key")
    key, value = nodes[0]
    lines = text.splitlines(keepends=True)
    line = lines[key.start_mark.line]
    if value.flow_style:
        if data["channels"] or not re.match(r"^channels:\s*\[\]\s*(?:#.*)?$", line.strip()):
            raise RightsError("Use a block list for channels before remembering another channel.")
        lines[key.start_mark.line] = line.replace("[]", "", 1)
        position = key.start_mark.line + 1
    else:
        position = value.end_mark.line
    block = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True)
    block = "".join("  " + item + "\n" for item in block.splitlines())
    prefix = "".join(lines[:position])
    result = prefix + ("\n" if prefix and not prefix.endswith("\n") else "") + block
    result += "".join(lines[position:])
    expected = {**data, "channels": [*data["channels"], entry]}
    if yaml.safe_load(result) != expected:
        raise RightsError("Could not safely append to this watchlist layout; edit it directly.")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".watchlist.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as out:
            out.write(result)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
