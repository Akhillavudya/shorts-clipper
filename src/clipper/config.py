"""Loading the four YAML files in ``config/``.

The pipeline reads a lot of tunables and none of them belong in Python. This
module is the single reader, so a stage asks for ``settings.moments`` rather
than opening a file and guessing at defaults.

Every accessor takes a default, because a config file that is missing a key
should degrade to the documented behaviour rather than raising three stages
into a run. A config file that is missing entirely does raise - that is a
broken checkout, not a tuning choice.

``llm/router.py`` keeps its own ``load_settings``; it predates this module and
is deliberately standalone so the LLM layer stays importable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path("config")

SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
CAPTION_STYLE_PATH = CONFIG_DIR / "caption_style.yaml"
SCORING_WEIGHTS_PATH = CONFIG_DIR / "scoring_weights.yaml"
WATCHLIST_PATH = CONFIG_DIR / "watchlist.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read one YAML file into a dict. An empty file is an empty dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=8)
def _load_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    """Cache keyed on mtime, so editing a config file during a run is picked up.

    Caption style in particular gets edited between renders; keying on mtime
    means the next read sees the change without restarting anything.
    """
    del mtime_ns  # part of the cache key only
    return load_yaml(path_str)


def load(path: Path | str) -> dict[str, Any]:
    """Read a config file, reusing the parse unless the file changed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    return _load_cached(str(path), path.stat().st_mtime_ns)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``cfg["a"]["b"]`` as ``get(cfg, "a.b")``, defaulting on any gap.

    Returns ``default`` when a key is missing *or* explicitly null, since a
    commented-out YAML value parses as ``None`` and means "unset", not "None".
    """
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


@dataclass(frozen=True)
class Settings:
    """``config/settings.yaml``, with typed access to the blocks stages use."""

    data: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path | str = SETTINGS_PATH) -> Settings:
        return cls(data=load(path), path=Path(path))

    def get(self, dotted: str, default: Any = None) -> Any:
        return get(self.data, dotted, default)

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        return value if isinstance(value, dict) else {}

    # -- the blocks each stage reads ------------------------------------

    @property
    def jobs_root(self) -> Path:
        return Path(self.get("paths.jobs_root", "jobs"))

    @property
    def assets_root(self) -> Path:
        return Path(self.get("paths.assets_root", "assets"))

    @property
    def fonts_dir(self) -> Path:
        return Path(self.get("paths.fonts_dir", self.assets_root / "fonts"))

    @property
    def music_dir(self) -> Path:
        return Path(self.get("paths.music_dir", self.assets_root / "music"))

    @property
    def transcription(self) -> dict[str, Any]:
        return self.section("transcription")

    @property
    def moments(self) -> dict[str, Any]:
        return self.section("moments")

    @property
    def edit(self) -> dict[str, Any]:
        return self.section("edit")

    @property
    def rights(self) -> dict[str, Any]:
        return self.section("rights")

    @property
    def discovery(self) -> dict[str, Any]:
        return self.section("discovery")


def load_caption_style(path: Path | str = CAPTION_STYLE_PATH) -> dict[str, Any]:
    """``config/caption_style.yaml``. Read fresh on every render, by design.

    Caption style is the most-iterated file in the project (CLAUDE.md hard rule
    6), so no value from it is ever cached into a Python constant.
    """
    return load(path)


def load_scoring_weights(path: Path | str = SCORING_WEIGHTS_PATH) -> dict[str, Any]:
    """``config/scoring_weights.yaml``, retuned by the calibration loop."""
    return load(path)


__all__ = [
    "CAPTION_STYLE_PATH",
    "CONFIG_DIR",
    "SCORING_WEIGHTS_PATH",
    "SETTINGS_PATH",
    "WATCHLIST_PATH",
    "Settings",
    "get",
    "load",
    "load_caption_style",
    "load_scoring_weights",
    "load_yaml",
]
