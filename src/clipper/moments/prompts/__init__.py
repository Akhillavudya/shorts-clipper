"""Prompt templates for the recall and rerank passes.

Kept out of code so prompts can be iterated on and diffed independently. The
loader itself lives in ``clipper.prompts``; discovery keeps its own directory
with the same shape.
"""

from __future__ import annotations

from pathlib import Path

from clipper.prompts import MissingPrompt, PromptSet

PROMPT_DIR = Path(__file__).resolve().parent

_PROMPTS = PromptSet(PROMPT_DIR)

prompt_path = _PROMPTS.path
load = _PROMPTS.load
render = _PROMPTS.render
available = _PROMPTS.available

__all__ = ["PROMPT_DIR", "MissingPrompt", "available", "load", "prompt_path", "render"]
