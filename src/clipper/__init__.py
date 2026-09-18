"""shorts-clipper — long-form video in, vertical Shorts out.

A CLI with named stages and one JSON state file per job (see FULL_PLAN.md §1).
No agent framework: each stage reads ``jobs/<video_id>/state.json``, does its
work, and writes it back, so any stage can be rerun in isolation.
"""

__version__ = "0.1.0"
