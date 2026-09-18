"""Stage 4 — find and score candidate moments (FULL_PLAN.md §5).

Two LLM passes (recall, then precision rerank). The LLM never emits final cut
points; boundary snapping is deterministic.
"""
