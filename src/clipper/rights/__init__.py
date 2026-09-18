"""Rights gate (FULL_PLAN.md §0).

``rights_status`` is a first-class field on every video record because
reused-content enforcement is applied at the *channel* level, not per video.
The renderer refuses to export anything marked ``red`` without an explicit
override.
"""
