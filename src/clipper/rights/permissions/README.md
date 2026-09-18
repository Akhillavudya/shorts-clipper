# Permission records

One file per creator who has granted clipping permission. This directory is the
evidence behind every `amber` entry in `config/watchlist.yaml`.

Save, per creator:

- the full email thread (`.eml` or `.md`), including their reply granting
  permission
- or a screenshot / archived copy of their public clipping policy
- the date, and any stated limits: length cap, attribution wording, whether it
  covers Shorts, which channels it applies to

Name files by the watchlist key, e.g. `UCxxxxxxxxxxxxxxxxxxxxxx.md`.

These are committed deliberately. Reused-content enforcement is applied at the
channel level, so if a claim is ever filed this directory is the answer. Redact
personal contact details that are not needed as evidence.
