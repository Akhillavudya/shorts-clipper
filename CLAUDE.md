# CLAUDE.md

Guidance for working in this repo. `FULL_PLAN.md` is the research and design
document; it explains *why* the decisions below were made. This file is the
short version plus the rules that must not be broken.

## What this is

Long-form video in, vertical Shorts out. Target 1-3 Shorts/day, semi-automated.
A Python CLI with named stages and one JSON state file per job. No agent
framework, and none is wanted.

## Commands

```bash
uv sync                          # create/refresh .venv from pyproject.toml
uv run clipper --help            # the CLI
uv run pytest                    # tests (no network, no API keys needed)
uv run ruff check . --fix        # lint
uv run ruff format .             # format
```

Requires Python 3.11+. `ffmpeg` must be on PATH for any stage past `ingest`.
It is installed at `%LOCALAPPDATA%\Programs\ffmpeg\bin` (Gyan full build
9.0.1) and that directory is on the user `Path`. It was unpacked there by hand
because `winget install Gyan.FFmpeg` fails on this machine with
`copy_file: Access is denied` writing to its own package directory.

## Architecture

```
discover -> ingest -> transcribe -> moments -> render -> publish -> upload -> calibrate
```

The four middle stages are the per-job pipeline: each reads
`jobs/<video_id>/state.json`, does its work, writes it back. Stages are
independently rerunnable. If a stage fails you rerun that stage, not the
pipeline.

The others are not per-job stages and do not appear in `JOB_STAGES`:

- `discover` runs *before* a job exists. It writes a dated report under
  `jobs/.discover/`, and you pick from it by hand.
- `review` serves a local Flask page between `moments` and `render`. It writes
  `approved`, `review_note` and `reviewed_at` on a clip and runs no stage.
- `publish` writes `clips/clip_NN/upload.json` for a rendered clip and records
  the live `youtube_video_id` on the clip state.
- `upload` sends one clip through `videos.insert` and records the id it gets
  back. One clip per invocation, never a set: it is the only irreversible
  stage. It records itself on the *clip* as an `upload` stage, which is why
  `upload` is in neither `JOB_STAGES` nor `CLIP_STAGES` - `ClipState.stage()`
  creates it on demand.
- `auth` is not a stage at all. It runs the one-time Google consent flow and
  prints the refresh token `upload` and `calibrate` both need.
- `calibrate` reads across *every* job at once, because the sample is the point.

```
jobs/<video_id>/
  state.json          # single source of truth
  source.mp4  audio.wav
  transcript.json     # word-level, diarized
  candidates.json     # LLM pass 1
  scored.json         # LLM pass 2 - keep forever, the calibration loop needs it
  llm.log             # JSONL: every model call
  .llm_cache/         # keyed by hash(prompt + model + schema)
  clips/clip_01/
    cut.mp4  reframed.mp4  crop_path.cmd  captions.ass  final.mp4
    meta.json  upload.json
    .upload_session.json     # only while an upload is in flight or failed

jobs/.discover/              # not a job; discovery run state, gitignored
  quota.json                 # units spent, keyed on the UTC date
  uploads_playlists.json     # channel id -> uploads playlist id, forever
  discover-YYYY-MM-DD.json   # the dated shortlist
  llm.log  .llm_cache/
```

Use `clipper.state.JobStore` for all state access. The `store.stage(...)`
context manager records status and timings and flushes on both exit paths:

```python
store = JobStore("jobs")
state = store.load_or_create(video_id, source_url=url)

if store.should_run(state.stage("transcribe"), force=force):
    with store.stage(state, "transcribe") as st:
        st.detail["backend"] = "gemini"
        ...
```

A stage left as `running` means the process died mid-stage. That is deliberate -
it is how a crash is told apart from a stage that never started.

## Hard rules

These are load-bearing. Breaking one is a real bug, not a style preference.

1. **Never bypass the rights gate.** `rights_status` is a first-class field and
   `render`, `publish` and `upload` all refuse `red` and `unknown` without an
   explicit override. Reused-content enforcement is at the **channel** level:
   one bad batch suspends monetization for everything. Do not "temporarily"
   default it to green.
2. **The LLM never emits cut points.** It returns word *indices*. Timestamps
   come from `moments/snap.py`, deterministically. Do not let a model return
   seconds.
3. **Never call `search.list` on a schedule.** 100 quota units against a
   10,000/day budget. Discovery is watchlist-driven: `playlistItems.list` costs
   1 unit.
4. **Ask the router for a tier, never a model.** `router.complete("recall", ...)`.
   Model IDs live only in `config/settings.yaml`. No model string belongs in
   `src/`.
5. **Never stream-copy when cutting.** `-c copy` snaps to keyframes and will be
   seconds off, silently destroying boundaries that were computed to the
   millisecond.
6. **Caption style lives in `config/caption_style.yaml`,** not in Python. It is
   iterated on constantly.
7. **Word-level timestamps are non-negotiable.** They drive both cut boundaries
   and karaoke captions. A transcription backend without them is not usable.
8. **Keep `scored.json` forever.** Raw rubric scores are what the calibration
   loop correlates against retention. Archiving only the weighted total throws
   away the only real edge this project has.
9. **Every track in `assets/music/` needs its licence recorded in that
   directory's `README.md`.** The rights gate covers the footage and knows
   nothing about the audio laid over it, and a Content ID claim on the music
   takes the revenue for the whole Short. "Free to use" on a music site
   usually means free with attribution, and not free on a monetized upload.
10. **An upload is private, single, and asked for.** `clipper upload` takes one
    `--clip`, lands the video `private` unless told otherwise, and refuses a
    clip that already carries a `youtube_video_id` without `--force`. Nothing
    uploads as part of `clipper run`. A pipeline that publishes straight to
    public removes the last place a bad clip can be caught, and a duplicate
    upload is a duplicate video on a channel judged for inauthentic content.

## The LLM layer

`FULL_PLAN.md` section 2 specified Claude Haiku 4.5 and Sonnet 5. We run on free
tiers instead, so those choices moved into config. Two logical tiers:

| Tier     | Task                          | Configured as              |
|----------|-------------------------------|----------------------------|
| `recall` | pass 1, whole transcript      | Gemini Flash-Lite          |
| `rerank` | pass 2, ~20 candidates        | Gemini Flash               |
| fallback | when the primary is failing   | Groq Llama                 |

```python
from clipper.llm import Router

router = Router(job_dir=store.job_dir(video_id))
data = router.complete("recall", prompt, system=SYSTEM, json_schema=SCHEMA)
```

`complete()` returns a parsed dict. JSON mode is required, not optional -
every call in this pipeline returns structured output.

Free-tier survival, in `llm/`:

- `limiter.py` - token buckets for RPM/TPM, a UTC-day counter for RPD. Aims at
  90% of the configured ceiling, because a 429 costs more than the wait.
- `router.py` - exponential backoff with jitter, honouring `Retry-After`;
  failover to the fallback provider after `max_attempts`; `AuthError` fails over
  immediately rather than burning retries.
- `cache.py` - responses cached under `jobs/<id>/.llm_cache/`. **Re-running a
  stage during development must not respend quota.** The key covers the schema
  and system prompt too, so changing the requested shape misses rather than
  returning a stale shape.
- Every call is appended to `jobs/<id>/llm.log` as JSONL:
  `jq -s 'map(.tokens_in + .tokens_out) | add' jobs/<id>/llm.log`

Adding a provider: implement the `Provider` protocol in `llm/base.py`, map the
SDK's failures onto the error taxonomy (`RateLimitError`, `TransientError`,
`AuthError`, `ResponseFormatError`), and register it in `_build_provider`.
The router's retry logic keys off those types and nothing else.

### Rate limits are unverified

Neither Google nor Groq publishes per-model free-tier numbers in their docs any
more - both point at account-specific dashboards. The `rpm`/`tpm`/`rpd` values
in `settings.yaml` are conservative placeholders marked `TODO`. Confirm them at
<https://aistudio.google.com/rate-limit> and
<https://console.groq.com/settings/limits> and replace them. They are set low on
purpose: the limiter under-shooting costs latency, over-shooting costs 429s.

Model IDs in `settings.yaml` were taken from the provider docs, not from
memory. Do not "correct" them from recall - re-check the docs.

## Config

| File | Holds |
|---|---|
| `config/settings.yaml` | paths, LLM tiers, provider limits, thresholds |
| `config/watchlist.yaml` | channels + `rights_status` + permission notes |
| `config/caption_style.yaml` | every caption visual, ASS colours are `&HAABBGGRR&` |
| `assets/music/README.md` | the licence of every music bed. Not optional. |
| `assets/models/README.md` | where each detection model came from, and its licence |
| `config/scoring_weights.yaml` | rubric weights, retuned by the calibration loop |
| `settings.yaml` `publish:` block | privacy, category, tag behaviour for `upload.json` |
| `settings.yaml` `publish.upload:` block | chunk size, retries, and the upload half of the shared Data API quota |
| `.env` `YOUTUBE_CLIENT_ID`/`_SECRET`/`_REFRESH_TOKEN` | OAuth for `upload` and `calibrate`. `clipper auth` mints the refresh token. |

`config/watchlist.yaml` is the highest-leverage file in the project. Evidence
for `amber` entries goes in `src/clipper/rights/permissions/`.

## Testing

- **No test may hit a real API.** Providers are faked; see `tests/test_router.py`.
- Time is injected, never slept. `tests/conftest.py` has `FakeClock` and
  `FakeWallClock`; `clock.sleep()` advances both, because a real sleep does.
- Run `uv run pytest` before claiming anything works.

## Current status

**Every phase of the build order is implemented.** `ingest -> transcribe ->
moments -> render` runs end to end from the CLI, and so do `discover`,
`review`, `publish`, `upload` and `calibrate`. 619 tests, none of which touch a
network or need ffmpeg.

Implemented and tested: `state.py`, `llm/`, `media.py`, `config.py`,
`prompts.py`, `pipeline.py`, `google_auth.py`, `ingest/`, `transcribe/`,
`moments/`, `edit/`, `discover/`, `review/`, `publish/`, `calibrate/`,
`rights/registry.py`, and the CLI (`discover`, `ingest`, `transcribe`,
`moments`, `render`, `run`, `review`, `publish`, `upload`, `auth`,
`calibrate`, plus `status` and `doctor`).

Eight things worth knowing about the newer stages:

- **`review` is the only stage with a human in it, and it runs nothing.**
  `clipper review [job]` serves `src/clipper/review/` on `127.0.0.1:5151`:
  scored candidates with their rubric, their words, an inline preview and
  keep/cut buttons. It writes three fields on a clip and never touches a stage
  record, because a multi-minute ffmpeg run behind a browser tab that can be
  closed halfway is a half-written mp4. `clipper render <job> --approved-only`
  is the follow-up; without the flag, render still takes everything that was
  not explicitly rejected. Two consequences of reviewing *before* rendering:
  the preview is normally `source.mp4` seeked to the cut points (the page stops
  at the end; `final.mp4` takes over once it exists), and the media route
  streams out of the jobs root - which is why the bind address is loopback and
  `presenter.MEDIA_SUFFIXES` is an allowlist rather than a blocklist.

- **`render` can lay a music bed** (`--music <track>`, `--no-music`, or
  `edit.music_enabled`). It is a separate clip stage writing `mixed_music.wav`
  next to `mixed.wav`, so the speech-only mix survives for comparison and the
  bed can be re-cut without re-running loudnorm. The bed is *normalized* to
  `edit.music_bed_lufs`, not gained by it - a gain means something different
  for every track - and `music_duck` sidechains it under the voice, keyed on
  the speech itself. Speech is normalized before the bed is added, never
  after: loudnorm over speech+music lets a louder bed buy itself volume by
  pushing the voice down.

- **The rights gate now fires three times.** `render` gates the file; `publish`
  gates the title, description and credit that would accompany it; `upload`
  gates the irreversible part. All three take the same `--allow-red` override
  and all three record it on the job state.

- **`upload` is the one irreversible stage, and it is built like it.** One
  `--clip` per invocation, never a set. `private` unless `--privacy` says
  otherwise. A clip that already has a `youtube_video_id` is refused without
  `--force`. It is not part of `clipper run`.

  The transfer is the resumable protocol, not a single POST: the server is
  authoritative about the offset, so after any failure `publish/upload.py`
  *asks* where it got to rather than assuming its last chunk was lost. The
  session URI is saved as `clips/clip_NN/.upload_session.json` and keyed on a
  SHA-256 of `final.mp4` - a re-render invalidates it, because resuming a
  session opened for the old bytes splices two encodes into something that
  decodes as neither. A dead session (404/410) restarts exactly once.

  One `videos.insert` costs **1600** of the project's 10,000 Data API units per
  day, and that is the same budget `discover` spends one unit at a time. Both
  charge `jobs/.discover/quota.json`; `publish.upload.daily_quota_ceiling` is
  the upload-side guard on the shared total. The charge happens *before* the
  session opens, because Google bills the attempt.

  OAuth, not an API key: `clipper auth` runs the consent flow once and prints a
  refresh token. `clipper publish --published <id>` still exists for uploads
  done by hand - `calibrate` only needs the id, not the mechanism.
- **`discover` drops red and unknown channels before it spends a quota unit on
  them.** They can never be rendered, so shortlisting one is an invitation to
  override the gate. `--include-blocked` lists them anyway, labelled.
- **`ingest` caps the download at `ingest.max_source_height` (1440p).**
  `bestvideo[height>=1080]` takes the *best* match, so a channel that uploads
  in 4K costs a ~1 GB download per episode to produce a 40-second Short. 2160p
  is the only height that never upscales through the 9:16 crop; 1440p upscales
  1.33x against 1080p's 1.78x, for 45% of the bytes.

- **`calibrate` is a dry run by default**, and even with `--apply` it refuses to
  write weights that do not predict better than the ones in force. It rewrites
  only the numbers in `scoring_weights.yaml`, never the comments.

- **`render` tracks the active speaker** (`edit.reframe: track`, or
  `--centre-crop` for one render). `edit/track.py` samples faces with MediaPipe
  BlazeFace every `edit.detect_interval_s`, picks the speaker by size with a
  centre bias and hysteresis, smooths the path with an EMA and a deadzone, and
  resets hard at PySceneDetect shot boundaries. The path reaches ffmpeg as a
  `sendcmd` script (`clips/clip_NN/crop_path.cmd`) driving a named `crop`
  filter, so the encode is unchanged and only the geometry moves. Every value
  in that script is an integer, never an expression: `crop`'s `eval` option is
  gone in ffmpeg 8 and defaulted to `init` in 7, so an expression in `t` would
  be evaluated once on one build and per frame on another. Tracking degrades
  to a centre crop rather than failing, and the reason lands on the clip's
  `reframe` stage as `track_note`.

Deliberately deferred, not missing:

- **Reframing tracks one speaker, not two.** The plan's two-person treatments
  - a stacked split with both faces, or cutting between them on diarization
  turns - are not built, and neither is using the diarization label plus mouth
  movement to decide who is speaking. Size, position and hysteresis is what
  picks the face. In a two-shot where both faces are the same size and neither
  is talking, that is a guess.
- **`transcribe/whisperx_local.py` needs `uv pip install whisperx`.** It is
  not a declared dependency - it pulls a multi-gigabyte CUDA stack, and the
  default path is the cloud backend.
- **Every network call in the project is unexercised outside the fakes.** No
  API key and no OAuth client is set, so no LLM call, no YouTube Data API call,
  no Podcast Index call, no Analytics call and no `videos.insert` has run
  against a real endpoint - and no consent flow has been run at all, so
  `clipper auth` is the least exercised code in the repo. `clipper doctor`
  reports both halves. The HTTP clients are all faked with
  `httpx.MockTransport` in tests, which proves the request shape and the error
  handling, not the endpoint's actual behaviour.

  The upload is the one where that gap was narrowed by hand: `pipeline.upload`
  was driven against the real 16.7 MB `jobs/NUBY4-kogOg` `final.mp4` through a
  mock transport that verifies every byte it is handed. Two 8 MiB chunks, an
  injected 503 in the middle, the offset re-queried, and what the fake server
  ended up holding was byte-identical to the file. That proves the chunking and
  the resume logic on real media; it proves nothing about what Google does.
- **ffmpeg is the exception, and Phase 2 was checked on real media.** probe,
  video_info, duration_s, detect_silences, cut, reframe, loudnorm measure and
  two-pass normalize, ASS write and burn-in encode all succeeded on a
  synthesised 1920x1080 source first. Since then the whole render has run on a
  real 2560x1440 podcast episode (`jobs/NUBY4-kogOg`), which is what Phase 2
  needed:
  - `moments` found 1071 silence spans and recorded no `snap_note`, so
    boundaries were snapped for real rather than falling back to sentence
    extension.
  - `final.mp4` measures -14.5 LUFS integrated / -1.3 dBTP against the -14.0
    target, so the two-pass loudnorm holds on real speech.
  - The tracked crop was confirmed by eye on extracted frames: the same frame
    that centre-crops to half a face at the edge is centred when tracked.

  What that still does not cover is awkward media - variable frame rate,
  rotation metadata, missing or multi-track audio.

Two things worth knowing before touching a stage:

- `moments` degrades rather than fails without ffmpeg: silence detection
  returns nothing and snapping falls back to sentence extension. That is
  intentional (it is otherwise a pure-text stage) and is recorded in
  `state.json` as `snap_note`.
- Word timings from the Gemini backend are model-reported, not forced
  alignment. Good enough for cut boundaries, marginal for karaoke. An
  out-of-sync caption is this, and whisperx is the fix.

Build order is `FULL_PLAN.md` section 12, and nothing is left of it - Phase 5
closed with `videos.insert` and the consent flow behind it. What the project
needs next is not code: a filled-in `config/watchlist.yaml`, an OAuth client in
`.env`, and then real uploads, so `calibrate` has a sample to correlate
against.

Three dependency notes:

- `flask` is declared for `review/` alone, and imported inside the `review`
  command rather than at CLI import time. `clipper moments` should not pay to
  import a web framework.
- `anthropic` is still a declared dependency because it was requested, but
  nothing imports it. Remove it if the plan's Claude tiers are not coming back.
- `google-api-python-client` is likewise declared and unused. `discover/` and
  `calibrate/` talk to Google over plain `httpx`, matching `llm/groq.py` and
  `llm/ollama.py`, as do `publish/upload.py` and `google_auth.py` - six
  read-only REST endpoints, one write, and the OAuth token endpoint. A plain
  client is something a test can fake with `httpx.MockTransport` without first
  fetching a discovery document over the network. Remove the dependency,
  or use it - but do not add a second way of calling the same APIs.

Three things the newer stages need before they do anything useful:

1. **`config/watchlist.yaml` is empty.** `discover` scans nothing and every
   source resolves to `unknown`, which `render`, `publish` and `upload` all
   refuse. This is the highest-leverage file in the project and it is
   maintained by hand.
2. **`calibrate` needs `youtube_video_id` on published clips.** `clipper upload`
   records it; for an upload done by hand, `clipper publish <job> --clip N
   --published <id>` does. Without it there is nothing to join retention
   against, and `scored.json` - the thing hard rule 8 says to keep forever -
   has no counterpart to correlate with.
3. **`upload` and `calibrate` need an OAuth client.** Type *Desktop app*, from
   the Google Cloud console, with the YouTube Data API v3 and YouTube Analytics
   API enabled. Put the id and secret in `.env`, run `clipper auth` once, paste
   the refresh token back. `clipper doctor` reports whether all three are set.
