# WEB_UI_PLAN.md

## Implementation checkpoint — 2026-09-20

Phases **0-6 are implemented**. Phase 4 adds the React jobs list and job detail,
lazy clip previews with source-span seeking, rubric bars, keep/cut/undo and
review notes, keyboard shortcuts, and individual MP4 / streamed ZIP downloads.
Queue rows now open the React job page. The Python package's built bundle is
updated alongside the frontend source. The legacy review UI remains for phase 8.

The dashboard styling pass is complete: responsive sidebar navigation, white
and near-black themes with Light / Dark / Auto controls, queue summary cards,
a clearer creation form, searchable and filterable clip library, and consistent
review cards. Theme preference is applied before first paint. Frontend build
(including TypeScript compilation) and ESLint pass; the packaged bundle is rebuilt.

**Phase 5 complete:** rights classification with required reasons, conservative
watchlist appends, blocked-job resume, confirmed overrides, stage retry, and
render-only controls for kept/all/individual clips. In-process queue locking
serializes worker startup with API mutations; no cross-process lock is used.
Completed ingest is reused on retries without resolving the source again.

**Phase 6 implemented:** Settings & storage page, read-only operational config,
per-job disk accounting and source deletion, bounded raw-log tails, rotating JSON
server logs, configurable disk floor, queue ownership via an OS-held runner.lock,
port-conflict errors and explicit remote-bind acknowledgement. Startup binds before
workers begin; shutdown marks work interrupted and restart can stop surviving workers
under the exclusive claim. Deletion resets ingest and preserves local originals,
clips and scored.json. Retry restores deleted downloaded sources before render.

**Next: phase 7 (optional download progress), then phase 8 (Jinja retirement).**
All 934 Python tests, Ruff, frontend build and ESLint pass; real mid-render shutdown/restart and browser interaction
verification remain manual.
Browser interaction verification still needs a manual pass; the implementation
session had no connected browser. Automated API, media and build checks are
recorded in `CLAUDE.md`.

The implementation plan for the local web frontend. `FULL_PLAN.md` is the
research document for the pipeline; this is the same kind of document for the
UI that drives it. Written 2026-09-17, to be implemented from the next session.

---

## Context

`clipper` today is eight CLI stages driven by hand: `ingest <url>`, `transcribe
<id>`, `moments <id>`, `render <id>`. Every stage blocks a terminal for minutes
to tens of minutes, and you have to come back and type the next one. The
pipeline is complete and well tested (619 tests), but it is unusable as a daily
tool because *you* are the scheduler.

The goal: open a local web page, paste a YouTube URL, walk away, come back to
finished vertical Shorts you can preview and download from the page. No CLI
typing.

Four decisions, already made:

- **Automation stops after `render`.** Clips are rendered and downloadable.
  Uploading stays a deliberate, separate action (hard rule 10).
- **Local only**, bound to `127.0.0.1` on this Windows PC. No auth, no HTTPS,
  no hosting.
- **React + Vite SPA** (TypeScript, Tailwind, shadcn/ui) over a Flask JSON API.
  The built bundle is committed so `clipper ui` needs Python only.
- **One job at a time**, in a visible queue.

### The three facts that shape everything below

1. **No background execution, no cancellation, no locking.** Every pipeline
   function is synchronous. `media.run` uses
   `subprocess.run(capture_output=True)`, so nothing streams and there is no
   `Popen` handle to terminate. `JobStore.save()` is atomic but there is no
   read-modify-write protection anywhere, and `pipeline.moments` does
   `state.clips = []` before rebuilding.
2. **A pasted URL has no job id yet.** `pipeline.ingest` calls
   `download.resolve()` — a network metadata call — to learn the `video_id`
   that names the job directory. A queue entry must exist before the job does.
3. **`config/watchlist.yaml` has `channels: []`.** `registry.resolve()` returns
   `unknown` for any channel not publishing under CC BY, and `render` refuses
   it. Without a rights answer in the flow, "paste any link" fails at the last
   and most expensive stage.

### Shape of the change

| Area | Path | Nature |
|---|---|---|
| Server layer | `src/clipper/server/` | new Python package |
| Worker entrypoint | `src/clipper/server/worker.py` | new, spawned as a child process |
| SPA source | `frontend/` | new, repo root |
| Built bundle | `src/clipper/webui/dist/` | new package, committed |
| Read model | `src/clipper/review/presenter.py` | **kept as-is**, becomes the API's read model |
| Media route | `review/app.py` `/media/...` | **kept verbatim**, Range + traversal tests untouched |

`pipeline.py`, `state.py` and `edit/` are untouched until phase 7. The web layer
is additive on purpose: 619 tests pass today.

---

## 1. The job runner

**One child process per job, spawned and reaped by a single supervisor thread
inside the Flask process. The queue lives on disk.**

```
Flask process (clipper ui)
├── Werkzeug threaded server        — API, always responsive
└── supervisor thread (JobRunner)   — tick() loop; spawns and reaps, never works
        └── Popen: python -m clipper.server.worker --entry <id> ...
                └── pipeline.run_all(auto=True)
                        └── ffmpeg / yt-dlp children
```

### Why not a thread in the Flask process

- **Crash isolation.** `edit/track.py` runs MediaPipe and OpenCV in-process. An
  access violation there takes the interpreter down, and with it the UI you are
  watching. The entire point of this feature is "walk away and come back";
  coming back to a dead server is the failure that matters most.
- **Cancellation.** Threads cannot be killed. `on_progress` returns `None` and
  its value is discarded, so a callback cannot signal stop, and `media.run`
  exposes no process handle. Killing the tree is the *only* way to stop a job.
- **Contention.** `detect_faces` is a pure-Python frame loop and PySceneDetect
  does a second full decode. Both hold the GIL and would make the UI crawl.

A persistent worker pool loses for a different reason: it is the per-job design
plus lifetime management (health checks, restarts, drain on shutdown) with
nothing to amortise. Jobs are minutes long; interpreter startup is milliseconds.
A fresh process per job also gets a clean `config._load_cached` LRU, so an
edited YAML is always picked up.

### Cancellation protocol

Exit codes are the contract: `0` done · `1` failed · `3` **blocked by rights** ·
`130` cancelled. `3` is reserved so a rights block is never reported as a
generic failure.

Cancel is graceful-first: the supervisor writes
`jobs/_queue/<entry_id>.cancel`, the worker checks for it between stages and
between clips and exits `130` cleanly. A cancel during a three-minute
`-preset slow` encode cannot be caught there, so `kill_tree` fires after 10 s.
That leaves the stage marked `running` in `state.json` — which `should_run`
already treats as needing a rerun. **The existing state machine handles a hard
kill correctly; nothing in `state.py` needs to change.**

### Single writer per file, across processes

| File | Sole writer |
|---|---|
| `jobs/_queue/queue.json` | supervisor (Flask process) |
| `jobs/_queue/events/<entry_id>.jsonl` | the worker child |
| `jobs/_queue/<entry_id>.cancel` | supervisor (sentinel) |
| `jobs/<video_id>/state.json` | the worker child, while running |

No cross-process lock is needed because no file has two writers. The worker
reports `video_id` back by emitting a `resolved` event; the supervisor copies it
into `queue.json`.

`_queue` is invisible to the rest of the system for free: `JobStore.list_jobs()`
only returns directories containing `state.json`, and `_job_dir()` requires
`store.exists(job_id)`.

### Modules

```python
# src/clipper/server/queue.py — Flask-free, no threads
@dataclass
class QueueEntry:
    entry_id: str            # uuid4 hex, assigned at submit — identity before video_id exists
    source: str
    status: QueueStatus      # queued running done failed cancelled interrupted blocked
    options: RunOptions      # rights, music, no_music, centre_crop, force, render_only, clip_index
    video_id: str | None = None
    rights_status: str | None = None
    pid: int | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_alive: datetime | None = None
    error: str | None = None
    exit_code: int | None = None

class QueueStore:
    def load(self) -> list[QueueEntry]
    def save(self, entries) -> None                      # mkstemp + os.replace, as JobStore.save
    def enqueue(self, source, options) -> QueueEntry
    def get(self, entry_id) -> QueueEntry | None
    def update(self, entry_id, **fields) -> QueueEntry
    def next_queued(self) -> QueueEntry | None
    def running(self) -> QueueEntry | None
    def running_video_id(self) -> str | None             # §3 depends on this
    def reclaim(self, *, is_alive) -> list[QueueEntry]   # running + dead pid -> interrupted
    def cancel_path(self, entry_id) -> Path
```

```python
# src/clipper/server/runner.py — the supervisor
class ProcessHandle(Protocol):
    pid: int
    def poll(self) -> int | None: ...
    def kill_tree(self, *, timeout_s: float = 10.0) -> None: ...

class JobRunner:
    def __init__(self, *, queue, store, events, spawn=None, clock=None,
                 min_free_bytes: int = 10 * 1024**3) -> None
    def tick(self) -> None       # ONE unit of work — the testability hinge
    def start(self) -> None      # thread: while not stopped: tick(); sleep(0.25)
    def cancel(self, entry_id) -> QueueEntry
    def shutdown(self, *, timeout_s: float = 10.0) -> None

def spawn_worker(entry, *, repo_root, config_path, jobs_root, log_path) -> ProcessHandle
def kill_tree(pid, *, timeout_s: float = 10.0) -> None
def is_alive(pid) -> bool
```

`tick()` is the decision that makes the runner testable at all. The thread is a
trivial `while` loop; every test calls `tick()` directly with a `FakeClock` and
a `FakeProcess`. No sleeping, no real processes.

```python
# src/clipper/server/worker.py — the child entrypoint
def main(argv: list[str] | None = None) -> int
def run_entry(entry_id, *, config_path, jobs_root, source, options, log) -> int
```

**Restart recovery.** On startup, `reclaim()` marks any entry left `running`
with a dead pid as `interrupted` — the server and its worker died together.
Every stage is idempotent (`should_run` skips completed stages; output files are
guarded by `or not X.is_file()`), so resuming is safe and the UI offers one
button to do it.

---

## 2. Progress: persisted event log, polled

Per-job append-only JSONL at `jobs/_queue/events/<entry_id>.jsonl`:

```json
{"seq":12,"ts":"2026-09-17T14:03:11.442Z","kind":"progress","stage":"clip_01/reframe","detail":"tracked crop to 1080x1920"}
```

`kind` ∈ `submitted · started · resolved · rights · progress · stage · blocked ·
error · finished · cancelled`.

**Why persisted, not in-memory:** the producer is a different process, so
in-memory in the child is unreachable. It also survives a browser refresh and a
server restart mid-job — you are going to walk away, so history must outlive the
tab. And it is the only durable record of the minutes in between: `state.json`
flips `running → done` with nothing between, and `st.detail` mutations are only
flushed when the stage context manager exits.

**Why polling, not SSE:** the repo has zero SSE/websocket code and this adds
none. One browser tab on localhost; 1 req/s is free. Werkzeug's threaded dev
server pins a thread per open SSE stream and browsers auto-reconnect — a
thread-leak class of bug for no visible gain. Polling asserts cleanly with
`app.test_client()`, no async and no generators, which is the difference between
§8 being easy and being a project. And it degrades correctly: shut the laptop
for an hour, one `?since=` request catches up. **Write this reasoning into
`server/events.py`'s docstring so nobody "upgrades" it later.**

```python
# src/clipper/server/events.py
@dataclass(frozen=True)
class Event: seq: int; ts: datetime; kind: str; stage: str; detail: str

class EventLog:
    def append(self, entry_id, kind, stage="", detail="") -> Event
    def read(self, entry_id, *, since=0, limit=500) -> list[Event]
    def tail(self, entry_id) -> Event | None
    def progress_callback(self, entry_id) -> Callable[[str, str], None]
```

`progress_callback` returns exactly the `Progress = Callable[[str, str], None]`
shape `pipeline` already expects, so the worker passes
`on_progress=log.progress_callback(entry_id)` into `run_all` and **nothing in
`pipeline.py` changes**. It throttles duplicate `(stage, detail)` pairs and caps
at ~2000 events per job. `read()` skips a truncated final line rather than
raising — a killed writer leaves half a line, and a UI that 500s because the job
crashed is the wrong failure.

**Poll endpoint:** `GET /api/queue?since=<seq>` → entries + running + new events
+ cursor. Poll at **1000 ms while anything runs, 5000 ms idle**, and stop
entirely when the tab is hidden (`document.visibilityState`).

### The silent gaps

A `-crf 19 -preset slow` encode sits silent for two minutes. Show four true
things and no false ones:

1. **Last event + elapsed since it**, ticking client-side:
   `render · clip_02/audio_mix · two-pass loudnorm to -14 LUFS · 2m 14s`.
2. **An indeterminate bar.** Never a synthetic percentage.
3. **Real counts where real counts exist:** `stage 3 of 4` (from `JOB_STAGES`)
   and inside render `clip 2 of 7` (from `st.detail["clips"]`, which
   `pipeline.render` already writes before the loop).
4. **A liveness heartbeat, which is a different claim from progress.** The
   supervisor — not the worker, which is blocked inside ffmpeg — stamps
   `last_alive` each tick from `proc.poll() is None`. So the UI can honestly say
   *"process alive, 2m 14s since last progress"*, which distinguishes "silently
   working" from "hung". A static line beneath (`renders usually take 2-5 min
   per clip`) is phrased as an expectation, never a prediction.

### Optional, explicitly scoped

**yt-dlp `progress_hooks` — build it, phase 7.** `_ydl_options` sets
`noprogress: True` and registers no hooks; both `resolve()` and `download()`
already accept `ydl_options`, and `_ydl_options(**overrides)` merges them —
`pipeline.ingest` simply never passes any. Ingest is the first thing you see and
the longest pure wait, and yt-dlp hands over exact
`downloaded_bytes / total_bytes / eta / speed` for free. ~15 lines, throttled to
≤1 event/sec (the hook fires per chunk). Keep `noprogress: True` — that
suppresses the *console* bar, which is what we want; hooks fire regardless.

**ffmpeg `-progress pipe:1` — do NOT build.** It means converting `media.run`
from `subprocess.run` to a streaming `Popen`. `media.run` is the single choke
point for cut, reframe, audio_mix, captions, render and silencedetect, and **its
return value is stderr**, which `parse_silences` and the two-pass loudnorm both
parse. Rewriting it risks the most load-bearing 25 lines in the project for a
nicer bar. If ever wanted: add a *new* `media.run_with_progress(...)`, leave
`media.run` untouched, and call it from exactly one place — the final encode in
`edit/render.py` — behind a settings flag. Not scheduled.

---

## 3. The state-file race

`store.stage()` holds a `JobState` object for the whole stage — minutes.
`pipeline.moments` does `state.clips = []` and rebuilds.
`presenter.apply_review` does load → mutate → save. Overlap loses a write in
either direction, silently.

**Recommendation: single-writer discipline enforced at the API boundary. No
lock.**

A file lock would need a new dependency or `msvcrt.locking` and would have to be
added to every stage — and it still would not work: the read-modify-write window
*is* the whole stage, so a correct lock blocks the review endpoint for twenty
minutes, and any lock granular enough not to would not cover the race. More
code, more platform surface, still wrong.

The discipline, in three parts:

1. **While a job is running, only its worker writes `state.json`.** The queue is
   already the authority on what is running.
2. **`POST /api/jobs/<id>/clips/<clip>/review` returns `409`** when
   `id == queue.running_video_id()`, with a body naming the stage to wait for.
   `state.json` is not touched.
3. **The UI never normally hits that 409.** Every job payload carries
   `"writable": bool`; the SPA disables verdict buttons on a running job and
   says *"verdicts unlock when this job finishes"*. The 409 is the backstop for
   a stale tab.

**Why refuse rather than defer.** A pending-verdict sidecar applied later is a
second source of truth for `approved` — it recreates the merge problem it was
meant to avoid and makes "what did I decide?" ambiguous. Refusing is honest,
visible, and bounded by a job you are already watching. The workflow does not
need it: candidates only become reviewable *after* `moments`, by which point the
job is nearly done.

`presenter.apply_review` needs no change — it already re-reads fresh on every
call, which is the correct half. The discipline closes the other half.

**Two residual cases, handled with information rather than locks:**

- *A CLI run in a terminal while the server runs.* The supervisor writes
  `jobs/_queue/runner.json` (`pid`, `started_at`, `port`) at startup and refuses
  to start a second server. When a stage reads `running` but no queue entry owns
  it, the job page says *"stage says running, but no process owns this job — it
  was interrupted"* instead of pretending. This resolves the `should_run`
  ambiguity (a running stage is currently indistinguishable from a crashed one)
  without changing `should_run`'s semantics — liveness now lives in the queue,
  which is where it belongs.
- *Two browser tabs.* Both read-only while running, both through one API
  otherwise. No exposure.

---

## 4. API surface

All new routes under `/api/` — a prefix `review/app.py`'s 404 handler already
special-cases to return JSON. `/media/<job_id>/<path:relpath>` is kept verbatim.

### Queue

| Method | Route | Body / response |
|---|---|---|
| POST | `/api/queue` | `{"sources": [url, …], "options": {...}}` → `202 {"entries": […]}` |
| GET | `/api/queue?since=<seq>` | `{entries, running, events, seq, runner:{alive, free_bytes}}` |
| DELETE | `/api/queue/<entry_id>` | remove a queued entry; `409` if running |
| POST | `/api/queue/<entry_id>/cancel` | `202`; sentinel then tree-kill |
| POST | `/api/queue/<entry_id>/retry` | `{"force": false, "from_stage": null}` |

`sources` is an array so "paste several links" is one request. **No `allow_red`
at submit time** — see §5.

### Jobs

| Method | Route | Notes |
|---|---|---|
| GET | `/api/jobs` | `presenter.build_index(store)` |
| GET | `/api/jobs/<id>` | `presenter.build_job(...)` + queue overlay + `writable` |
| GET | `/api/jobs/<id>/events?since=` | history for the entry that produced this job |
| GET | `/api/jobs/<id>/log?tail=200` | `text/plain`, the worker's raw stdout/stderr |
| POST | `/api/jobs/<id>/clips/<clip>/review` | `presenter.apply_review`; `409` while running |
| POST | `/api/jobs/<id>/rights` | `{status, reason, remember}` — §5 |
| POST | `/api/jobs/<id>/render` | `{approved_only, clip_index, allow_red}` → render-only entry |
| DELETE | `/api/jobs/<id>/source` | reclaim multi-GB source, keep clips — §9 |
| DELETE | `/api/jobs/<id>` | requires `{"confirm": "<video_id>"}` |

### Media and downloads

| Method | Route | Notes |
|---|---|---|
| GET | `/media/<job_id>/<path>` | **unchanged** — inline preview, `conditional=True`, Range/206, suffix allowlist, traversal guard |
| GET | `/api/jobs/<id>/clips/<clip>/download` | same file, `as_attachment=True`, `download_name=f"{slug}-{clip_id}.mp4"` |
| GET | `/api/jobs/<id>/download` | zip of every rendered `final.mp4`, `stream_with_context` + `ZIP_STORED` (mp4 is already compressed) |

A separate download route is required: `as_attachment` is what makes the browser
save rather than play, and the preview route must stay inline. Same `_job_dir()`
guard.

### Settings / doctor

`GET /api/doctor` (shared with `clipper doctor`), `GET /api/settings`
(**read-only**), `GET /healthz` (kept).

**No settings editor.** The YAML files are largely load-bearing comments —
`watchlist.yaml` is 90% explanation and hard rule 6 makes `caption_style.yaml`
the most-iterated file in the project. A web form that round-trips YAML destroys
them. Show values and file paths; editing stays in the editor.

### The serialization layer — the real new work

`presenter.py`'s dataclasses are frozen with `@property` accessors, so
`dataclasses.asdict` drops `verdict`, `score_label`, `span_label`, `percent`,
`duration_label`, `pending`, `done`, `label`, `is_source`, and leaves raw
`datetime`s. The existing review POST hand-writes its dict for exactly this
reason.

```python
# src/clipper/server/schemas.py — Flask-free, pure functions
def iso(dt) -> str | None
def axis_json(a) -> dict                       # + percent
def preview_json(p, *, job_id) -> dict | None  # + label, is_source, url
def counts_json(c) -> dict                     # + pending, done
def clip_card_json(c, *, job_id) -> dict
def job_summary_json(s) -> dict
def job_view_json(v, *, writable, entry) -> dict
def queue_entry_json(e) -> dict
def event_json(e) -> dict
def check_json(c) -> dict
```

Four rules it enforces:

1. **Every property is emitted explicitly, by hand** — not by reflection. The
   wire format is a deliberate contract, and hand-writing it is what lets
   `test_server_schemas.py` assert the exact key set.
2. **Every datetime goes through `iso()`.**
3. **`preview_json` is where relative paths become URLs**
   (`/media/<job_id>/<quoted path>`), preserving `presenter.py`'s stated rule
   that it builds paths and never URLs. Routing stays out of the presenter.
4. **Every surfaced error passes through `strip_ansi()`** (§9).

This also lets the hand-written review POST response be replaced by
`clip_card_json`, deleting the duplication.

---

## 5. The rights gate

Hard rule 1: never bypass, never default to green. The bad outcome to design
against is: paste a link, walk away 40 minutes, come back to
`render failed: RightsBlocked`, having burned the download and the
transcription.

The earliest the gate can be evaluated is **immediately after `ingest`** —
that is when `resolve()` has produced `channel_id` and `licence`.

### The flow

1. **The paste box carries a rights selector**, defaulting to *"Check the
   watchlist"* (`rights=None`). The other options are green / amber / red, each
   labelled with what it means — **reuse the wording already written in
   `config/watchlist.yaml`'s comments verbatim** (own footage / public clipping
   policy / permission by email / runs its own clips channel). That education is
   already written; do not rewrite it.
2. The worker runs `ingest`, then emits a `rights` event with `status` and
   `state.rights_note`. The supervisor stamps `rights_status` onto the queue
   entry, so the row shows it within a minute or two of submission.
3. **Exportable → nothing happens.** Rights never interrupts a green job.
4. **Not exportable → the job stops right there**, before `transcribe`, with
   queue status **`blocked`** — a distinct status and colour, never `failed`.
   Nothing further is spent: no LLM quota, no CPU. This is the cheap failure the
   whole design is for.
5. **The job page shows a rights panel:**
   - Current status, `state.rights_note` **verbatim** (it already contains the
     full explanation including the `licence is not CC BY` clause), channel id,
     channel title, and the raw licence string yt-dlp reported.
   - **"Classify this job"** → `POST /api/jobs/<id>/rights` `{status, reason}`.
     This is exactly `--rights`: it affects one job and is recorded as an
     override with the reason. The reason field is **required** (≥10 chars) and
     the button is disabled without it. The job then resumes from `transcribe`.
   - **"Remember this channel"** checkbox → also appends an entry to
     `config/watchlist.yaml`, so the channel never asks again. This is how the
     highest-leverage file in the project finally starts getting filled in.
   - Below, **"Render anyway"** — the `--allow-red` equivalent — behind a
     confirm that spells out the consequence in the project's own words:
     *reused-content enforcement is at the channel level; one bad batch suspends
     monetization for everything.* Also requires a reason.
   - Links out to the source URL, plus the note that YouTube's filter must be
     "Creative Commons — Attribution (CC BY)" specifically.
6. Once overridden, the job carries a persistent **OVERRIDDEN** badge
   (`presenter.JobSummary.rights_override` already exposes it) and the reason is
   shown next to every clip. Recording an override is only worth anything if it
   is visible.

### Writing the watchlist

`src/clipper/rights/watchlist_edit.py` — `append_channel(path, entry)`.

It does a conservative **text-level** edit: replace the `channels: []` line with
a block list, or append an item to an existing block. **Not a YAML round-trip** —
the file is ~60 lines of hand-written commentary that PyYAML would silently
delete, and those comments are the reasoning (the CC-BY / NC / ND trap notes in
particular). The appended entry carries `id`, `name`, `rights_status`, a
`checked:` date and a `note:` holding the reason typed in the UI.

There is a real argument for not writing this file from a server at all — a
click is a weaker decision than an edit. The mitigation is that the reason is
mandatory, it is written into the entry, and the entry is appended in a form you
can read and revise. Revisit if it starts filling with thoughtless entries.

### Deliberately not built

No global "allow everything" toggle. No default-on `allow_red`. **No
`allow_red` option on the submit form at all** — you cannot meaningfully
pre-authorise an override for a channel you have not seen. The override is
reachable only *after* ingest, from the job page, with the rights note in front
of you and a reason typed. That is a deliberate tightening relative to the CLI,
and it is the right one: someone typing `--allow-red` knew what it meant; a
browser checkbox carries no such guarantee.

---

## 6. The SPA

```
frontend/
  package.json  vite.config.ts  tsconfig.json  tailwind.config.ts
  index.html  components.json
  src/
    main.tsx  App.tsx
    lib/      api.ts  types.ts  poll.ts  format.ts
    hooks/    usePolling.ts  useKeyboardShortcuts.ts
    components/ui/                        # shadcn primitives
    components/
      SubmitBox  QueueList  QueueRow  ProgressLine
      JobCard  ClipCard  VideoPreview  RubricBars
      VerdictButtons  RightsPanel  DoctorPanel  DiskPanel
    pages/  QueuePage  JobsPage  JobPage  SettingsPage
```

### Build output and serving

`vite build --outDir ../src/clipper/webui/dist --emptyOutDir` — the bundle lands
**inside the Python package**, so an install ships it and `clipper ui` works
with no separate static path.

```python
# src/clipper/webui/__init__.py
DIST: Path = Path(__file__).resolve().parent / "dist"
def is_built() -> bool
def register(app, *, dist=DIST) -> None
```

`register` adds a static handler for `dist/assets/**` and an SPA catch-all
returning `dist/index.html` for anything not under `/api/`, `/media/` or
`/healthz`. **It must be registered last** so it cannot shadow the API — and
there is a test for exactly that. If `dist/` is missing, the catch-all returns a
plain page saying `cd frontend && npm install && npm run build`, not a 404 or a
traceback.

**Dev mode:** `vite.config.ts` proxies `/api`, `/media`, `/healthz` to
`127.0.0.1:5151`. Two terminals: `uv run clipper ui --no-browser` and
`npm run dev` on `:5173`. Same-origin through the proxy, so no CORS code and no
CORS dependency.

**Commit the bundle.** The deliverable is "it works after `uv sync`"; requiring
Node to *run* a Python CLI is a real regression in setup cost. Node 22.17 is
already installed here for when you change the frontend. Mitigate diff noise
with `.gitattributes` marking `src/clipper/webui/dist/** -diff
linguist-generated=true`, a fixed `rollupOptions.output` naming scheme so
filenames are stable across builds, and a rule that `frontend/` and its `dist/`
are committed together. Hatchling already includes non-Python files under
`src/clipper/`.

### Design tokens — port, don't replace

`review.css` already defines a considered palette on `:root`, and the valuable
part is that its tokens are **semantic to this domain**: `--keep`/`--keep-bg`,
`--cut`/`--cut-bg`, `--warn`/`--warn-bg`, and a `--focus` distinct from
`--accent`. shadcn's default slate has no concept of keep-vs-cut.

Map the existing tokens onto shadcn's expected names (`--background`,
`--foreground`, `--card`, `--primary`, `--destructive`, `--ring`,
`--muted-foreground`) and keep `--keep`/`--cut`/`--warn` as extra semantic
colours in `tailwind.config.ts`. Preserve `color-scheme: light dark` and the
`prefers-color-scheme` block. Carry forward the **44 px minimum touch targets**
and the `prefers-reduced-motion` guards (Tailwind's `motion-safe:` /
`motion-reduce:`).

### Behaviour to preserve from `review.js`

Three things in those 229 lines are hard-won and must survive the rewrite:

1. **Keyboard shortcuts `j`/`k`/`a`/`r`/`u`/`space`** → `useKeyboardShortcuts.ts`.
   Keep the exact bindings; they are muscle memory.
2. **Lazy `<video>` mounting.** Do not set `src` until the card is near the
   viewport (`IntersectionObserver`). The preview fallback is `source.mp4` — a
   multi-GB file — so mounting twenty cards eagerly opens twenty range-request
   streams against the same file.
3. **Seek-to-start, pause-at-end.** On `loadedmetadata`,
   `currentTime = preview.start`; on `timeupdate`, pause at `preview.end`. This
   is what makes reviewing against the source video usable at all.

### Pages

| Page | Contents |
|---|---|
| `QueuePage` `/` | `SubmitBox` (textarea, one URL per line, + rights selector + music/centre-crop toggles); `QueueList` with per-row `ProgressLine` (stage · last event · elapsed ticker · indeterminate bar · stage N of 4 · clip N of M), cancel/retry, rights badge, free-disk banner. **This page alone is the product goal.** |
| `JobsPage` `/jobs` | `JobCard` grid: title, channel, rights badge, stage or `complete`, counts (`kept / cut / to do`), updated-at, per-job disk size. |
| `JobPage` `/jobs/:id` | Header (title, channel, duration, rights badge, source link, `writable`). `RightsPanel` when not exportable. `ClipCard` list: `VideoPreview`, suggested title, hook rewrite, span + duration, `RubricBars` with weights, transcript, `VerdictButtons`, per-clip **Download**, plus **Download all (.zip)**. Live `ProgressLine` if running; collapsed raw-log tail. |
| `SettingsPage` `/settings` | `DoctorPanel` (ok/warn/fail + hints), `DiskPanel` (free space, per-job sizes, delete-source), read-only config values with file paths. |

### The Jinja templates and `clipper review`

- **Phases 0-4: `review/` is untouched.** The API is additive; `/`,
  `/jobs/<id>`, the review POST and all 324 lines of `tests/test_review.py` keep
  passing.
- When the SPA takes over `/` (end of phase 4) the two UIs collide. Resolve with
  `create_app(..., ui: Literal["spa","jinja"] = "jinja")`. `clipper ui` passes
  `spa`; `clipper review` passes `jinja`. Under `spa` the Jinja pages move to
  `/legacy/`.
- **Phase 8: delete** `templates/`, `static/review.css`, `static/review.js` and
  the Jinja routes. `clipper review` becomes an alias for `clipper ui` printing
  one deprecation line. Rewrite the HTML-asserting tests against JSON; **leave
  the media-traversal and Range tests verbatim** — they guard the one route that
  is unchanged.
- **Keep `presenter.py` permanently.** Flask-free, well tested, and now the
  API's read model. It was written for this.
- Do not run both UIs indefinitely. Two review UIs writing `approved` is
  precisely the second-writer problem of §3.

---

## 7. Windows specifics

**Process-tree kill.**

```python
CREATE_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
Popen([sys.executable, "-m", "clipper.server.worker", ...],
      cwd=str(repo_root), creationflags=CREATE_FLAGS,
      stdout=log_fh, stderr=subprocess.STDOUT, env=os.environ.copy())
```

Cancel is `taskkill /T /F /PID <pid>`. `proc.terminate()` alone kills the Python
child and **orphans ffmpeg/yt-dlp**, which keeps a handle on `source.mp4` and
keeps burning CPU. `/T` walks the tree. Wrap both platforms in
`runner.kill_tree(pid, *, timeout_s=10.0)`; on POSIX, `start_new_session=True` +
`os.killpg`.

**`CREATE_NO_WINDOW`** so starting or cancelling a job does not flash a console
window on the desktop. This matters specifically for "open a page and walk
away".

**CWD-relative config — fix it properly.** `config.CONFIG_DIR = Path("config")`
is relative, so a server started elsewhere silently finds no config.
`clipper ui` captures `Path.cwd()` at startup, validates
`config/settings.yaml` exists there (exit `2` with *"run this from the repo
root, or pass --config"* if not), and passes **absolute** `--config` and
`--jobs-root` to every `Popen` **and** `cwd=<repo root>`. Never rely on CWD
inheritance; every path crossing a process boundary is absolute.

**`load_dotenv()`.** `cli.main()` loads it, so `clipper ui` inherits keys — but
the worker is spawned as `python -m clipper.server.worker`, which **bypasses
`cli.main()`**. `env=os.environ.copy()` covers it, and `worker.main()` should
call `load_dotenv(override=False)` anyway, since the worker is also runnable by
hand for debugging. Do **not** call `load_dotenv` in the server process — the
comment in `cli.py` about the review UI never picking up keys by accident stays
true for the web process, which makes no LLM calls. Only the worker needs keys.

**Paths in JSON.** `str(Path)` gives backslashes — fine for display, never for a
URL. `preview_json` uses the presenter's forward-slash relative path
(`clips/clip_01/final.mp4`) verbatim and `quote()`s each segment.

**Spaces in paths.** This repo lives under `…\New systems\shorts-clipper\`.
Every `Popen` takes an **argument list, never a shell string**; `shell=True`
appears nowhere. `media.py`'s existing `cwd`-plus-bare-filename trick for ffmpeg
filter escaping is already correct and is not touched.

**OneDrive.** `jobs_root` is inside a OneDrive folder. Multi-GB sources get
sync-scanned and can become cloud-only placeholders, and `os.replace` can
transiently raise `PermissionError` while the sync client holds a handle. The
doctor panel detects a `jobs_root` under a OneDrive path and recommends
relocating via `paths.jobs_root: D:/clipper-jobs`. This is a genuine hazard on
this exact machine, not a theoretical one.

**Port in use.** Catch `OSError` on bind and print *"port 5151 is in use —
another `clipper ui` may already be running"* rather than a Werkzeug traceback.
Pair with the `runner.json` claim file from §3.

---

## 8. Testing

Follow `tests/test_review.py` exactly: no server started, `create_app(...)` +
`app.test_client()`, a real `JobStore` in `tmp_path`, and assertions on **both**
the HTTP response **and** the state on disk.

| File | Covers |
|---|---|
| `tests/test_server_queue.py` | enqueue/ordering; atomic persistence (save, reload via a *fresh* `QueueStore`, identical); cancel a queued entry; `reclaim()` turns running + dead pid into `interrupted`; `running_video_id()` |
| `tests/test_server_events.py` | append/read; `since=` cursor; monotonic `seq`; ANSI stripped; a **truncated final line** is skipped not fatal; unicode round-trips; duplicate throttling |
| `tests/test_server_runner.py` | **the critical file.** `JobRunner(..., spawn=fake_spawn, clock=FakeClock())` with a `FakeProcess` implementing `ProcessHandle`. Drive `tick()` directly — no thread, no sleeping |
| `tests/test_server_worker.py` | `worker.run_entry` with `pipeline.run_all` monkeypatched to a fake that fires `on_progress` and returns a `JobState` |
| `tests/test_server_api.py` | every route through `test_client()` |
| `tests/test_server_schemas.py` | the wire contract |
| `tests/test_webui.py` | SPA catch-all and route precedence |

**The tests that earn their keep:**

- **Cancellation, no real process:** `runner.cancel(id)` → assert the `.cancel`
  sentinel was written, `FakeProcess.killed is True`, entry is `cancelled`.
- **Crash isolation, as a testable claim:** `FakeProcess` exits with
  `-1073741819` (`0xC0000005`, the access violation a MediaPipe segfault
  produces). Assert the entry is `failed` with a message naming a crash rather
  than a Python exception, **and that the next queued entry still starts**. That
  is the entire justification for the subprocess design, verified.
- **The §3 race, verified by bytes:** review POST on a non-running job lands in
  `state.json`; on the *running* job it returns `409` **and the state file's
  bytes are identical before and after**. Read, POST, read, `assert before ==
  after`.
- **The datetime class of bugs in one line:**
  `json.dumps(clip_card_json(...))` must not raise.
- **The forgotten-field regression:** iterate `dataclasses.fields()` of each
  presenter dataclass and assert every field name appears in the corresponding
  JSON payload. Catches "someone added a field to `ClipCard` and forgot the
  serializer".
- **Route shadowing:** `GET /api/nope` returns JSON 404, not `index.html`.
- **Zip integrity:** `zipfile.ZipFile(BytesIO(resp.data))` opens and
  `namelist()` matches the rendered clips.
- **No secret leakage:** `/api/doctor` reports key *presence* only — assert no
  response value equals any env var set in the test.

**Conventions honoured.** No network: no route resolves a URL (resolution lives
in the worker, whose `pipeline` call is faked). No real ffmpeg: the server layer
never shells out, and media routes serve fixture bytes. No sleeping: `tick()` +
`FakeClock` from `tests/conftest.py`, exactly as the limiter tests do.

**Frontend tests: not in v1.** Low value against a hand-driven local UI, real
maintenance cost. Instead `npm run build` runs `tsc --noEmit` first, and
`lib/types.ts` is kept honest by `test_server_schemas.py` asserting exact key
sets. The one narrow exception worth adding later is a Vitest test for
`VideoPreview`'s seek/pause logic, because that behaviour is being ported from
code that already works.

---

## 9. Production readiness

**Structured logging.** `server/logging.py::configure(level, log_dir)` →
`RotatingFileHandler` at `jobs/_queue/logs/server.log` (5 × 2 MB) plus console.
Each worker's raw stdout/stderr is redirected into
`jobs/_queue/logs/worker-<entry_id>.log` — that is how yt-dlp's and ffmpeg's
chatter is captured without polluting the structured event log.
`GET /api/jobs/<id>/log?tail=200` serves the tail; the job page links it. This
is what turns *"it failed"* into *"I can see why"*.

**ANSI stripping.** `server/text.py::strip_ansi(text)`, regex
`\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])`. Applied in exactly two places: when the
worker writes an `error` event, and in `schemas.py` when surfacing
`StageState.error`. Deliberately **not** applied to the raw log file — that
keeps the original bytes. Surfaced errors are truncated (~2000 chars) with a
"… see raw log" link; an ffmpeg stderr tail is 15 lines of mostly noise.

**Graceful shutdown.** Ctrl-C must not orphan a 20-minute render.
`JobRunner.shutdown()`: stop accepting work, write the cancel sentinel, wait
`timeout_s`, then `kill_tree`. Mark the entry **`interrupted`** — distinct from
`cancelled`, because you did not ask — and leave the stage `running` so
`should_run` reruns it. Next startup shows it with a one-click retry. Registered
via `atexit` and a `SIGINT` handler. **Disable the Werkzeug reloader entirely**
(`debug=False, use_reloader=False`): a reloader that restarts the process
mid-job is a disaster, and the SPA is served from a built bundle so there is
nothing to reload.

Kill rather than detach: an orphaned worker writing `state.json` after the
server is gone recreates the second-writer problem with nobody watching.

**Disk awareness.** 1440p sources are 0.5-2 GB, are never deleted, and land on a
drive currently at 86% with 70 GB free. `server/disk.py`:

```python
def free_bytes(path) -> int                  # shutil.disk_usage
def job_sizes(store) -> list[JobSize]        # cached by dir mtime
def source_bytes(store, video_id) -> int
```

- **Preflight in `tick()`:** if free < a configurable floor (default 10 GB),
  refuse to start and mark the entry `blocked` with a disk reason — better than
  failing three stages in with a half-written file.
- **`DELETE /api/jobs/<id>/source`** — the highest-value disk action, since the
  source is ~95% of a job's bytes and is only needed for a re-render. Explicit,
  with a warning that re-render means re-download. The system already handles
  it: `source_video_path()` raises a clear error and `should_run` re-runs
  ingest.

**Doctor — one source of truth.** Extract the checks out of `cli.py` into
`src/clipper/server/doctor.py`:

```python
@dataclass(frozen=True)
class Check: name: str; status: Literal["ok","warn","fail"]; detail: str; hint: str | None
def run_checks(settings, *, jobs_root: Path, cwd: Path) -> list[Check]
```

`cli.doctor` renders it with rich; `/api/doctor` serializes it. The CLI gains
the new checks for free. Checks: ffmpeg + ffprobe · config files present · CWD
is the repo root · provider key presence (boolean only) · OAuth presence · face
model file · **watchlist channel count, with the loud "empty → every source
resolves to unknown and render will refuse it"** · free disk · jobs_root under
OneDrive · `webui/dist` present · another runner holding the claim file.

**Bind policy is stricter than `review`'s.** Keep `_is_loopback`'s warning, but
`clipper ui` **refuses** a non-loopback bind without an explicit acknowledgement
flag. `review` was read-mostly; `ui` can start multi-GB downloads and delete
files. The risk calculus changed and the code should say so.

### Not built

Upload from the browser (hard rule 10) · auth / login / HTTPS · a YAML settings
editor · a database · Celery / RQ / APScheduler · SSE or websockets ·
**multi-job parallelism** (two renders on one desktop is slower than serial
*and* makes the state race real again) · Docker · auto-retry on failure (a job
that silently re-downloads 2 GB three times is worse than a red row) · ETAs
computed from historical averages (dishonest) · a "render all jobs" button ·
telemetry.

---

## 10. Delivery order

| Phase | Ships | Testable milestone |
|---|---|---|
| **0 — API skeleton + doctor** | `server/{__init__,api,schemas,doctor,text}.py`; blueprint on the existing `create_app` behind `api=True`; `GET /api/jobs`, `/api/jobs/<id>`, `/api/doctor`; `cli.doctor` refactored onto `server/doctor.py` | `curl localhost:5151/api/jobs` returns the two real jobs with `verdict`/`span_label` present and datetimes as strings. 619 tests still green. |
| **1 — durable queue, nothing runs** | `server/queue.py`, `server/events.py`; `POST/GET/DELETE /api/queue` | Submit three URLs, restart the server, all three still queued in order. |
| **2 — worker + runner** | `server/worker.py`, `server/runner.py`; tree-kill; cancel; heartbeat | Submit a **local file** (no network needed to demo) and watch queued → running → done with events streaming into `GET /api/queue?since=`. Cancel mid-run; confirm Task Manager shows no orphaned ffmpeg. **Test this phase hardest.** |
| **3 — SPA shell** | `frontend/` scaffold, Tailwind carrying the `review.css` tokens, shadcn, `QueuePage`, `webui/` + catch-all, dist committed, Vite proxy | **The product goal, minimally:** open `127.0.0.1:5151`, paste a URL, walk away, come back to a finished job. |
| **4 — job list + detail** | `JobsPage`, `JobPage`, `ClipCard`, `VideoPreview` (lazy mount + seek/pause), `RubricBars`, verdict buttons, keyboard hook, download-one + download-zip | **The stated goal, fully met:** preview and download finished clips from the page. |
| **5 — rights + retry + render-only** | `RightsPanel`, `POST /api/jobs/<id>/rights`, `blocked` status, `watchlist_edit.py`, retry, `POST /api/jobs/<id>/render` | Paste an arbitrary link against the empty watchlist → stops at `blocked` with the reason → classify with a reason → the channel appears in `config/watchlist.yaml` → render from the page, no terminal. |
| **6 — operations** | logging + raw-log route, disk panel + preflight + delete-source, graceful shutdown, `SettingsPage`, port-in-use message, bind refusal | Kill the server mid-render, restart: the queue shows `interrupted` with a working retry, and no ffmpeg is left running. |
| **7 — yt-dlp progress hooks** *(optional)* | `progress_hooks` passthrough in `download.py`, plumbed through `pipeline.ingest`, throttled | Ingest shows a real percentage and ETA. |
| **8 — retire the Jinja UI** | delete templates / `review.css` / `review.js` / Jinja routes; `clipper review` → deprecating alias; rewrite HTML-asserting tests as JSON | One UI. `presenter.py`, the `/media/` route and its traversal/Range tests survive verbatim. |

Not scheduled: ffmpeg `-progress pipe:1`.

---

## Verification

- `uv run pytest` — the 619 existing tests stay green throughout; new suites per
  §8.
- `uv run ruff check . && uv run ruff format --check .`
- **End to end by hand**, the only thing that really proves it: `clipper ui`,
  paste the URL of the CC BY video already in `jobs/wYIRDcjPjD8/state.json` (a
  known-green channel, so the rights path is not in the way), confirm clips
  render and download. Then paste an unlisted channel and confirm the rights
  card appears instead of a 40-minute failure.
- Cancel a job mid-download and check Task Manager: no orphaned `ffmpeg.exe` or
  yt-dlp.

## Before starting

**This repo is not under version control** — there is a `.gitignore` but no
`.git`, so the intent was there. This plan adds a new package, a React app and a
committed build output, and eventually retires a working UI. Run `git init` and
commit the current green state first, and check that `.gitignore` does not
exclude `src/clipper/webui/dist/`.

## Critical files

- `src/clipper/pipeline.py` — `run_all(..., auto=True, on_progress=...)` is the
  worker's single entry point; `Progress = Callable[[str, str], None]` is the
  contract the event log adapts to.
- `src/clipper/state.py` — `JobStore.stage()` / `save()` / `should_run()` define
  the race in §3 and the kill-safety in §1; `JOB_STAGES` drives the honest
  "stage N of 4".
- `src/clipper/review/app.py` — `create_app` gains the `/api` blueprint, the
  `ui=` switch and the SPA catch-all; `/media/` and `_job_dir()` are reused
  unchanged.
- `src/clipper/review/presenter.py` — the API's read model; its
  `@property`-bearing frozen dataclasses are exactly what `server/schemas.py`
  exists to serialize.
- `src/clipper/cli.py` — `main()`'s `load_dotenv` placement (§7), the `review`
  command's serve pattern that `ui` follows, and the `doctor` body being
  extracted.
- `tests/test_review.py` — the shape every new server test copies.
