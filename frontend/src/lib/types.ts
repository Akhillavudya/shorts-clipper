/**
 * The wire format, typed.
 *
 * Every type here is the TypeScript counterpart of one function in
 * `src/clipper/server/schemas.py`, and the pairing is deliberate: that module
 * writes every field out by hand precisely so this file can be written by hand
 * against it and a test can assert the exact key set on the Python side. If
 * you add a field there, add it here; nothing generates either from the other,
 * which is the cost of a contract you can read in one sitting.
 */

/** `state.StageStatus`. */
export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped"

/** `queue.QueueStatus`. Seven terminal-or-not states, each meaning one thing.
 *
 * `blocked` is not `failed` (a rights block is the cheap correct outcome) and
 * `interrupted` is not `cancelled` (nobody asked for it). The UI colours them
 * apart for exactly that reason. */
export type QueueStatus =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "cancelled"
  | "interrupted"
  | "blocked"

export type RightsStatus = "green" | "amber" | "red" | "unknown"

export type CheckStatus = "ok" | "warn" | "fail"

/** `schemas.run_options_json`. */
export interface RunOptions {
  rights: string | null
  music: string | null
  no_music: boolean
  centre_crop: boolean
  force: boolean
  render_only: boolean
  approved_only: boolean
  clip_index: number | null
  allow_red: boolean
  override_reason: string | null
}

/** What the submit box may set. `allow_red` is absent on purpose: the API
 * returns 400 for it, because a rights override cannot be pre-authorised for a
 * channel nobody has seen yet (plan section 5). */
export type SubmitOptions = Partial<
  Pick<RunOptions, "rights" | "music" | "no_music" | "centre_crop" | "force">
>

/** `schemas.queue_entry_json`. */
export interface QueueEntry {
  entry_id: string
  source: string
  status: QueueStatus
  label: string
  options: RunOptions
  video_id: string | null
  rights_status: RightsStatus | null
  pid: number | null
  submitted_at: string | null
  started_at: string | null
  finished_at: string | null
  last_alive: string | null
  error: string | null
  exit_code: number | null
}

/** `schemas.event_json` — one line of `jobs/_queue/events/<entry_id>.jsonl`. */
export interface ProgressEvent {
  seq: number
  ts: string
  kind: string
  stage: string
  detail: string
}

/** `GET /api/queue`. One request per poll tick, carrying both the whole queue
 * and a page of events for the entry named by `events_entry_id`. */
export interface QueueResponse {
  entries: QueueEntry[]
  running: QueueEntry | null
  events: ProgressEvent[]
  events_entry_id: string | null
  seq: number
  runner: {
    /** False under `clipper review`, which attaches no supervisor, and false
     * if the supervisor thread died. Either way nothing is draining the
     * queue, and the page has to say so rather than looking idle. */
    alive: boolean
    free_bytes: number | null
  }
}

/** `POST /api/queue`. */
export interface SubmitResponse {
  entries: QueueEntry[]
}

/** `schemas.counts_json`. */
export interface Counts {
  total: number
  approved: number
  rejected: number
  pending: number
  done: boolean
}

/** `schemas.job_summary_json`. */
export interface JobSummary {
  video_id: string
  title: string
  channel_title: string
  rights_status: RightsStatus
  rights_override: boolean
  exportable: boolean
  stage: string
  stage_failed: boolean
  counts: Counts
  updated_at: string | null
}

export interface JobsResponse {
  jobs: JobSummary[]
  jobs_root: string
}

export interface RubricAxis {
  name: string
  label: string
  score: number
  weight: number | null
  percent: number
}

export interface Preview {
  path: string
  kind: string
  start: number
  end: number
  label: string
  is_source: boolean
  url: string
}

export interface Clip {
  clip_id: string
  index: number
  score: number | null
  score_label: string
  type: string | null
  reason: string | null
  suggested_title: string | null
  hook_rewrite: string | null
  emphasis_words: string[]
  start: number | null
  end: number | null
  span_label: string
  duration_s: number | null
  duration_label: string
  rubric: RubricAxis[]
  transcript: string
  approved: boolean | null
  verdict: "approved" | "rejected" | "pending"
  review_note: string
  reviewed_at: string | null
  preview: Preview | null
  rendered: boolean
  render_note: string | null
  youtube_video_id: string | null
  source_link: string | null
}

export interface JobView {
  channel_id: string | null
  licence: string | null
  rights_override_reason: string | null
  video_id: string
  title: string
  channel_title: string | null
  source_url: string | null
  duration_s: number | null
  duration_label: string
  rights_status: RightsStatus
  rights_note: string | null
  rights_override: boolean
  exportable: boolean
  counts: Counts
  clips: Clip[]
  warnings: string[]
  writable: boolean
  entry: QueueEntry | null
  stages: Stage[]
  next_stage: string | null
}

/** `schemas.stage_json`. */
export interface Stage {
  name: string
  status: StageStatus
  started_at: string | null
  finished_at: string | null
  duration_s: number | null
  attempts: number
  error: string | null
  detail: Record<string, unknown>
}

/** `schemas.check_json`. */
export interface Check {
  name: string
  status: CheckStatus
  detail: string
  hint: string
}

export interface DoctorResponse {
  checks: Check[]
  status: CheckStatus
}

/** What the API hands back on a non-2xx. Every `abort()` inside the blueprint
 * answers in this shape, so there is no HTML error page to parse as JSON. */
export interface ApiError {
  error: string
}
