/**
 * Display helpers. No state, no DOM - just strings, so the formatting rules
 * live in one readable place instead of being inlined across six components.
 */

import type { QueueStatus, RightsStatus } from "@/lib/types"

/** `state.JOB_STAGES`, in order. The source of "stage 3 of 4".
 *
 * Duplicated from Python rather than fetched, and that is a deliberate
 * trade: it is a four-element tuple that has not changed since the pipeline
 * was written, and an extra round trip on every page load to learn it would
 * buy nothing. If it ever does change, `stagePosition` is the one place here
 * that breaks.
 */
export const JOB_STAGES = ["ingest", "transcribe", "moments", "render"] as const

export type JobStage = (typeof JOB_STAGES)[number]

/** 1-based position of a stage, or `null` for anything that is not one of the
 * four - `resolve`, `download`, `clip_01/reframe` and the rest of the
 * fine-grained progress names all land here. */
export function stagePosition(stage: string): number | null {
  const index = (JOB_STAGES as readonly string[]).indexOf(stage)
  return index === -1 ? null : index + 1
}

/** `clip_01/reframe` -> `{ clip: "clip_01", step: "reframe" }`.
 *
 * `pipeline.render` prefixes every per-clip progress note with the clip id, so
 * this is how the queue row knows which clip a forty-minute render is on
 * without the server having to model it.
 */
export function splitClipStage(stage: string): { clip: string; step: string } | null {
  const match = /^(clip_\d+)\/(.+)$/.exec(stage)
  return match ? { clip: match[1], step: match[2] } : null
}

/** `clip_07` -> `7`. */
export function clipNumber(clipId: string): number | null {
  const match = /^clip_(\d+)$/.exec(clipId)
  return match ? Number(match[1]) : null
}

/** Seconds as `3:41` or `1:02:11`. Used for source durations and elapsed. */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "-"
  const total = Math.max(0, Math.round(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n: number) => String(n).padStart(2, "0")
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`
}

/** Milliseconds since an ISO timestamp, or `null` if there isn't one. */
export function sinceMs(iso: string | null | undefined, now: number): number | null {
  if (!iso) return null
  const then = Date.parse(iso)
  return Number.isNaN(then) ? null : Math.max(0, now - then)
}

/** A short "how long ago", for the last-event line: `4s`, `3m 12s`, `1h 04m`.
 *
 * Deliberately coarse past an hour. The number that matters while a render is
 * running is "has anything happened recently", and a silent gap is the signal
 * - see the runner's heartbeat, which is a liveness claim and not a progress
 * one.
 */
export function elapsed(ms: number | null): string {
  if (ms === null) return "-"
  const total = Math.floor(ms / 1000)
  if (total < 60) return `${total}s`
  const m = Math.floor(total / 60)
  const s = total % 60
  if (m < 60) return `${m}m ${String(s).padStart(2, "0")}s`
  const h = Math.floor(m / 60)
  return `${h}h ${String(m % 60).padStart(2, "0")}m`
}

/** Bytes as `48.2 GB`. Binary units, because that is what Explorer shows and
 * this number is checked against Explorer. */
export function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "unknown"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let size = value
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024
    unit += 1
  }
  return `${size.toFixed(size >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`
}

/** A local wall-clock time, for "submitted at". Dates are ISO-8601 with an
 * offset on the wire, so `new Date()` reads them correctly. */
export function clockTime(iso: string | null | undefined): string {
  if (!iso) return "-"
  const date = new Date(iso)
  return Number.isNaN(date.getTime())
    ? "-"
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
}

/** One sentence per queue status, shown under the row.
 *
 * `blocked` and `interrupted` get the longest ones on purpose: both are states
 * a person meets for the first time in the middle of something else, and both
 * have a specific next action that is not obvious from the word alone.
 */
export const STATUS_HINT: Record<QueueStatus, string> = {
  queued: "waiting for the job before it to finish",
  running: "a worker owns this job's state file",
  done: "rendered - open it to preview and keep or cut",
  failed: "the worker exited with an error",
  cancelled: "you stopped this one",
  interrupted: "the server died while this was running - retry resumes it",
  blocked: "stopped at the rights gate before spending any quota",
}

/** Tailwind classes per status. Kept here rather than in the row so the queue
 * list and, later, the job pages cannot drift apart on what "blocked" looks
 * like. The domain colours are the ported `--keep` / `--cut` / `--warn`. */
export const STATUS_CLASS: Record<QueueStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-secondary text-secondary-foreground",
  done: "bg-keep-bg text-keep",
  failed: "bg-cut-bg text-cut",
  cancelled: "bg-muted text-muted-foreground",
  interrupted: "bg-warn-bg text-warn",
  blocked: "bg-warn-bg text-warn",
}

export const RIGHTS_CLASS: Record<RightsStatus, string> = {
  green: "bg-keep-bg text-keep",
  amber: "bg-warn-bg text-warn",
  red: "bg-cut-bg text-cut",
  unknown: "bg-muted text-muted-foreground",
}
