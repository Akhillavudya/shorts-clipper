/**
 * What a running job is doing, in one line plus a bar.
 *
 * Every number on this line is one the pipeline actually reports. There is no
 * percentage and no ETA, because neither exists: the stages emit names, and a
 * synthetic percentage that creeps to 90% and sits there teaches you to stop
 * believing the page. What is honest is the stage, the last thing that
 * happened, how long ago it happened, and - the one claim the supervisor
 * really can make - that the process is alive.
 *
 * The silence is information too. "reframe, 4m 12s ago" on a long episode is
 * normal; the same line at twenty minutes is a hung ffmpeg, and this is where
 * you notice.
 */

import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import {
  JOB_STAGES,
  clipNumber,
  elapsed,
  sinceMs,
  splitClipStage,
  stagePosition,
} from "@/lib/format"
import type { ProgressEvent, QueueEntry } from "@/lib/types"

interface ProgressLineProps {
  entry: QueueEntry
  events: ProgressEvent[]
  /** Re-rendered every second by `useNow`, so the elapsed figures advance
   * between polls rather than jumping in whole intervals. */
  now: number
  /** How many clips this job's `render` will walk, once `moments` has
   * produced them. `null` until the job page has been read - the queue itself
   * does not know, and "clip 3" without an "of" is better than a guess. */
  clipTotal?: number | null
  className?: string
}

/** The last event whose `kind` is one of the given ones. */
function lastOf(events: ProgressEvent[], kinds: string[]): ProgressEvent | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (kinds.includes(events[i].kind)) return events[i]
  }
  return null
}

export function ProgressLine({
  entry,
  events,
  now,
  clipTotal = null,
  className,
}: ProgressLineProps) {
  const running = entry.status === "running"

  // `kind: "stage"` is emitted by the worker at each of the four boundaries;
  // `kind: "progress"` is everything the pipeline reports inside one. The
  // first gives "stage 3 of 4", the second gives the detail line.
  const stageEvent = lastOf(events, ["stage"])
  const latest = events.length > 0 ? events[events.length - 1] : null

  const position = stageEvent ? stagePosition(stageEvent.stage) : null
  const stageName = stageEvent?.stage ?? JOB_STAGES[0]

  const clip = latest ? splitClipStage(latest.stage) : null
  const clipIndex = clip ? clipNumber(clip.clip) : null

  const sinceLast = sinceMs(latest?.ts ?? entry.started_at, now)
  const runningFor = sinceMs(entry.started_at, now)

  const detail = latest?.detail || latest?.stage || "starting"

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      {running && (
        <div
          className="relative h-1 w-full overflow-hidden rounded-full bg-secondary indeterminate-bar"
          role="progressbar"
          aria-label={`${stageName} in progress`}
          // No `aria-valuenow`: an indeterminate progressbar is exactly what
          // the ARIA spec says to omit it for, and screen readers announce
          // "busy" rather than inventing a number.
        />
      )}

      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        {running && <Spinner className="size-3.5 text-muted-foreground" />}
        <span className="font-mono font-medium">{stageName}</span>
        {position !== null && (
          <span className="text-muted-foreground">
            stage {position} of {JOB_STAGES.length}
          </span>
        )}
        {clipIndex !== null && (
          <span className="text-muted-foreground">
            clip {clipIndex}
            {clipTotal ? ` of ${clipTotal}` : ""}
            {clip ? ` · ${clip.step}` : ""}
          </span>
        )}
      </p>

      <p className="flex flex-wrap items-baseline gap-x-2 text-sm text-muted-foreground">
        <span className="min-w-0 break-words">{detail}</span>
        <span className="tabular-nums">· {elapsed(sinceLast)} ago</span>
        {runningFor !== null && (
          <span className="tabular-nums">· running {elapsed(runningFor)}</span>
        )}
        {entry.pid !== null && (
          // Deliberately surfaced. When a render has been silent for two
          // minutes, "pid 24180" is both the evidence that something is
          // alive and the thing you type into Task Manager if it is not.
          <span className="font-mono text-xs">· pid {entry.pid}</span>
        )}
      </p>
    </div>
  )
}
