/**
 * One submission, from paste to terminal status.
 *
 * The row is addressed by `entry_id`, not by `video_id`, and that is not an
 * implementation detail leaking out: at submit time all that exists is a URL.
 * The video id arrives minutes later from `ingest`, and until it does the row
 * still has to be showable, cancellable and deletable. `label` is the server's
 * answer to "what do I call this in the meantime".
 */

import { ExternalLinkIcon, RotateCcwIcon, TrashIcon, XIcon } from "lucide-react"
import { Link } from "react-router-dom"

import { ProgressLine } from "@/components/ProgressLine"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import {
  RIGHTS_CLASS,
  STATUS_CLASS,
  STATUS_HINT,
  clockTime,
} from "@/lib/format"
import type { ProgressEvent, QueueEntry } from "@/lib/types"

interface QueueRowProps {
  entry: QueueEntry
  /** Events for this entry, or `[]`. The server sends a page for exactly one
   * entry per poll - the running one - so every other row renders without. */
  events: ProgressEvent[]
  now: number
  clipTotal: number | null
  /** False when no supervisor is attached. Cancel and retry both answer 503 in
   * that case, so the buttons say so instead of offering a request the server
   * has already told us it cannot honour. */
  runnerAlive: boolean
  onCancel: (entry: QueueEntry) => void
  onRetry: (entry: QueueEntry) => void
  onDelete: (entry: QueueEntry) => void
  busy: boolean
}

/** Statuses a retry makes sense for. Not `done`: re-running a finished job
 * from the queue page would silently re-render clips someone has already
 * reviewed, and `force` is what that actually wants. */
const RETRYABLE = new Set(["failed", "cancelled", "interrupted", "blocked"])

export function QueueRow({
  entry,
  events,
  now,
  clipTotal,
  runnerAlive,
  onCancel,
  onRetry,
  onDelete,
  busy,
}: QueueRowProps) {
  const running = entry.status === "running"
  const active = running || entry.status === "queued"

  return (
    <li
      className={cn(
        "flex flex-col gap-3 rounded-xl border bg-card p-4 sm:p-5",
        running && "border-primary/40"
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge className={cn("font-medium", STATUS_CLASS[entry.status])}>
              {entry.status}
            </Badge>
            {entry.rights_status && (
              <Badge
                className={cn("font-medium", RIGHTS_CLASS[entry.rights_status])}
                title="rights_status - render, publish and upload all refuse red and unknown"
              >
                {entry.rights_status}
              </Badge>
            )}
            <span className="text-sm font-medium break-all">{entry.label}</span>
          </div>
          <p
            className="truncate text-xs text-muted-foreground"
            title={entry.source}
          >
            {entry.source}
          </p>
          <p className="text-xs text-muted-foreground">
            {STATUS_HINT[entry.status]} · queued {clockTime(entry.submitted_at)}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {busy && <Spinner className="size-4 text-muted-foreground" />}

          {entry.video_id && (
            <Button variant="outline" size="sm" asChild>
              <Link to={`/jobs/${encodeURIComponent(entry.video_id)}`}>
                <ExternalLinkIcon data-icon="inline-start" />
                Open
              </Link>
            </Button>
          )}

          {active && (
            <Button
              variant="outline"
              size="sm"
              disabled={busy || !runnerAlive}
              title={
                runnerAlive
                  ? "Stop this job. A running worker exits at its next stage boundary."
                  : "No supervisor is attached, so there is nothing to cancel against."
              }
              onClick={() => onCancel(entry)}
            >
              <XIcon data-icon="inline-start" />
              Cancel
            </Button>
          )}

          {RETRYABLE.has(entry.status) && (
            <Button
              variant="outline"
              size="sm"
              disabled={busy || !runnerAlive}
              title="Put it back in the queue, resuming from the first stage that is not done."
              onClick={() => onRetry(entry)}
            >
              <RotateCcwIcon data-icon="inline-start" />
              Retry
            </Button>
          )}

          {!running && (
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Remove ${entry.label} from the queue`}
              title="Remove this row. The job directory is left alone."
              disabled={busy}
              onClick={() => onDelete(entry)}
            >
              <TrashIcon />
            </Button>
          )}
        </div>
      </div>

      {running && (
        <ProgressLine
          entry={entry}
          events={events}
          now={now}
          clipTotal={clipTotal}
        />
      )}

      {entry.error && (
        <p
          className={cn(
            "rounded-md px-3 py-2 font-mono text-xs break-words",
            entry.status === "blocked"
              ? "bg-warn-bg text-warn"
              : "bg-cut-bg text-cut"
          )}
        >
          {entry.error}
          {entry.exit_code !== null && (
            <span className="opacity-70"> (exit {entry.exit_code})</span>
          )}
        </p>
      )}
    </li>
  )
}
