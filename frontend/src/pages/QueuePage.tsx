/**
 * Paste a link, walk away, come back to finished Shorts.
 *
 * This page alone is the product goal (plan section 10, phase 3). It owns the
 * poll loop for the whole app because the queue is the only thing that changes
 * without the user doing something, and one loop in one place is what keeps
 * "how often does this refresh" answerable.
 */

import * as React from "react"
import {
  AlertTriangleIcon,
  HardDriveIcon,
  PauseIcon,
  Clock3,
  LoaderCircle,
  CheckCheck,
  Plus,
  Library,
} from "lucide-react"
import { Link } from "react-router-dom"
import { StatCard } from "@/components/StatCard"
import { Button } from "@/components/ui/button"

import { QueueList } from "@/components/QueueList"
import { SubmitBox } from "@/components/SubmitBox"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { useDocumentVisible, useNow, usePolling } from "@/hooks/usePolling"
import * as api from "@/lib/api"
import { bytes } from "@/lib/format"
import { mergeEvents, nextSince, pollIntervalMs } from "@/lib/poll"
import type {
  ProgressEvent,
  QueueEntry,
  QueueResponse,
  SubmitOptions,
} from "@/lib/types"

/** Below this, a 1440p download plus its intermediate renders is a real risk
 * of filling the volume mid-encode. The plan's disk preflight is phase 6; this
 * is the banner half of it, which costs one number the queue route already
 * sends. */
const LOW_DISK_BYTES = 20 * 1024 ** 3

export function QueuePage() {
  const [queue, setQueue] = React.useState<QueueResponse | null>(null)
  const [events, setEvents] = React.useState<ProgressEvent[]>([])
  const [eventsEntryId, setEventsEntryId] = React.useState<string | null>(null)
  const [clipTotal, setClipTotal] = React.useState<number | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [submitError, setSubmitError] = React.useState<string | null>(null)
  const [busyEntryId, setBusyEntryId] = React.useState<string | null>(null)

  const visible = useDocumentVisible()

  // Read inside the tick without making the tick depend on them, which would
  // restart the timer chain on every poll.
  const cursor = React.useRef({ entryId: null as string | null, seq: 0 })
  const active = React.useMemo(
    () =>
      (queue?.entries ?? []).some(
        (e) => e.status === "queued" || e.status === "running"
      ),
    [queue]
  )

  const now = useNow(active && visible)

  const refresh = React.useCallback(async () => {
    try {
      const response = await api.getQueue({ since: cursor.current.seq })
      setError(null)

      setEvents((previous) =>
        mergeEvents(
          previous,
          cursor.current.entryId,
          response.events,
          response.events_entry_id
        )
      )
      setEventsEntryId(response.events_entry_id)
      cursor.current = {
        entryId: response.events_entry_id,
        seq: nextSince(
          cursor.current.entryId,
          response.events_entry_id,
          response.seq
        ),
      }
      setQueue(response)

      // One extra request, and only while a worker owns a job that has a video
      // id: it is what turns "clip 3" into "clip 3 of 7". The queue does not
      // know how many clips `moments` produced - only the job state does.
      const running = response.running
      if (running?.video_id && running.status === "running") {
        try {
          const job = await fetch(
            `/api/jobs/${encodeURIComponent(running.video_id)}`
          )
          if (job.ok) {
            const payload = (await job.json()) as {
              counts?: { total?: number }
            }
            setClipTotal(payload.counts?.total ?? null)
          }
        } catch {
          // A missing clip count costs one word on one line. It is never worth
          // failing the poll over.
          setClipTotal(null)
        }
      } else {
        setClipTotal(null)
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [])

  usePolling(refresh, {
    intervalMs: () =>
      pollIntervalMs({ active, visible, failed: error !== null }),
    deps: [refresh, active, visible, error !== null],
  })

  const submit = async (sources: string[], options: SubmitOptions) => {
    setSubmitError(null)
    try {
      const response = await api.submit(sources, options)
      // Optimistic, so the rows appear on the same frame as the click rather
      // than up to a second later. The next tick replaces them wholesale.
      setQueue((previous) =>
        previous
          ? { ...previous, entries: [...previous.entries, ...response.entries] }
          : previous
      )
      await refresh()
    } catch (cause) {
      setSubmitError(cause instanceof Error ? cause.message : String(cause))
      throw cause
    }
  }

  /** Every row action is the same shape: disable the row, call, refresh, and
   * surface whatever sentence the server came back with. */
  const rowAction = (entry: QueueEntry, call: () => Promise<unknown>) => {
    setBusyEntryId(entry.entry_id)
    setSubmitError(null)
    void call()
      .catch((cause: unknown) =>
        setSubmitError(cause instanceof Error ? cause.message : String(cause))
      )
      .then(() => refresh())
      .finally(() => setBusyEntryId(null))
  }

  const entries = queue?.entries ?? []
  const runnerAlive = queue?.runner.alive ?? false
  const freeBytes = queue?.runner.free_bytes ?? null
  const lowDisk = freeBytes !== null && freeBytes < LOW_DISK_BYTES

  return (
    <div className="flex flex-col gap-7">
      <header className="page-heading">
        <div>
          <p className="eyebrow">YOUR CREATIVE WORKSPACE</p>
          <h1>Make more from every video.</h1>
          <p>
            Find the best moments. Shape your story. Create your next short.
          </p>
        </div>
        <Button asChild variant="outline">
          <Link to="/jobs">
            <Library size={16} /> Open library
          </Link>
        </Button>
      </header>
      <div className="stats-grid">
        <StatCard
          label="In queue"
          value={
            queue ? entries.filter((e) => e.status === "queued").length : "—"
          }
          detail="Waiting to start"
          icon={Clock3}
        />
        <StatCard
          label="Processing"
          value={
            queue ? entries.filter((e) => e.status === "running").length : "—"
          }
          detail="Finding your next moments"
          icon={LoaderCircle}
        />
        <StatCard
          label="Completed"
          value={
            queue ? entries.filter((e) => e.status === "done").length : "—"
          }
          detail="Finished queue entries"
          icon={CheckCheck}
        />
        <StatCard
          label="Needs attention"
          value={
            queue
              ? entries.filter((e) =>
                  ["failed", "blocked", "interrupted"].includes(e.status)
                ).length
              : "—"
          }
          detail="Check details below"
          icon={AlertTriangleIcon}
        />
      </div>
      <div className="creation-grid">
        <section aria-labelledby="submit-heading" className="studio-panel">
          <div className="panel-heading">
            <div>
              <h2 id="submit-heading">Create new clips</h2>
              <p>Start with a video. We will find the highlights.</p>
            </div>
            <span className="panel-icon">
              <Plus size={18} />
            </span>
          </div>
          <div className="panel-body">
            <SubmitBox
              onSubmit={submit}
              error={submitError}
              onDismissError={() => setSubmitError(null)}
            />
          </div>
        </section>
        <aside className="workflow-panel" aria-label="How it works">
          <p className="eyebrow">FROM SOURCE TO SHORT</p>
          <h2>Three steps. More possibilities.</h2>
          {[
            {
              title: "Add your video",
              text: "Paste a YouTube link or a local video path to get started.",
            },
            {
              title: "Let the moments surface",
              text: "Follow the queue as your video becomes a collection of short clips.",
            },
            {
              title: "Make the final call",
              text: "Preview each clip, keep your favorites, and download your edits.",
            },
          ].map((step, i) => (
            <div className="workflow-step" key={step.title}>
              <span>{String(i + 1).padStart(2, "0")}</span>
              <div>
                <h3>{step.title}</h3>
                <p>{step.text}</p>
              </div>
            </div>
          ))}
          <p className="workflow-footnote">
            <HardDriveIcon size={14} /> Your queue is saved between sessions.
          </p>
        </aside>
      </div>

      <section aria-labelledby="queue-heading" className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="queue-heading" className="section-title">
            Processing queue
            {entries.length > 0 && (
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                {entries.length}
              </span>
            )}
          </h2>
          <p className="flex items-center gap-3 text-xs text-muted-foreground">
            {!visible && (
              <span className="flex items-center gap-1">
                <PauseIcon className="size-3" />
                paused while this tab is hidden
              </span>
            )}
            <span className="flex items-center gap-1">
              <HardDriveIcon className="size-3" />
              {bytes(freeBytes)} free
            </span>
          </p>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertTriangleIcon />
            <AlertTitle>Cannot reach the server</AlertTitle>
            <AlertDescription>
              {error} Retrying every few seconds; the queue on disk is
              unaffected.
            </AlertDescription>
          </Alert>
        )}

        {queue && !runnerAlive && (
          <Alert>
            <AlertTriangleIcon />
            <AlertTitle>Processing is offline</AlertTitle>
            <AlertDescription>
              No supervisor is attached to this server, so entries will sit at{" "}
              <code className="font-mono">queued</code> indefinitely. Start it
              with <code className="font-mono">clipper ui</code>, or{" "}
              <code className="font-mono">clipper review --run-queue</code>.
            </AlertDescription>
          </Alert>
        )}

        {lowDisk && (
          <Alert>
            <HardDriveIcon />
            <AlertTitle>Low disk space</AlertTitle>
            <AlertDescription>
              {bytes(freeBytes)} free. A 1440p source plus its intermediate
              renders is several gigabytes, and running out mid-encode leaves a
              half-written mp4.
            </AlertDescription>
          </Alert>
        )}

        {!queue && !error && (
          <p role="status" className="text-sm text-muted-foreground">
            Loading your queue…
          </p>
        )}
        {queue && (
          <QueueList
            entries={entries}
            events={events}
            eventsEntryId={eventsEntryId}
            now={now}
            clipTotal={clipTotal}
            runnerAlive={runnerAlive}
            busyEntryId={busyEntryId}
            onCancel={(entry) =>
              rowAction(entry, () => api.cancel(entry.entry_id))
            }
            onRetry={(entry) =>
              rowAction(entry, () => api.retry(entry.entry_id))
            }
            onDelete={(entry) =>
              rowAction(entry, () => api.dequeue(entry.entry_id))
            }
          />
        )}
      </section>
    </div>
  )
}
