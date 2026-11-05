/**
 * The queue, newest work at the bottom - the order it will actually run in.
 *
 * Not sorted by status, and not with the running one pinned to the top. The
 * queue is a list of things that will happen in sequence, and reordering it
 * for visual convenience destroys the one piece of information it carries.
 */

import { ListChecksIcon } from "lucide-react"

import { QueueRow } from "@/components/QueueRow"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import type { ProgressEvent, QueueEntry } from "@/lib/types"

interface QueueListProps {
  entries: QueueEntry[]
  events: ProgressEvent[]
  /** Which entry `events` belongs to. Sequence numbers restart per entry, so
   * handing one entry's events to another row would be nonsense rather than
   * merely noise. */
  eventsEntryId: string | null
  now: number
  clipTotal: number | null
  runnerAlive: boolean
  busyEntryId: string | null
  onCancel: (entry: QueueEntry) => void
  onRetry: (entry: QueueEntry) => void
  onDelete: (entry: QueueEntry) => void
}

export function QueueList({
  entries,
  events,
  eventsEntryId,
  now,
  clipTotal,
  runnerAlive,
  busyEntryId,
  onCancel,
  onRetry,
  onDelete,
}: QueueListProps) {
  if (entries.length === 0) {
    return (
      <Empty className="studio-empty">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <ListChecksIcon />
          </EmptyMedia>
          <EmptyTitle>Ready for your first video</EmptyTitle>
          <EmptyDescription>
            Add a video above to begin. Your progress will appear here, one step
            at a time.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <ul className="flex flex-col gap-3">
      {entries.map((entry) => (
        <QueueRow
          key={entry.entry_id}
          entry={entry}
          events={eventsEntryId === entry.entry_id ? events : []}
          now={now}
          clipTotal={clipTotal}
          runnerAlive={runnerAlive}
          busy={busyEntryId === entry.entry_id}
          onCancel={onCancel}
          onRetry={onRetry}
          onDelete={onDelete}
        />
      ))}
    </ul>
  )
}
