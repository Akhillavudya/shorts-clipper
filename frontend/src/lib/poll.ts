/**
 * The polling rules, as pure functions (plan section 2).
 *
 * Polling rather than SSE, and that choice is the plan's: an EventSource that
 * the browser reconnects on its own, against a Flask dev server with one
 * thread per connection, is a thread-leak class of bug for a page that is
 * meant to be left open all afternoon. A `GET /api/queue?since=` on a timer is
 * boring, cancellable, and correct after a laptop sleeps.
 *
 * Separated from the hook so the interesting decisions - how often, and how to
 * splice a page of events onto the ones already on screen - are testable and
 * readable without React in the way.
 */

import type { ProgressEvent } from "@/lib/types"

/** While something is running. Fast enough that a stage change feels
 * immediate; the request is a few hundred bytes over loopback. */
export const ACTIVE_MS = 1000

/** Nothing running. The queue only changes when you change it, so this is
 * really just "notice that someone started a job in another tab". */
export const IDLE_MS = 5000

/** After a failed request. The server is probably gone - `clipper ui` is a
 * terminal someone can ctrl-c - and hammering a dead port five times a second
 * fills the console with noise for no gain. */
export const ERROR_MS = 4000

/** How many events one entry keeps on screen. The server caps its own log at
 * a couple of thousand; this caps the DOM. The tail is what matters, so the
 * head is what gets dropped. */
export const MAX_EVENTS_KEPT = 300

export interface PollState {
  /** Is anything queued or running? */
  active: boolean
  /** Is the tab visible? A hidden tab polls nothing at all. */
  visible: boolean
  /** Did the last request throw? */
  failed: boolean
}

/** How long to wait before the next tick, or `null` to stop entirely.
 *
 * `null` for a hidden tab is the whole point: a page left open for six hours
 * behind other windows should cost nothing, and the first tick after it comes
 * back is immediate rather than waiting out an interval.
 */
export function pollIntervalMs(state: PollState): number | null {
  if (!state.visible) return null
  if (state.failed) return ERROR_MS
  return state.active ? ACTIVE_MS : IDLE_MS
}

/** Splice a page of events onto what is already on screen.
 *
 * Two rules, both of which come from the cursor being per-entry:
 *
 * - **A changed entry id discards everything.** Sequence numbers restart at 1
 *   for each entry, so carrying events across a change would interleave two
 *   jobs into one incoherent list.
 * - **Within one entry, `seq` deduplicates.** A tick that overlaps the
 *   previous one - which a retried request or a slow response will do -
 *   re-delivers events the page already has.
 */
export function mergeEvents(
  previous: ProgressEvent[],
  previousEntryId: string | null,
  incoming: ProgressEvent[],
  entryId: string | null
): ProgressEvent[] {
  if (entryId === null) return []
  const base = previousEntryId === entryId ? previous : []
  if (incoming.length === 0) return base

  const seen = new Set(base.map((event) => event.seq))
  const merged = base.concat(incoming.filter((event) => !seen.has(event.seq)))
  merged.sort((a, b) => a.seq - b.seq)
  return merged.length > MAX_EVENTS_KEPT ? merged.slice(-MAX_EVENTS_KEPT) : merged
}

/** The cursor for the next request.
 *
 * Zero whenever the focused entry changed, for the reason above: the server
 * would happily answer `since=57` for an entry whose log only reaches 12, and
 * the page would show an empty feed for a job that is plainly running.
 */
export function nextSince(
  previousEntryId: string | null,
  entryId: string | null,
  seq: number
): number {
  return previousEntryId === entryId ? seq : 0
}
