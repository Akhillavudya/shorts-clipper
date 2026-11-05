/**
 * Run an async callback on a timer that reschedules itself.
 *
 * `setTimeout` chained after each completed tick rather than `setInterval`,
 * and that is the load-bearing detail: `setInterval` queues a new request
 * whether or not the last one came back, so a laptop that slept for an hour
 * wakes to a burst of stacked fetches against a server that may not be there.
 * A chain cannot overlap itself.
 *
 * The interval is recomputed after every tick from whatever the callback
 * learned, so "something started running" speeds the page up on the next tick
 * without any component having to say so.
 */

import * as React from "react"

export interface PollingOptions {
  /** Milliseconds until the next tick, or `null` to stop. Called after every
   * completed tick, so it sees the state the tick produced. */
  intervalMs: () => number | null
  /** Re-run immediately when these change, cancelling any pending timer. */
  deps?: React.DependencyList
}

/** Whether the document is currently visible, as state.
 *
 * Exposed separately because the page shows it: a queue that is not polling
 * should not pretend its numbers are live.
 */
export function useDocumentVisible(): boolean {
  const [visible, setVisible] = React.useState(
    () => typeof document === "undefined" || document.visibilityState === "visible"
  )

  React.useEffect(() => {
    const onChange = () => setVisible(document.visibilityState === "visible")
    document.addEventListener("visibilitychange", onChange)
    return () => document.removeEventListener("visibilitychange", onChange)
  }, [])

  return visible
}

export function usePolling(tick: () => Promise<void>, options: PollingOptions): void {
  const { intervalMs, deps = [] } = options

  // Held in refs so that changing the callback - which happens on every render,
  // because it closes over state - does not cancel an in-flight timer and
  // restart the whole chain.
  const tickRef = React.useRef(tick)
  const intervalRef = React.useRef(intervalMs)

  // Written in a layout effect rather than during render. Assigning to a ref
  // while rendering is a tearing hazard under concurrent React - the render
  // that assigned it may be thrown away - and the commit phase is early
  // enough here, because nothing reads these until a timer fires.
  React.useLayoutEffect(() => {
    tickRef.current = tick
    intervalRef.current = intervalMs
  })

  React.useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const schedule = () => {
      if (cancelled) return
      const wait = intervalRef.current()
      if (wait === null) return
      timer = setTimeout(run, wait)
    }

    const run = async () => {
      if (cancelled) return
      try {
        await tickRef.current()
      } finally {
        schedule()
      }
    }

    void run()

    return () => {
      cancelled = true
      if (timer !== undefined) clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

/** A monotonically updating clock, for elapsed tickers.
 *
 * Separate from the poll loop because the two want different rates: the
 * elapsed time on a running job should tick every second even when the queue
 * itself is only checked every five, and a ticker that advances in five-second
 * jumps reads as a stalled page.
 */
export function useNow(active: boolean, everyMs = 1000): number {
  const [now, setNow] = React.useState(() => Date.now())

  React.useEffect(() => {
    if (!active) return undefined
    const timer = setInterval(() => setNow(Date.now()), everyMs)
    return () => clearInterval(timer)
  }, [active, everyMs])

  return now
}
