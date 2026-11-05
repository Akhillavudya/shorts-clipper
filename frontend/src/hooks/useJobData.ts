import { useEffect, useRef, useState } from "react"
import { useDocumentVisible } from "@/hooks/usePolling"

/** Serial polling with cleanup: a late response cannot update another page. */
export function useJobData<T>(
  load: () => Promise<T>,
  interval: (data: T) => number
) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const visible = useDocumentVisible()
  const intervalRef = useRef(interval)
  useEffect(() => {
    intervalRef.current = interval
  }, [interval])
  useEffect(() => {
    if (!visible) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      let delay = 5000
      try {
        const next = await load()
        if (disposed) return
        setData(next)
        setError(null)
        delay = intervalRef.current(next)
      } catch (err) {
        if (!disposed)
          setError(err instanceof Error ? err.message : String(err))
      }
      if (!disposed) timer = setTimeout(poll, delay)
    }
    void poll()
    return () => {
      disposed = true
      clearTimeout(timer)
    }
  }, [load, visible])
  return { data, error }
}
