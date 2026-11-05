import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import type { Preview } from "@/lib/types"

export function VideoPreview({ preview }: { preview: Preview }) {
  const frame = useRef<HTMLDivElement>(null)
  const [mounted, setMounted] = useState(false)
  const [error, setError] = useState(false)
  useEffect(() => {
    const node = frame.current
    if (!node || !("IntersectionObserver" in window)) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setMounted(true)
          observer.disconnect()
        }
      },
      { rootMargin: "200px" }
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return (
    <div ref={frame} className="flex flex-col gap-2">
      <div className="flex min-h-44 items-center justify-center overflow-hidden rounded-lg bg-black">
        {mounted ? (
          <video
            className="max-h-[28rem] w-full"
            controls
            playsInline
            preload="metadata"
            src={preview.url}
            aria-label={preview.label}
            onLoadedMetadata={(event) => {
              event.currentTarget.currentTime = preview.start
            }}
            onPlay={(event) => {
              const video = event.currentTarget
              if (
                video.currentTime < preview.start ||
                video.currentTime >= preview.end
              )
                video.currentTime = preview.start
              document.querySelectorAll("video").forEach((other) => {
                if (other !== video) other.pause()
              })
            }}
            onTimeUpdate={(event) => {
              const video = event.currentTarget
              if (
                preview.end > preview.start &&
                video.currentTime >= preview.end
              ) {
                video.pause()
                video.currentTime = preview.start
              }
            }}
            onError={() => setError(true)}
          />
        ) : (
          <Button
            variant="secondary"
            data-mount-preview
            onClick={() => setMounted(true)}
          >
            Load preview
          </Button>
        )}
      </div>
      <p className="text-xs text-muted-foreground">{preview.label}</p>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          Preview could not be loaded. The media file may be missing or
          unsupported.
        </p>
      )}
    </div>
  )
}
