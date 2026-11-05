import { useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { RubricBars } from "@/components/RubricBars"
import { VerdictButtons } from "@/components/VerdictButtons"
import { VideoPreview } from "@/components/VideoPreview"
import { downloadUrl, reviewClip } from "@/lib/api"
import type { Clip } from "@/lib/types"

export function ClipCard({
  clip,
  jobId,
  writable,
  rightsNote,
  onReview,
  onRender,
}: {
  clip: Clip
  jobId: string
  writable: boolean
  rightsNote: string | null
  onReview: (clip: Clip) => void
  onRender?: () => void
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const saving = useRef(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const note = draft ?? clip.review_note
  async function verdict(approved: boolean | null) {
    if (!writable || saving.current) return
    saving.current = true
    setBusy(true)
    setError(null)
    setSaved(false)
    try {
      const result = await reviewClip(jobId, clip.clip_id, approved, note)
      onReview(result.clip)
      setDraft(null)
      setSaved(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      saving.current = false
      setBusy(false)
    }
  }
  return (
    <article
      id={clip.clip_id}
      data-clip-card
      tabIndex={0}
      className="rounded-xl border bg-card p-5 outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="mb-1 font-mono text-xs text-muted-foreground">
            {clip.clip_id} · {clip.span_label} · {clip.duration_label}
          </p>
          <h3 className="text-lg font-semibold">
            {clip.suggested_title || `Clip ${clip.index}`}
          </h3>
        </div>
        <div className="flex gap-2">
          <Badge variant="secondary">Score {clip.score_label}</Badge>
          <Badge variant={clip.approved === false ? "destructive" : "outline"}>
            {clip.verdict}
          </Badge>
        </div>
      </header>
      {onRender && <Button className="mb-4" variant="outline" onClick={onRender}>Render this clip</Button>}
      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-3">
          {clip.preview ? (
            <VideoPreview key={clip.preview.url} preview={clip.preview} />
          ) : (
            <p className="rounded-lg bg-muted p-8 text-center text-sm">
              No preview available yet.
            </p>
          )}
          {clip.render_note && (
            <p className="text-sm text-muted-foreground">{clip.render_note}</p>
          )}
          {clip.rendered && writable && (
            <Button asChild variant="outline">
              <a href={downloadUrl(jobId, clip.clip_id)}>Download MP4</a>
            </Button>
          )}
          {rightsNote && (
            <p className="text-sm text-warn">Rights override: {rightsNote}</p>
          )}
        </div>
        <div className="space-y-4">
          {clip.reason && <p className="text-sm">{clip.reason}</p>}
          {clip.hook_rewrite && (
            <p className="text-sm">
              <strong>Hook: </strong>
              {clip.hook_rewrite}
            </p>
          )}
          <RubricBars axes={clip.rubric} />
          <details>
            <summary className="cursor-pointer text-sm font-medium">
              Transcript
            </summary>
            <p className="mt-2 text-sm whitespace-pre-wrap text-muted-foreground">
              {clip.transcript || "Transcript unavailable."}
            </p>
          </details>
          <div className="space-y-2">
            <label
              htmlFor={`note-${clip.clip_id}`}
              className="text-sm font-medium"
            >
              Review note
            </label>
            <Textarea
              id={`note-${clip.clip_id}`}
              value={note}
              disabled={!writable || busy}
              onChange={(event) => {
                setDraft(event.target.value)
                setSaved(false)
              }}
              placeholder="What makes this moment worth keeping or cutting?"
            />
          </div>
          <VerdictButtons
            approved={clip.approved}
            disabled={!writable || busy}
            onVerdict={(value) => void verdict(value)}
          />
          <Button
            variant="ghost"
            disabled={!writable || busy || note === clip.review_note}
            onClick={() => void verdict(clip.approved)}
          >
            Save note
          </Button>
          {busy && (
            <p role="status" className="text-xs">
              Saving…
            </p>
          )}
          {saved && (
            <p role="status" className="text-xs text-keep">
              Review saved.
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
      </div>
    </article>
  )
}
