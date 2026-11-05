import { useCallback, useRef, useState } from "react"
import { Link, useParams } from "react-router-dom"
import { ClipCard } from "@/components/ClipCard"
import { ProgressLine } from "@/components/ProgressLine"
import { RightsPanel } from "@/components/RightsPanel"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useJobData } from "@/hooks/useJobData"
import { useNow } from "@/hooks/usePolling"
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts"
import { downloadUrl, getJob, getQueue, renderJob, retry } from "@/lib/api"
import type { Clip, ProgressEvent } from "@/lib/types"

export function JobPage() {
  const { id = "" } = useParams()
  return <JobDetail key={id} id={id} />
}

function JobDetail({ id }: { id: string }) {
  const [revision, setRevision] = useState(0)
  const refresh = () => setRevision((value) => value + 1)
  const [actionBusy, setActionBusy] = useState(false)
  const [actionError, setActionError] = useState("")
  const [actionMessage, setActionMessage] = useState("")
  const [fromStage, setFromStage] = useState("")
  const [forceRender, setForceRender] = useState(false)
  const load = useCallback(async () => {
    void revision
    const job = await getJob(id)
    let events: ProgressEvent[] = []
    if (job.entry?.status === "running") {
      const queue = await getQueue({ entry: job.entry.entry_id, limit: 2000 })
      events = queue.events
    }
    return { job, events }
  }, [id, revision])
  const { data, error } = useJobData(load, (value) =>
    value.job.writable ? 5000 : 1000
  )
  const [reviews, setReviews] = useState<Record<string, Clip>>({})
  const root = useRef<HTMLDivElement>(null)
  useKeyboardShortcuts(root)
  const job = data?.job
  const active = job?.entry?.status === "running" || job?.entry?.status === "queued"
  const disabled = actionBusy || active || !job?.writable || !!error
  async function action(run: () => Promise<unknown>) {
    if (disabled) return
    setActionBusy(true); setActionError(""); setActionMessage("")
    try { await run(); setActionMessage("Job queued. Progress will appear here."); refresh() }
    catch (err) { setActionError(err instanceof Error ? err.message : String(err)) }
    finally { setActionBusy(false) }
  }
  const now = useNow(job?.entry?.status === "running")
  const clips =
    job?.clips.map((clip) => {
      const saved = reviews[clip.clip_id]
      return saved && (saved.reviewed_at ?? "") > (clip.reviewed_at ?? "")
        ? saved
        : clip
    }) ?? []
  const approved = clips.filter((clip) => clip.approved === true).length
  const rejected = clips.filter((clip) => clip.approved === false).length
  return (
    <div ref={root} className="flex flex-col gap-6">
      <Link to="/jobs" className="text-sm text-primary hover:underline">
        ← Back to library
      </Link>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
      {!job && !error && <p role="status">Loading job…</p>}
      {job && (
        <>
          <header className="review-heading space-y-3">
            <p className="eyebrow">CLIP REVIEW</p>
            <h1 className="text-3xl font-semibold tracking-tight break-words">
              {job.title || job.video_id}
            </h1>
            <p className="text-sm text-muted-foreground">
              {job.channel_title || "Local source"} · {job.duration_label}
            </p>
            <div className="flex flex-wrap gap-2">
              {job.entry && <a className="text-sm text-primary underline" href={`/api/jobs/${encodeURIComponent(id)}/log?tail=200`} target="_blank" rel="noreferrer">View worker log</a>}
              <Badge variant="outline">Rights: {job.rights_status}</Badge>
              {job.rights_override && (
                <Badge variant="outline">OVERRIDDEN</Badge>
              )}
              {job.source_url && /^https?:\/\//i.test(job.source_url) && (
                <a
                  className="text-sm text-primary underline"
                  href={job.source_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  View source
                </a>
              )}
            </div>
            {job.rights_note && (
              <p className="text-sm text-muted-foreground">{job.rights_note}</p>
            )}
            {job.rights_override_reason && <p className="text-sm text-warn">Override reason: {job.rights_override_reason}</p>}
            {!job.writable && (
              <p
                role="status"
                className="rounded-lg bg-warn-bg p-3 text-sm text-warn"
              >
                Verdicts and downloads unlock when this job finishes.
              </p>
            )}
            {job.warnings.map((warning) => (
              <p key={warning} className="text-sm text-warn">
                {warning}
              </p>
            ))}
          </header>
          {!job.exportable ? <RightsPanel job={job} disabled={disabled} onChanged={refresh} /> : (
            <details className="rounded-xl border bg-card p-4">
              <summary className="min-h-11 cursor-pointer text-sm font-medium">Change source rights</summary>
              <RightsPanel job={job} disabled={disabled} onChanged={refresh} />
            </details>
          )}
          <section className="space-y-3 rounded-xl border bg-card p-5" aria-label="Job actions">
            <h2 className="font-semibold">Continue or render</h2>
            {active && <p role="status" className="text-sm">Job {job.entry?.status}. Controls unlock when it finishes.</p>}
            {job.entry && !active && <div className="flex flex-wrap items-center gap-2">
              <select aria-label="Retry from stage" className="min-h-11 rounded-md border bg-background px-3 text-sm" value={fromStage} disabled={disabled} onChange={(e) => setFromStage(e.target.value)}>
                <option value="">Resume unfinished stages</option>
                {job.stages.filter((stage) => !job.entry?.options.render_only || stage.name === "render").map((stage) => <option key={stage.name} value={stage.name}>Restart from {stage.name}</option>)}
              </select>
              <Button variant="outline" disabled={disabled || !job.exportable} onClick={() => void action(() => retry(job.entry!.entry_id, { from_stage: fromStage || null }))}>Retry job</Button>
            </div>}
            {clips.length > 0 && <>
              <label className="flex min-h-11 items-center gap-2 text-sm"><input type="checkbox" checked={forceRender} disabled={disabled} onChange={(e) => setForceRender(e.target.checked)} />Rebuild finished renders</label>
              <div className="flex flex-wrap gap-2">
                <Button disabled={disabled || !job.exportable || !approved} onClick={() => void action(() => renderJob(id, { approved_only: true, force: forceRender }))}>Render kept clips</Button>
                <Button variant="outline" disabled={disabled || !job.exportable || rejected === clips.length} onClick={() => void action(() => renderJob(id, { force: forceRender }))}>Render all except cut</Button>
              </div>
            </>}
            {actionError && <p role="alert" className="text-sm text-destructive">{actionError}</p>}
            {actionMessage && <p role="status" className="text-sm">{actionMessage}</p>}
          </section>
          <div
            className="flex flex-wrap gap-2 rounded-xl border bg-card p-4"
            aria-label="Pipeline stages"
          >
            {job.stages.map((stage) => (
              <Badge
                key={stage.name}
                variant={
                  stage.status === "failed" ? "destructive" : "secondary"
                }
              >
                {stage.name}: {stage.status}
              </Badge>
            ))}
          </div>
          {job.stages
            .filter((stage) => stage.error)
            .map((stage) => (
              <p key={stage.name} className="text-sm text-destructive">
                {stage.name}: {stage.error}
              </p>
            ))}
          {job.entry?.status === "running" && (
            <ProgressLine
              entry={job.entry}
              events={data.events}
              now={now}
              clipTotal={job.counts.total}
            />
          )}
          <div className="review-toolbar flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card p-5">
            <div>
              <p className="font-medium">
                {approved} kept · {rejected} cut ·{" "}
                {clips.length - approved - rejected} to review
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                J / K navigate · A keep · R cut · U undo · Space play / pause
              </p>
            </div>
            {job.writable && clips.some((clip) => clip.rendered) && (
              <Button asChild>
                <a href={downloadUrl(id)}>Download all (.zip)</a>
              </Button>
            )}
          </div>
          {!clips.length && (
            <p className="studio-empty text-sm text-muted-foreground">
              No candidate clips yet. They appear after the moments stage
              finishes.
            </p>
          )}
          {clips.map((clip) => (
            <ClipCard
              key={clip.clip_id}
              clip={clip}
              jobId={id}
              writable={job.writable && !error}
              rightsNote={job.rights_override ? job.rights_override_reason || job.rights_note : null}
              onRender={job.exportable && !disabled ? () => void action(() => renderJob(id, { clip_index: clip.index, force: forceRender })) : undefined}
              onReview={(saved) =>
                setReviews((previous) => ({
                  ...previous,
                  [saved.clip_id]: saved,
                }))
              }
            />
          ))}
        </>
      )}
    </div>
  )
}
