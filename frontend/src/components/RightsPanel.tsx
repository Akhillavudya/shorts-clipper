import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { classifyRights, renderJob } from "@/lib/api"
import type { JobView, RightsStatus } from "@/lib/types"

export function RightsPanel({ job, disabled, onChanged }: {
  job: JobView; disabled: boolean; onChanged: () => void
}) {
  const [status, setStatus] = useState<RightsStatus>(job.rights_status)
  const [reason, setReason] = useState("")
  const [remember, setRemember] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const [message, setMessage] = useState("")
  const [confirm, setConfirm] = useState(false)
  const ready = !disabled && !busy && reason.trim().length >= 10
  async function save(override: boolean) {
    if (!ready) return
    setBusy(true); setError(""); setMessage("")
    try {
      if (override) {
        await renderJob(job.video_id, { allow_red: true, override_reason: reason.trim() })
        setMessage("Override recorded. Job queued.")
      } else {
        const result = await classifyRights(job.video_id, status, reason.trim(), remember)
        setMessage(result.resumed ? "Rights saved. Job queued to resume." : "Rights saved.")
      }
      setConfirm(false)
      onChanged()
    } catch (err) { setError(err instanceof Error ? err.message : String(err)) }
    finally { setBusy(false) }
  }
  return (
    <section className="space-y-4 rounded-xl border bg-card p-5" aria-label="Source rights">
      <div>
        <h2 className="text-lg font-semibold">Source rights</h2>
        <p className="text-sm text-muted-foreground">Current status: {job.rights_status}</p>
        <p className="mt-2 whitespace-pre-wrap text-sm">{job.rights_note}</p>
        <p className="mt-2 text-xs text-muted-foreground break-all">
          {job.channel_title || "Local source"} · Channel: {job.channel_id || "Not available"}
          <br />Licence: {job.licence || "Not reported"}
        </p>
      </div>
      <label className="block space-y-2 text-sm">
        <span>Classify this job</span>
        <select value={status} disabled={disabled || busy} onChange={(e) => setStatus(e.target.value as RightsStatus)} className="min-h-11 w-full rounded-md border bg-background px-3">
          <option value="unknown">Unknown — not cleared</option>
          <option value="green">Green — your own footage; a public clipping policy; written permission; YouTube CC-BY licence; US federal works; public domain.</option>
          <option value="amber">Amber — a creator who said yes by email.</option>
          <option value="red">Red — creators who already run official clip channels.</option>
        </select>
      </label>
      <label className="block space-y-2 text-sm">
        <span>Reason or permission evidence (at least 10 characters)</span>
        <Textarea value={reason} onChange={(e) => setReason(e.target.value)} disabled={disabled || busy} />
      </label>
      <label className="flex min-h-11 items-center gap-2 text-sm">
        <input type="checkbox" checked={remember} disabled={!job.channel_id || disabled || busy} onChange={(e) => setRemember(e.target.checked)} />
        Remember this channel in the watchlist
      </label>
      {status === "amber" && <p className="text-xs text-muted-foreground">Save the permission email under src/clipper/rights/permissions and reference it in the reason.</p>}
      <p className="text-xs text-muted-foreground">YouTube’s filter must be “Creative Commons — Attribution (CC BY)” specifically.</p>
      <div className="flex flex-wrap gap-2">
        <Button disabled={!ready} onClick={() => void save(false)}>Save classification</Button>
        {!job.exportable && <Button variant="outline" disabled={!ready} onClick={() => setConfirm(true)}>Render anyway…</Button>}
      </div>
      {confirm && <div className="space-y-3 rounded-lg bg-warn-bg p-4 text-sm text-warn" role="alert">
        <p>Reused-content enforcement is at the channel level; one bad batch suspends monetization for everything. Continue with the recorded reason?</p>
        <div className="flex gap-2">
          <Button variant="destructive" disabled={!ready} onClick={() => void save(true)}>Confirm override and continue</Button>
          <Button variant="outline" disabled={busy} onClick={() => setConfirm(false)}>Cancel</Button>
        </div>
      </div>}
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {message && <p role="status" className="text-sm">{message}</p>}
    </section>
  )
}
