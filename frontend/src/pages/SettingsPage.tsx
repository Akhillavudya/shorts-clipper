import { useCallback, useState } from "react"
import { Link } from "react-router-dom"
import { useJobData } from "@/hooks/useJobData"
import { getDoctor, request } from "@/lib/api"
import { Button } from "@/components/ui/button"

interface Storage { free_bytes: number | null; jobs: { video_id: string; bytes: number; source_bytes: number }[] }
interface Settings { config_path: string; jobs_root: string; values: Record<string, unknown> }
const size = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(2)} GB`

export function SettingsPage() {
  const [revision, setRevision] = useState(0)
  const [busy, setBusy] = useState("")
  const [message, setMessage] = useState("")
  const load = useCallback(async () => {
    void revision
    const [doctor, disk, settings] = await Promise.all([
      getDoctor(), request<Storage>("/api/disk"), request<Settings>("/api/settings"),
    ])
    return { doctor, disk, settings }
  }, [revision])
  const { data, error } = useJobData(load, () => 15000)
  async function remove(id: string) {
    if (!window.confirm("Delete this downloaded source? Finished clips stay available. Retry the pipeline to download it again before re-rendering.")) return
    setBusy(id); setMessage("")
    try {
      await request(`/api/jobs/${encodeURIComponent(id)}/source`, { method: "DELETE" })
      setMessage("Source deleted. Finished clips and review data were kept.")
      setRevision((v) => v + 1)
    } catch (err) { setMessage(err instanceof Error ? err.message : String(err)) }
    finally { setBusy("") }
  }
  return <div className="space-y-6">
    <header><p className="eyebrow">WORKSPACE</p><h1 className="text-3xl font-semibold">Settings & storage</h1></header>
    {error && <p role="alert" className="text-destructive">{error}</p>}
    {message && <p role="status">{message}</p>}
    {!data && !error && <p>Loading workspace checks…</p>}
    {data && <>
      <section className="rounded-xl border p-5 space-y-4"><h2 className="text-xl font-semibold">System checks</h2>
        {data.doctor.checks.map((check) => <div key={check.name} className="border-b pb-3">
          <p><strong>{check.name}</strong> · <span className={check.status === "fail" ? "text-destructive" : check.status === "warn" ? "text-warn" : "text-primary"}>{check.status}</span></p>
          <p className="text-sm break-words">{check.detail}</p>{check.hint && <p className="text-sm text-muted-foreground">{check.hint}</p>}
        </div>)}
      </section>
      <section className="rounded-xl border p-5 space-y-4"><h2 className="text-xl font-semibold">Storage</h2>
        <p>{data.disk.free_bytes === null ? "Free space unavailable" : `${size(data.disk.free_bytes)} free`}</p>
        <p className="text-sm text-muted-foreground">Deleting a downloaded source keeps finished clips. Local original files are never deleted.</p>
        {data.disk.jobs.map((job) => <div key={job.video_id} className="flex flex-wrap items-center justify-between gap-3 border-t pt-3">
          <Link className="text-primary underline" to={`/jobs/${encodeURIComponent(job.video_id)}`}>{job.video_id}</Link>
          <span>{size(job.bytes)} total · {size(job.source_bytes)} source</span>
          <Button variant="outline" disabled={!!busy || !job.source_bytes || !!error} onClick={() => void remove(job.video_id)}>{busy === job.video_id ? "Deleting…" : "Delete source"}</Button>
        </div>)}
      </section>
      <section className="rounded-xl border p-5 space-y-3"><h2 className="text-xl font-semibold">Configuration</h2>
        <p className="text-sm break-all">{data.settings.config_path}</p><p className="text-sm break-all">Jobs: {data.settings.jobs_root}</p>
        <p className="text-sm text-muted-foreground">Read only. Edit the YAML file and restart the server to apply operational settings.</p>
        {Object.entries(data.settings.values).map(([key, value]) => <p key={key}><code>{key}</code>: {String(value)}</p>)}
      </section>
    </>}
  </div>
}
