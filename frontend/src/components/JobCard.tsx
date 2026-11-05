import { Link } from "react-router-dom"
import { ArrowUpRight, Film, Scissors } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { JobSummary } from "@/lib/types"
export function JobCard({ job }: { job: JobSummary }) {
  return (
    <Link
      to={`/jobs/${encodeURIComponent(job.video_id)}`}
      className="library-card"
    >
      <div className="library-card-cover">
        <div className="film-motif" aria-hidden="true">
          <Film size={36} strokeWidth={1} />
        </div>
        <span className="cover-caption">SOURCE VIDEO</span>
        <Badge
          className="cover-status"
          variant={job.stage_failed ? "destructive" : "secondary"}
        >
          {job.stage}
        </Badge>
        <span className="cover-open">
          <ArrowUpRight size={18} />
        </span>
      </div>
      <div className="library-card-body">
        <p className="eyebrow">{job.channel_title || "Local source"}</p>
        <h2>{job.title || job.video_id}</h2>
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">Rights: {job.rights_status}</Badge>
          {job.rights_override && <Badge variant="outline">Overridden</Badge>}
        </div>
        <div className="library-card-counts">
          <span>
            <Scissors size={14} /> {job.counts.total} moments
          </span>
          <span>
            {job.counts.approved} kept · {job.counts.pending} to review
          </span>
        </div>
        {job.updated_at && (
          <time
            className="text-xs text-muted-foreground"
            dateTime={job.updated_at}
          >
            Updated{" "}
            {new Date(job.updated_at).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </time>
        )}
      </div>
    </Link>
  )
}
