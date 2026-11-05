import { useState } from "react"
import { Link } from "react-router-dom"
import {
  Library,
  Search,
  Plus,
  CheckCheck,
  Scissors,
  Clock3,
} from "lucide-react"
import { JobCard } from "@/components/JobCard"
import { StatCard } from "@/components/StatCard"
import { Button } from "@/components/ui/button"
import { useJobData } from "@/hooks/useJobData"
import { getJobs } from "@/lib/api"

export function JobsPage() {
  const { data, error } = useJobData(getJobs, () => 5000)
  const [search, setSearch] = useState("")
  const [filter, setFilter] = useState("all")
  const jobs = data?.jobs ?? []
  const filtered = jobs.filter(
    (job) =>
      [job.title, job.channel_title, job.video_id]
        .join(" ")
        .toLowerCase()
        .includes(search.toLowerCase()) &&
      (filter === "all" ||
        (filter === "review" ? job.counts.pending > 0 : job.counts.done))
  )
  return (
    <>
      <div className="page-heading">
        <div>
          <p className="eyebrow">YOUR CREATIVE WORKSPACE</p>
          <h1>Clip library</h1>
          <p>All your videos. Your next great clip, a little closer.</p>
        </div>
        <Button asChild>
          <Link to="/">
            <Plus size={16} /> New clips
          </Link>
        </Button>
      </div>
      <div className="stats-grid">
        <StatCard
          label="Source videos"
          value={data ? jobs.length : "—"}
          detail="In your workspace"
          icon={Library}
        />
        <StatCard
          label="Moments found"
          value={data ? jobs.reduce((n, j) => n + j.counts.total, 0) : "—"}
          detail="Across all your videos"
          icon={Scissors}
        />
        <StatCard
          label="Kept clips"
          value={data ? jobs.reduce((n, j) => n + j.counts.approved, 0) : "—"}
          detail="Selected by you"
          icon={CheckCheck}
        />
        <StatCard
          label="To review"
          value={data ? jobs.reduce((n, j) => n + j.counts.pending, 0) : "—"}
          detail="Waiting for your decision"
          icon={Clock3}
        />
      </div>
      <div className="library-toolbar">
        <div className="filter-tabs" role="group" aria-label="Filter videos">
          {[
            { value: "all", label: "All videos" },
            { value: "review", label: "To review" },
            { value: "done", label: "Reviewed" },
          ].map((item) => (
            <button
              key={item.value}
              type="button"
              aria-pressed={filter === item.value}
              onClick={() => setFilter(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <label className="search-field">
          <Search size={16} />
          <input
            aria-label="Search videos"
            placeholder="Search your videos…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
      </div>
      {error && (
        <p
          role="alert"
          className="rounded-lg border border-destructive/20 bg-cut-bg p-4 text-sm text-destructive"
        >
          {error}
        </p>
      )}
      {!data && !error && (
        <p role="status" className="text-sm text-muted-foreground">
          Loading your library…
        </p>
      )}
      {data && !filtered.length && (
        <div className="studio-empty">
          <div className="empty-symbol">
            <Library size={24} />
          </div>
          <h2>
            {jobs.length ? "No matching videos" : "Your next story starts here"}
          </h2>
          <p>
            {jobs.length
              ? "Try a different search or filter."
              : "Add your first video and find the moments that matter."}
          </p>
          {!jobs.length && (
            <Button asChild>
              <Link to="/">Create your first clips</Link>
            </Button>
          )}
        </div>
      )}
      <div className="grid gap-5 md:grid-cols-2 2xl:grid-cols-3">
        {filtered.map((job) => (
          <JobCard key={job.video_id} job={job} />
        ))}
      </div>
    </>
  )
}
