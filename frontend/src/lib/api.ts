/**
 * The one place that talks to Flask.
 *
 * Relative URLs throughout, in both modes: built, the bundle is served by the
 * same Flask process on :5151; in development, Vite's proxy forwards `/api`
 * to it from :5173. Same origin either way, which is why this project has no
 * CORS code and no CORS dependency (plan section 6).
 */

import type {
  DoctorResponse,
  Clip,
  Counts,
  JobView,
  JobsResponse,
  QueueEntry,
  QueueResponse,
  SubmitOptions,
  SubmitResponse,
} from "@/lib/types"

/** A non-2xx response, carrying the server's own sentence.
 *
 * The API's sentences are written to be shown - "allow_red cannot be set at
 * submit time", "no job runner is attached to this server" - so the UI
 * surfaces `message` verbatim rather than translating a status code into a
 * vaguer one of its own.
 */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    })
  } catch {
    // The server going away is the common case here, not a network blip:
    // `clipper ui` is a terminal the user can ctrl-c. Saying so beats
    // "Failed to fetch".
    throw new ApiError(
      0,
      "the clipper server is not responding - is `clipper ui` still running?"
    )
  }

  const text = await response.text()
  let payload: unknown = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = null
    }
  }

  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `${response.status} ${response.statusText}`
    throw new ApiError(response.status, message)
  }

  return payload as T
}

/** The whole queue plus a page of events, in one request.
 *
 * `since` is a cursor into *one* entry's log and `entry` chooses which. The
 * server restarts sequence numbers per entry, so a caller whose
 * `events_entry_id` changed between ticks must reset `since` to 0 -
 * `usePolling` does exactly that.
 */
export function getQueue(
  options: { since?: number; entry?: string; limit?: number } = {}
): Promise<QueueResponse> {
  const params = new URLSearchParams()
  if (options.since) params.set("since", String(options.since))
  if (options.entry) params.set("entry", options.entry)
  if (options.limit) params.set("limit", String(options.limit))
  const query = params.toString()
  return request<QueueResponse>(`/api/queue${query ? `?${query}` : ""}`)
}

/** Paste one or more sources. 202: nothing has run yet, it has been queued. */
export function submit(
  sources: string[],
  options: SubmitOptions = {}
): Promise<SubmitResponse> {
  return request<SubmitResponse>("/api/queue", {
    method: "POST",
    body: JSON.stringify({ sources, options }),
  })
}

/** Drop a queued entry. 409 while it is running - cancel it first. */
export function dequeue(entryId: string): Promise<{ entry: QueueEntry }> {
  return request<{ entry: QueueEntry }>(
    `/api/queue/${encodeURIComponent(entryId)}`,
    {
      method: "DELETE",
    }
  )
}

/** Ask an entry to stop. 202, because a running worker exits at its next
 * stage boundary and the tree-kill is ten seconds behind that. */
export function cancel(entryId: string): Promise<{ entry: QueueEntry }> {
  return request<{ entry: QueueEntry }>(
    `/api/queue/${encodeURIComponent(entryId)}/cancel`,
    {
      method: "POST",
    }
  )
}

/** Put a finished entry back in the queue, keeping its id and its history.
 *
 * Without `force` this resumes: `should_run` skips completed stages, which is
 * the difference between retrying an interrupted render and re-downloading two
 * gigabytes to find nothing was wrong with them.
 */
export function retry(
  entryId: string,
  options: { force?: boolean; from_stage?: string | null } = {}
): Promise<{ entry: QueueEntry }> {
  return request<{ entry: QueueEntry }>(
    `/api/queue/${encodeURIComponent(entryId)}/retry`,
    {
      method: "POST",
      body: JSON.stringify({
        force: options.force ?? false,
        from_stage: options.from_stage ?? null,
      }),
    }
  )
}

export function getJobs(): Promise<JobsResponse> {
  return request<JobsResponse>("/api/jobs")
}

export function getJob(id: string): Promise<JobView> {
  return request<JobView>(`/api/jobs/${encodeURIComponent(id)}`)
}

export function reviewClip(
  id: string,
  clipId: string,
  approved: boolean | null,
  note: string
) {
  return request<{ clip: Clip; counts: Counts }>(
    `/api/jobs/${encodeURIComponent(id)}/clips/${encodeURIComponent(clipId)}/review`,
    { method: "POST", body: JSON.stringify({ approved, note }) }
  )
}

export function downloadUrl(id: string, clipId?: string): string {
  const base = `/api/jobs/${encodeURIComponent(id)}`
  return clipId
    ? `${base}/clips/${encodeURIComponent(clipId)}/download`
    : `${base}/download`
}

export function getDoctor(): Promise<DoctorResponse> {
  return request<DoctorResponse>("/api/doctor")
}

export function classifyRights(id: string, status: string, reason: string, remember: boolean) {
  return request<{ status: string; resumed: boolean }>(`/api/jobs/${encodeURIComponent(id)}/rights`, {
    method: "POST", body: JSON.stringify({ status, reason, remember }),
  })
}

export function renderJob(id: string, options: {
  approved_only?: boolean; clip_index?: number; force?: boolean
  allow_red?: boolean; override_reason?: string
} = {}) {
  return request<{ entry: QueueEntry }>(`/api/jobs/${encodeURIComponent(id)}/render`, {
    method: "POST", body: JSON.stringify(options),
  })
}
