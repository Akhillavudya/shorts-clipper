/**
 * The route table.
 *
 * `BrowserRouter` rather than a hash router, which is what makes the Flask
 * catch-all in `webui/__init__.py` load-bearing: a deep link typed into the
 * address bar is a real request to Flask, and Flask has to answer it with
 * `index.html` for the router to see it at all.
 *
 * Queue, jobs and clip review are implemented. Phase 6 adds `/settings`.
 */

import { Navigate, Route, Routes } from "react-router-dom"

import { AppShell } from "@/components/AppShell"
import { QueuePage } from "@/pages/QueuePage"
import { JobsPage } from "@/pages/JobsPage"
import { JobPage } from "@/pages/JobPage"
import { SettingsPage } from "@/pages/SettingsPage"

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<QueuePage />} />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="/jobs/:id" element={<JobPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  )
}
