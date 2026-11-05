import { useEffect, type ReactNode } from "react"
import { NavLink, Link, useLocation } from "react-router-dom"
import {
  ArrowUpRight,
  Clapperboard,
  LayoutDashboard,
  Library,
  Monitor,
  Moon,
  Sun,
  Plus,
  ShieldCheck,
  Settings,
} from "lucide-react"
import { useTheme } from "@/components/theme-provider"
import { cn } from "@/lib/utils"

const navigation = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/jobs", label: "Clip library", icon: Library, end: false },
  { to: "/settings", label: "Settings & storage", icon: Settings, end: false },
]
export function AppShell({ children }: { children: ReactNode }) {
  const { theme, setTheme } = useTheme()
  const location = useLocation()
  useEffect(() => {
    if (location.hash === "#sources")
      document.getElementById("sources")?.focus()
    else window.scrollTo(0, 0)
  }, [location.pathname, location.hash, location.key])
  const section =
    location.pathname === "/"
      ? "Overview"
      : location.pathname === "/jobs"
        ? "Clip library"
        : location.pathname === "/settings" ? "Settings & storage" : "Clip review"
  return (
    <div className="studio-shell">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <aside className="studio-sidebar">
        <Link to="/" className="studio-logo" aria-label="Shorts Clipper home">
          <span className="logo-mark">
            <Clapperboard size={20} />
          </span>
          <span>
            clipper<span className="logo-period">.</span>
            <small>YOUR SHORTS STUDIO</small>
          </span>
        </Link>
        <div className="sidebar-section-label">Workspace</div>
        <nav aria-label="Main navigation" className="studio-nav">
          {navigation.map(({ icon: Icon, ...item }) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn("studio-nav-link", isActive && "is-active")
              }
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <Link to="/#sources" className="sidebar-create">
          <Plus size={17} /> Create new clips
        </Link>
        <div className="sidebar-bottom">
          <div className="sidebar-tip">
            <span className="eyebrow">Make your next moment</span>
            <p>
              One long video.
              <br />A new story in every clip.
            </p>
            <span className="text-xs text-muted-foreground">
              Create. Review. Make it yours.
            </span>
          </div>
          <a className="legacy-link" href="/legacy/">
            Classic review <ArrowUpRight size={14} />
          </a>
          <div className="workspace-profile">
            <div className="workspace-avatar">SC</div>
            <div>
              <strong>Personal workspace</strong>
              <span>Stored on your device</span>
            </div>
          </div>
        </div>
      </aside>
      <div className="studio-content">
        <header className="studio-topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="text-border">/</span>
            <strong>{section}</strong>
          </div>
          <div className="topbar-actions">
            <span className="local-label">
              <ShieldCheck size={14} /> Local workspace
            </span>
            <div className="theme-switch" role="group" aria-label="Appearance">
              {(
                [
                  { value: "light", icon: Sun, label: "Light theme" },
                  { value: "dark", icon: Moon, label: "Dark theme" },
                  { value: "system", icon: Monitor, label: "Use system theme" },
                ] as const
              ).map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setTheme(value)}
                  aria-label={label}
                  title={label}
                  aria-pressed={theme === value}
                  className={cn(theme === value && "is-active")}
                >
                  <Icon size={15} />
                  <span className="theme-label">
                    {value === "system"
                      ? "Auto"
                      : value === "dark"
                        ? "Dark"
                        : "Light"}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </header>
        <main id="main" className="studio-main">
          {children}
        </main>
        <footer className="studio-footer">
          <span>Built for the moments worth sharing.</span>
          <span>Create · Review · Download</span>
        </footer>
      </div>
    </div>
  )
}
