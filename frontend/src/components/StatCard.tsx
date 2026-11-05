import type { LucideIcon } from "lucide-react"
export function StatCard({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string
  value: string | number
  detail: string
  icon: LucideIcon
}) {
  return (
    <div className="stat-card">
      <div className="stat-label">
        {label}
        <Icon size={17} aria-hidden="true" />
      </div>
      <div className="stat-value">{value}</div>
      <p>{detail}</p>
    </div>
  )
}
