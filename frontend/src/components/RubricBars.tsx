import type { RubricAxis } from "@/lib/types"

export function RubricBars({ axes }: { axes: RubricAxis[] }) {
  return (
    <dl className="grid gap-2">
      {axes.map((axis) => (
        <div key={axis.name}>
          <div className="flex justify-between gap-2 text-xs">
            <dt>
              {axis.label}
              {axis.weight !== null && (
                <span className="text-muted-foreground">
                  {" "}
                  · weight {axis.weight}
                </span>
              )}
            </dt>
            <dd>{axis.score}/10</dd>
          </div>
          <div
            className="mt-1 h-1.5 overflow-hidden rounded-full bg-secondary"
            aria-hidden="true"
          >
            <div
              className="h-full rounded-full bg-primary"
              style={{ width: `${axis.percent}%` }}
            />
          </div>
        </div>
      ))}
    </dl>
  )
}
