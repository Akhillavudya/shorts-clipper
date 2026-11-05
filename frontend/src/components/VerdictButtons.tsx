import { Button } from "@/components/ui/button"

export function VerdictButtons({
  approved,
  disabled,
  onVerdict,
}: {
  approved: boolean | null
  disabled: boolean
  onVerdict: (value: boolean | null) => void
}) {
  return (
    <div
      className="flex flex-wrap gap-2"
      role="group"
      aria-label="Clip verdict"
    >
      <Button
        data-verdict="approve"
        disabled={disabled}
        aria-pressed={approved === true}
        variant={approved === true ? "default" : "outline"}
        onClick={() => onVerdict(true)}
      >
        Keep <kbd>A</kbd>
      </Button>
      <Button
        data-verdict="reject"
        disabled={disabled}
        aria-pressed={approved === false}
        variant={approved === false ? "destructive" : "outline"}
        onClick={() => onVerdict(false)}
      >
        Cut <kbd>R</kbd>
      </Button>
      <Button
        data-verdict="reset"
        disabled={disabled}
        variant="ghost"
        onClick={() => onVerdict(null)}
      >
        Undo <kbd>U</kbd>
      </Button>
    </div>
  )
}
