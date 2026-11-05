/**
 * Paste links, walk away.
 *
 * This box is the product goal from plan section "Context", and everything
 * about it is shaped by one fact: the person using it has just copied a URL
 * and wants to stop thinking. So one textarea, one button, and the options
 * folded away behind a summary that stays shut unless you open it.
 *
 * One line per source, and the whole paste is one request: `POST /api/queue`
 * takes an array because "paste three links" is one user action.
 *
 * There is no rights override here, and that absence is the point of plan
 * section 5 - the API returns 400 if you send one. You cannot meaningfully
 * pre-authorise a gate override for a channel you have not seen yet; the
 * override lives on the job page, after ingest, with the licence note in front
 * of you.
 */

import * as React from "react"
import { ChevronRightIcon, SendHorizonalIcon } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldSet,
} from "@/components/ui/field"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Spinner } from "@/components/ui/spinner"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import type { SubmitOptions } from "@/lib/types"

interface SubmitBoxProps {
  onSubmit: (sources: string[], options: SubmitOptions) => Promise<void>
  /** Shown above the button. The API's own sentence, verbatim. */
  error: string | null
  onDismissError: () => void
}

/** The sentinel for "let the watchlist decide", which is the normal case.
 *
 * A `<Select>` cannot carry an empty-string value - Radix reserves it for the
 * placeholder - so the absence is spelled out and mapped back to `null` on the
 * way to the API.
 */
const RIGHTS_AUTO = "auto"

const RIGHTS_OPTIONS: { value: string; label: string; hint: string }[] = [
  {
    value: RIGHTS_AUTO,
    label: "From the watchlist",
    hint: "resolve the channel as usual",
  },
  { value: "green", label: "green", hint: "cleared to reuse and monetize" },
  {
    value: "amber",
    label: "amber",
    hint: "permission on file, evidence in rights/permissions/",
  },
  { value: "red", label: "red", hint: "not reusable - render will refuse it" },
  {
    value: "unknown",
    label: "unknown",
    hint: "unclassified - render will refuse it",
  },
]

export function SubmitBox({ onSubmit, error, onDismissError }: SubmitBoxProps) {
  const [text, setText] = React.useState("")
  const [rights, setRights] = React.useState(RIGHTS_AUTO)
  const [noMusic, setNoMusic] = React.useState(false)
  const [centreCrop, setCentreCrop] = React.useState(false)
  const [force, setForce] = React.useState(false)
  const [busy, setBusy] = React.useState(false)

  const sources = React.useMemo(
    () =>
      text
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean),
    [text]
  )

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (sources.length === 0 || sources.length > 25 || busy) return
    setBusy(true)
    try {
      await onSubmit(sources, {
        rights: rights === RIGHTS_AUTO ? null : rights,
        no_music: noMusic,
        centre_crop: centreCrop,
        force,
      })
      // Cleared only on success. A failed submit that also wiped the textarea
      // would lose the paste, and the paste is the only thing here that cannot
      // be reconstructed.
      setText("")
    } catch {
      // The parent displays the server error; keep the source text for retry.
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-4">
      <FieldGroup>
        <Field>
          <FieldLabel htmlFor="sources">Video links or file paths</FieldLabel>
          <Textarea
            id="sources"
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              // Ctrl/Cmd+Enter submits. Plain Enter has to stay a newline:
              // the whole format of this box is one source per line.
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                void submit(event)
              }
            }}
            rows={4}
            aria-describedby="sources-help"
            aria-invalid={sources.length > 25}
            spellCheck={false}
            placeholder={
              "https://www.youtube.com/watch?v=...\nD:\\footage\\episode.mp4"
            }
            className="min-h-32 text-sm"
          />
          <FieldDescription id="sources-help">
            One video per line, up to 25 at a time. Videos process in the order
            you add them.
            {sources.length > 25 && (
              <span className="block text-destructive">
                Remove {sources.length - 25} sources to continue.
              </span>
            )}
          </FieldDescription>
        </Field>
      </FieldGroup>

      <details className="group rounded-lg border bg-card">
        {/* `list-none` removes the native triangle, which renders differently
            in every browser and cannot be styled; the chevron replaces it, and
            it has to be there - without any affordance the bar reads as a
            static label and the options are never found. */}
        <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-sm font-medium select-none">
          <ChevronRightIcon
            aria-hidden="true"
            className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-90"
          />
          Processing options
          <span className="min-w-0 truncate font-normal text-muted-foreground">
            Rights, audio &amp; framing
          </span>
        </summary>

        <div className="border-t px-3 py-3">
          <FieldSet>
            <FieldGroup>
              <Field orientation="responsive">
                <FieldLabel htmlFor="rights">Rights</FieldLabel>
                <Select value={rights} onValueChange={setRights}>
                  <SelectTrigger id="rights" className="w-full sm:w-64">
                    {/* Given children, `SelectValue` renders these instead of
                        cloning the chosen item - which is what keeps the hint
                        text in the list where it belongs. Echoed into the
                        trigger it overflows a 320px-wide field. */}
                    <SelectValue>
                      {
                        RIGHTS_OPTIONS.find((option) => option.value === rights)
                          ?.label
                      }
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {RIGHTS_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          <span className="flex flex-col">
                            <span>{option.label}</span>
                            <span className="text-xs text-muted-foreground">
                              {option.hint}
                            </span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <FieldDescription>
                  Choose a rights classification instead of the watchlist
                  default. Red and unknown sources cannot be rendered.
                </FieldDescription>
              </Field>

              <Field orientation="horizontal">
                <FieldLabel htmlFor="no-music">Disable added music</FieldLabel>
                <Switch
                  id="no-music"
                  checked={noMusic}
                  onCheckedChange={setNoMusic}
                />
              </Field>

              <Field orientation="horizontal">
                <FieldLabel htmlFor="centre-crop">Centre crop</FieldLabel>
                <Switch
                  id="centre-crop"
                  checked={centreCrop}
                  onCheckedChange={setCentreCrop}
                />
              </Field>

              <Field orientation="horizontal">
                <FieldLabel htmlFor="force">Rerun finished stages</FieldLabel>
                <Switch id="force" checked={force} onCheckedChange={setForce} />
              </Field>
              <FieldDescription>
                Leave off to resume from the last completed step. Turn on to
                process every step again.
              </FieldDescription>
            </FieldGroup>
          </FieldSet>
        </div>
      </details>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Could not add your videos</AlertTitle>
          <AlertDescription>
            <p>{error}</p>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onDismissError}
            >
              Dismiss
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Button
          type="submit"
          size="lg"
          disabled={sources.length === 0 || sources.length > 25 || busy}
        >
          {busy ? (
            <Spinner data-icon="inline-start" />
          ) : (
            <SendHorizonalIcon data-icon="inline-start" />
          )}
          {busy
            ? "Adding videos…"
            : sources.length > 1
              ? `Create clips from ${sources.length} videos`
              : "Create clips"}
        </Button>
        <span className="text-sm text-muted-foreground">
          <kbd className="rounded border px-1 font-mono text-xs">Ctrl</kbd>
          {" + "}
          <kbd className="rounded border px-1 font-mono text-xs">Enter</kbd>
        </span>
      </div>
    </form>
  )
}
