import { useEffect, type RefObject } from "react"

export function useKeyboardShortcuts(root: RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    let selected = 0
    const node = root.current
    if (!node) return
    const cards = () =>
      Array.from(node.querySelectorAll<HTMLElement>("[data-clip-card]"))
    const remember = (event: Event) => {
      const card = (event.target as HTMLElement).closest<HTMLElement>(
        "[data-clip-card]"
      )
      if (card) selected = cards().indexOf(card)
    }
    const keydown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.shiftKey ||
        target.isContentEditable ||
        target.closest("input, textarea, select, video, [role=combobox]")
      )
        return
      const list = cards()
      if (!list.length) return
      selected = Math.max(0, Math.min(selected, list.length - 1))
      if (event.key === "j" || event.key === "k") {
        event.preventDefault()
        selected = Math.max(
          0,
          Math.min(list.length - 1, selected + (event.key === "j" ? 1 : -1))
        )
        list[selected].focus({ preventScroll: true })
        list[selected].scrollIntoView({ block: "nearest" })
      }
      const verdict = { a: "approve", r: "reject", u: "reset" }[event.key]
      if (verdict && !event.repeat) {
        event.preventDefault()
        list[selected]
          .querySelector<HTMLButtonElement>(`[data-verdict=${verdict}]`)
          ?.click()
      }
      if (event.key === " " && !target.closest("button, a")) {
        event.preventDefault()
        const video = list[selected].querySelector("video")
        if (video) {
          if (video.paused) void video.play().catch(() => {})
          else video.pause()
        } else
          list[selected]
            .querySelector<HTMLButtonElement>("[data-mount-preview]")
            ?.click()
      }
    }
    node.addEventListener("focusin", remember)
    node.addEventListener("pointerdown", remember)
    document.addEventListener("keydown", keydown)
    return () => {
      node.removeEventListener("focusin", remember)
      node.removeEventListener("pointerdown", remember)
      document.removeEventListener("keydown", keydown)
    }
  }, [root])
}
