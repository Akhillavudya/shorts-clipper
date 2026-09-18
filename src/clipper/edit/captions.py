"""ASS caption generation (FULL_PLAN.md section 7).

Responsibility: word timestamps plus emphasis words -> captions.ass.

ASS, not SRT: SRT cannot express word-level highlighting.

- One dialogue line per ~3-5 words (the Shorts convention)
- Positioned in the lower third but above the platform UI overlay
- Karaoke \\k tags for word-by-word reveal, driven straight off the word
  timestamps
- Emphasis words from the rerank pass get a colour override (\\c&HXXXXXX&) and
  a scale bump
- Heavy outline and shadow so text survives any background

Every style value lives in config/caption_style.yaml, never in this module -
caption style is the thing that gets iterated on most.

Two ASS details that are easy to get wrong and hard to debug:

- **Timestamps are centiseconds**, formatted ``H:MM:SS.cc`` with a single-digit
  hour. Not milliseconds, and not zero-padded to two hours. A wrongly padded
  hour parses, and the subtitles simply never appear.
- **``\\k`` durations are also centiseconds, and they are relative** - each one
  says how long *this* word lasts, not when it starts. They accumulate from
  the line's own start time, so a line whose ``\\k`` values do not sum to its
  duration drifts, and the drift is invisible until the last word of the line
  highlights early.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipper.transcribe.schema import Word

CAPTIONS_FILENAME = "captions.ass"

#: Style name used for the caption lines. Referenced from every Dialogue row.
CAPTION_STYLE = "Caption"
HOOK_STYLE = "Hook"


def format_time(seconds: float) -> str:
    """Seconds -> ``H:MM:SS.cc``, the ASS timestamp format."""
    seconds = max(0.0, seconds)
    centis = int(round(seconds * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape(text: str) -> str:
    """Escape ASS's own syntax characters in caption text.

    Braces open an override block and a backslash starts a tag, so a spoken
    word containing either would be swallowed rather than displayed.
    """
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def normalize_word(text: str) -> str:
    """Lowercased, punctuation-stripped form, for emphasis matching."""
    return text.strip().strip(".,!?;:\"'()[]-—…").lower()


@dataclass
class CaptionLine:
    """One dialogue line: a few words shown together, revealed one at a time."""

    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.w for w in self.words)


def group_words(
    words: Sequence[Word],
    *,
    words_per_line: int = 4,
    max_gap_s: float = 0.8,
) -> list[CaptionLine]:
    """Chunk words into caption lines.

    Breaks on three things, in order of priority: a pause longer than
    ``max_gap_s``, terminal punctuation, and the word count. Punctuation
    matters more than the count - a line that runs past the end of a sentence
    and picks up two words of the next one reads badly even when it is exactly
    four words long.
    """
    lines: list[CaptionLine] = []
    current: list[Word] = []

    for word in words:
        if current:
            gap = word.start - current[-1].end
            if gap > max_gap_s or current[-1].ends_sentence or len(current) >= words_per_line:
                lines.append(CaptionLine(words=current))
                current = []
        current.append(word)

    if current:
        lines.append(CaptionLine(words=current))
    return lines


def _karaoke_spans(line: CaptionLine, *, min_word_ms: int) -> list[tuple[int, Word]]:
    """Per-word ``\\k`` durations in centiseconds, summing to the line length.

    A word shorter than ``min_word_ms`` is padded, because fast speech otherwise
    produces a strobing highlight. The padding is taken back off the longest
    word in the line, so the total still matches the line's duration and the
    reveal does not drift.
    """
    if not line.words:
        return []

    min_cs = max(1, int(round(min_word_ms / 10)))
    total_cs = max(1, int(round((line.end - line.start) * 100)))

    durations: list[int] = []
    for index, word in enumerate(line.words):
        # Charge each word the gap that follows it, so the highlight stays lit
        # through the pause instead of the line going blank mid-phrase.
        next_start = line.words[index + 1].start if index + 1 < len(line.words) else line.end
        span_cs = int(round((next_start - word.start) * 100))
        durations.append(max(min_cs, span_cs))

    # Reconcile against the line's real duration.
    drift = sum(durations) - total_cs
    while drift != 0:
        longest = max(range(len(durations)), key=lambda i: durations[i])
        if drift > 0:
            take = min(drift, durations[longest] - min_cs)
            if take <= 0:
                break
            durations[longest] -= take
            drift -= take
        else:
            durations[longest] -= drift  # drift is negative: this adds
            drift = 0

    return list(zip(durations, line.words, strict=True))


def render_line(
    line: CaptionLine,
    style: dict[str, Any],
    *,
    emphasis: set[str],
) -> str:
    """One Dialogue row, with karaoke tags and emphasis overrides."""
    karaoke = (style.get("karaoke") or {}).get("enabled", True)
    mode = (style.get("karaoke") or {}).get("mode", "k")
    min_word_ms = int((style.get("karaoke") or {}).get("min_word_ms", 90))

    emphasis_cfg = style.get("emphasis") or {}
    emphasis_on = bool(emphasis_cfg.get("enabled", True))
    emphasis_colour = (style.get("colours") or {}).get("emphasis")
    scale = int(emphasis_cfg.get("scale_percent", 112))

    pieces: list[str] = []
    spans = (
        _karaoke_spans(line, min_word_ms=min_word_ms) if karaoke else [(0, w) for w in line.words]
    )

    for duration_cs, word in spans:
        tags = ""
        if karaoke:
            tags += f"\\{mode}{duration_cs}"

        is_emphasis = emphasis_on and normalize_word(word.w) in emphasis
        if is_emphasis:
            if emphasis_colour:
                tags += f"\\c{emphasis_colour.strip('&')}&".replace("&&", "&")
            if scale and scale != 100:
                tags += f"\\fscx{scale}\\fscy{scale}"

        piece = (f"{{{tags}}}" if tags else "") + escape(word.w)
        if is_emphasis and (emphasis_colour or scale != 100):
            # Close the override so it does not bleed into the rest of the line.
            piece += "{\\r" + CAPTION_STYLE + "}"
        pieces.append(piece)

    body = " ".join(pieces)
    return (
        f"Dialogue: 0,{format_time(line.start)},{format_time(line.end)},"
        f"{CAPTION_STYLE},,0,0,0,,{body}"
    )


def _bool_field(value: Any) -> str:
    """ASS booleans are ``-1`` for true and ``0`` for false."""
    return "-1" if value else "0"


def render_styles(style: dict[str, Any], *, width: int, height: int) -> list[str]:
    """The ``[V4+ Styles]`` block, straight out of caption_style.yaml."""
    font = style.get("font") or {}
    colours = style.get("colours") or {}
    outline = style.get("outline") or {}
    layout = style.get("layout") or {}
    hook = style.get("hook_overlay") or {}

    name = font.get("name", "Arial")
    size = font.get("size", 78)
    bold = _bool_field(font.get("bold", True))
    italic = _bool_field(font.get("italic", False))
    spacing = font.get("spacing", 0)

    primary = colours.get("primary", "&H00FFFFFF&")
    # SecondaryColour is what \k reveals *from*: the un-spoken state. Using the
    # highlight colour here and primary as secondary would invert the karaoke.
    secondary = colours.get("highlight", "&H0000E5FF&")
    outline_colour = colours.get("outline", "&H00000000&")
    shadow_colour = colours.get("shadow", "&H80000000&")

    common = (
        f"{bold},{italic},0,0,100,100,{spacing},0,"
        f"{outline.get('border_style', 1)},{outline.get('width', 6)},"
        f"{outline.get('shadow', 3)}"
    )

    caption = (
        f"Style: {CAPTION_STYLE},{name},{size},"
        f"{primary},{secondary},{outline_colour},{shadow_colour},"
        f"{common},{layout.get('alignment', 2)},"
        f"{layout.get('margin_l', 80)},{layout.get('margin_r', 80)},"
        f"{layout.get('margin_v', 460)},1"
    )

    hook_style = (
        f"Style: {HOOK_STYLE},{name},{hook.get('size', 92)},"
        f"{hook.get('colour', primary)},{secondary},{outline_colour},{shadow_colour},"
        f"{common},{layout.get('alignment', 2)},"
        f"{layout.get('margin_l', 80)},{layout.get('margin_r', 80)},"
        f"{hook.get('margin_v', 1150)},1"
    )

    del width, height  # resolution is declared in the script info block
    return [caption, hook_style]


def render_hook(text: str, style: dict[str, Any], *, clip_duration_s: float) -> str | None:
    """The ``hook_rewrite`` overlay shown over the first seconds of the clip."""
    hook = style.get("hook_overlay") or {}
    if not hook.get("enabled", True) or not text.strip():
        return None
    duration = min(float(hook.get("duration_s", 2.0)), max(0.5, clip_duration_s))
    return (
        f"Dialogue: 1,{format_time(0.0)},{format_time(duration)},"
        f"{HOOK_STYLE},,0,0,0,,{escape(text.strip())}"
    )


def build(
    words: Sequence[Word],
    style: dict[str, Any],
    *,
    width: int = 1080,
    height: int = 1920,
    emphasis_words: Sequence[str] = (),
    hook_text: str | None = None,
    offset_s: float = 0.0,
    clip_duration_s: float | None = None,
) -> str:
    """Build a complete ASS file for one clip.

    ``words`` are episode-timeline words; ``offset_s`` is the clip's start time,
    subtracted so the subtitle file is relative to the clip. Getting this wrong
    puts every caption 40 minutes into a 40-second video, i.e. nowhere.
    """
    shifted = [
        Word(
            w=w.w,
            start=max(0.0, w.start - offset_s),
            end=max(0.0, w.end - offset_s),
            speaker=w.speaker,
        )
        for w in words
    ]

    layout = style.get("layout") or {}
    lines = group_words(shifted, words_per_line=int(layout.get("words_per_line", 4)))
    emphasis = {normalize_word(w) for w in emphasis_words if normalize_word(w)}

    if clip_duration_s is None:
        clip_duration_s = shifted[-1].end if shifted else 0.0

    out: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        *render_styles(style, width=width, height=height),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if hook_text:
        hook_line = render_hook(hook_text, style, clip_duration_s=clip_duration_s)
        if hook_line:
            out.append(hook_line)

    out.extend(render_line(line, style, emphasis=emphasis) for line in lines)
    return "\n".join(out) + "\n"


def write(
    words: Sequence[Word],
    style: dict[str, Any],
    out_path: Path | str,
    **kwargs,
) -> Path:
    """Build and write ``captions.ass``.

    UTF-8 without a BOM: libass reads a BOM as part of the first style name,
    and the whole file then renders in the default font.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build(words, style, **kwargs), encoding="utf-8", newline="\n")
    return out_path


def write_for_clip(
    words: Sequence[Word], style: dict[str, Any], clip_dir: Path | str, **kwargs
) -> Path:
    """``write`` into the conventional ``clips/clip_NN/captions.ass``."""
    return write(words, style, Path(clip_dir) / CAPTIONS_FILENAME, **kwargs)


__all__ = [
    "CAPTIONS_FILENAME",
    "CAPTION_STYLE",
    "HOOK_STYLE",
    "CaptionLine",
    "build",
    "escape",
    "format_time",
    "group_words",
    "normalize_word",
    "render_hook",
    "render_line",
    "render_styles",
    "write",
    "write_for_clip",
]
