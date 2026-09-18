# Shorts Clipper — Research Findings + Build Plan

**Date:** 7 September 2026
**Target:** 1–3 Shorts/day from long-form video, semi-automated, Python-first
**Repo:** new, separate from the existing faceless-shorts project

---

## 0. The finding that changes the design

You asked about copyright at the end. It belongs at the top, because it decides the architecture.

Three facts from current YouTube policy:

1. **Reused-content enforcement is at the channel level, not the video level.** If reviewers decide a channel violates it, monetization is suspended for the whole channel; you reapply after 30 days. One bad batch kills the asset, not one upload.
2. **Clips and compilations are still monetizable** — YouTube said explicitly in the July 2025 update that the reused-content policy (which covers commentary, clips, compilations, reactions) was unchanged. What got renamed and tightened was "repetitious content" → **inauthentic content**: mass-produced, templated output. In January 2026 a coordinated sweep removed a set of channels from YPP under that policy.
3. **Fair use is a defence you argue after a claim, not a shield that prevents one.** Content ID fingerprints audio and video automatically.

So the risky profile is exactly: *high-volume, templated, third-party footage, no added voice*. That is the default output of a naive clipping pipeline. The pipeline itself is fine; what you feed it decides whether the channel survives.

**Design consequence:** `rights_status` is a first-class field on every video record, and the renderer refuses to export anything marked `red` without an explicit override flag. Build it in from day one — retrofitting it after you have 200 uploads is how channels die.

### Source tiers

| Tier | Sources | Monetization risk |
|---|---|---|
| **Green** | Your own footage; creators with a public clipping policy or written permission; paid clipping campaigns (brand-authorized footage); YouTube CC-BY licence filter; US federal government works (public domain); Internet Archive public-domain material | Low |
| **Amber** | Mid-tier creators (10k–500k subs) who said yes by email. Very often they say yes — clips are free distribution for them. Save the email in the repo. | Low once documented |
| **Red** | Rogan, Lex, Huberman, Hormozi tier. They already run official clip channels staffed by their own teams. Raw 45-second highlights from these are the canonical reused-content pattern. | High |

Watch out for licences that *look* open but aren't: **MIT OpenCourseWare and Khan Academy are CC BY-NC-SA** (no commercial use, and monetized YouTube is commercial), **TED is CC BY-NC-ND** (no derivatives at all — a clip is a derivative). Neither is usable for a monetized channel. On YouTube, filter for the **Creative Commons – Attribution (CC BY)** licence specifically; that one does permit commercial derivatives with attribution.

### The transformation layer

Whatever tier you're on, these are what separate "a clip" from "a copy", and they're all things your pipeline does anyway:

- Editorial selection (you chose this 40s out of 2 hours)
- Reframing, captions, emphasis, pacing
- **Keep the clip tiny relative to the source** — 40s out of a 2-hour podcast is a much stronger position than 40s out of a 4-minute video
- Credit the creator on-screen and in the description, and link the full episode
- Never look like the creator's official account

The single strongest addition is **your own voice**: a 5-second framing line before the clip, or a text-overlay argument the clip is evidence for. That is also the thing that converts the channel from "clip farm" to "show", which is what both the policy and the algorithm now reward. It also reuses the script-generation work from your other project.

---

## 1. Architecture decision: no multi-agent system

You were right to be suspicious. Here is the actual split.

**Needs an LLM (3 calls total per video):**
- Ranking/summarizing discovery candidates ("why is this interesting")
- Finding candidate moments in a transcript
- Scoring/ranking those candidates, and picking emphasis words + title/description

**Does not need an LLM (deterministic Python):**
- Download (`yt-dlp`), audio extraction, all encoding (`ffmpeg`)
- Trend/velocity scoring — arithmetic on view counts beats a model here
- Boundary snapping to word timestamps and silence
- Scene detection (`PySceneDetect`), face detection/tracking (`MediaPipe` or `YOLO` + `ByteTrack`), crop-path smoothing
- Caption rendering (ASS with karaoke tags), keyword colouring
- Loudness normalization, music ducking, final render
- Dedupe, quota bookkeeping, state management

**Framework:** none. No LangGraph, no CrewAI. A Python CLI with named stages and one JSON state file per job. Each stage reads the state file, does its job, writes it back. Resumable, debuggable, inspectable with `cat`. If a stage fails you rerun that stage, not the pipeline.

```
jobs/<video_id>/
  state.json          # single source of truth for this job
  source.mp4
  audio.wav
  transcript.json     # word-level, diarized
  candidates.json     # LLM pass 1 output
  scored.json         # LLM pass 2 output
  clips/
    clip_01/
      cut.mp4  reframed.mp4  captions.ass  final.mp4  meta.json
```

---

## 2. Stack, decided

| Layer | Choice | Why | Cost |
|---|---|---|---|
| Download | `yt-dlp` | Standard. Note downloading is against YouTube ToS regardless of rights status — a separate issue from copyright; it's your call how you weigh it. Permissioned creators can often just send you the file. | free |
| Discovery | YouTube Data API v3 + Podcast Index (free) | See §3 | free (quota-capped) |
| Transcription (cloud) | **Gemini Transcribe** ~$0.005/min, has word-level timestamps + diarization | ~$0.03 for a 2-hour episode. Start here. | ~$0.03/video |
| Transcription (local) | **WhisperX** (faster-whisper + wav2vec2 alignment + pyannote diarization) | <100ms word timestamps — enough for karaoke captions. Needs NVIDIA GPU with ≥6GB VRAM. `whisper.cpp` if you're on a Mac. | free |
| Transcription (alt) | Parakeet TDT 0.6B v3 | English-only but faster and slightly more accurate than Whisper large-v3 (6.32% vs 7.44% WER). Optimization, not a starting point. | free |
| Moment finding | Claude **Haiku 4.5** ($1/$5 per Mtok) over chunked transcript | Cheap, and this is a recall task | ~$0.05/video |
| Scoring/rerank | Claude **Sonnet 5** ($2/$10 per Mtok) on top ~20 candidates | Precision task, small input | ~$0.05/video |
| Video edit | `ffmpeg` + OpenCV + MediaPipe/YOLO | See §6 | free |
| Captions | ASS subtitles, burned in with ffmpeg | See §7 | free |
| Music | YouTube Audio Library (or none) | See §8 | free |
| Orchestration | Python CLI + JSON state | — | free |

**Use the Batch API (50% off) for anything not interactive.** Prompt caching (cache reads at 10% of input) matters if you re-score the same transcript.

---

## 3. Discovery (Step 1)

The API constraint that shapes this: **10,000 quota units/day per Google Cloud project. `search.list` costs 100 units — roughly 100 searches/day.** (There are reports of a June 2026 granular-quota change moving `search.list` to its own ~100-call/day bucket and dropping `videos.insert` to 1 unit with a 100/day cap; either way ~100 searches is the ceiling. Check your Cloud Console quota page before you build against a number.)

**So don't build a search-based crawler.** Build a watchlist-based one:

1. **Curate a channel watchlist** (30–80 channels), each tagged with `rights_status` and a note (link to their clipping policy, or the date they gave permission). This is a YAML file you maintain by hand. It is the highest-leverage asset in the project.
2. Resolve each channel's **uploads playlist ID** once (`channels.list`, 1 unit, cached forever).
3. Daily: `playlistItems.list` on each uploads playlist — **1 unit**, not 100. 80 channels = 80 units.
4. Batch the resulting video IDs into `videos.list` (50 IDs per call, 1 unit) for stats.
5. Compute a **velocity score** deterministically:
   `views_per_hour_since_publish / channel_median_views_per_hour` — a normalized outlier score. This is what surfaces "this one is popping" and it needs no AI.
6. Supplement with `videos.list?chart=mostPopular` (1 unit) and the free **Podcast Index API** for RSS-side discovery. Taddy and Listen Notes are the paid options if you later want transcripts or charts — not needed for MVP.
7. **Only then** call an LLM: pass the top ~15 by velocity, get back the "why interesting / probable moments" blurb for your shortlist.

Output the list you specified (title, channel, link, why, probable moments, popularity) as a markdown table to stdout plus a JSON file. You pick, you run `clip ingest <url>`.

Total daily quota: well under 500 units. You will never hit the wall.

---

## 4. Transcription (Step 2a)

Start with Gemini Transcribe at ~$0.005/min because it's pennies and has diarization built in. Move to local WhisperX when you have a GPU and want zero marginal cost.

Normalize both into one internal schema so the rest of the pipeline never knows which was used:

```json
{
  "words": [{"w": "actually", "start": 412.34, "end": 412.71, "speaker": "SPEAKER_01"}],
  "segments": [{"start": 410.2, "end": 438.9, "speaker": "SPEAKER_01", "text": "..."}]
}
```

Word-level timestamps are non-negotiable — they drive both cut boundaries and karaoke captions.

---

## 5. Moment finding + scoring (Step 2b)

**Two passes. Never let the LLM emit the final cut points.**

**Pass 1 — recall (Haiku 4.5).** Chunk the transcript into ~10-minute windows with 60s overlap. Ask for every candidate span with: `start_word_index`, `end_word_index`, `type` (hook / surprise / insight / story / emotion / contrarian / educational / conclusion), `one_line_reason`. Ask for many — 30–50 across a 2-hour episode. Strict JSON output.

**Pass 2 — precision (Sonnet 5).** Feed the top ~20 candidates (deduped, merged where overlapping) with full text and ask for a rubric score, 1–10 each:

- **Hook** — does the first sentence stop a scroll on its own?
- **Self-containment** — does it make sense with zero context about the episode?
- **Payoff** — is there a resolution, not just a setup?
- **Surprise/emotion** — is there a reason to feel something?
- **Visual independence** — does it work when the video is just a talking head?
- **Quotability** — is there one line a viewer would repeat?
- **Length fit** — natural boundary in the 25–55s range?

Plus: `suggested_title`, `emphasis_words` (for caption highlighting), `hook_rewrite` (an on-screen text overlay for the first 2s).

Weighted sum, ranked. You review, you pick.

**Boundary snapping is deterministic.** Take the LLM's word indices → look up timestamps → extend backwards to the start of the sentence → snap outward to the nearest silence gap (>250ms, found with ffmpeg `silencedetect`) → pad 150ms each side. This kills the clipped-first-syllable problem that makes AI clips feel cheap.

**Log everything.** `scored.json` keeps the rubric scores. When you later pull retention data from YouTube Analytics, you can correlate predicted score against actual retention and tune the weights. After ~50 Shorts that calibration is your real edge over anyone using OpusClip.

---

## 6. Editing (Step 3)

All ffmpeg. Order of operations matters:

1. **Cut** — `ffmpeg -ss <start> -to <end> -i in.mp4` with re-encode, not stream copy. Stream copy snaps to keyframes and will be off by up to several seconds.
2. **Reframe to 9:16** —
   - `PySceneDetect` for shot boundaries (reset tracking at each cut)
   - Face/person detection every ~0.25s (`MediaPipe` face detection is enough for talking heads; `YOLOv8/v11` + `ByteTrack` if you need multi-person)
   - Pick the active speaker: largest/most central face, or use the diarization label + mouth movement
   - Build a crop-centre path, then **smooth it**: EMA + a deadzone so micro-movement doesn't cause jitter, and hard-reset at scene cuts
   - Two-person interview: either a stacked split (both faces, one above the other) or cut between them on speaker turns
   - Render by piping frames through OpenCV → ffmpeg, or generate a `sendcmd` file for the crop filter
3. **Audio cleanup** — `highpass=f=80`, optional `afftdn` denoise, then two-pass `loudnorm` to **-14 LUFS** (YouTube's target).
4. **Music** (if any) — bed at ~-26 dB under speech, or `sidechaincompress` ducking. Skip it entirely for dense talking-head content; it usually hurts.
5. **Captions** — burn in last (§7).
6. **Encode** — `libx264 -crf 19 -preset slow -pix_fmt yuv420p`, 1080×1920, AAC 192k, `+faststart`.

**Read these repos before writing your own version.** They have already solved the ugly parts:
- `artbyjazi/autoclip` — full local-first loop: ingest → transcribe → LLM highlight detection → speaker-tracked reframe → animated captions → export. Closest thing to your spec that already exists.
- `KazKozDev/auto-vertical-reframe` — PySceneDetect + YOLO-seg + ByteTrack + MediaPipe + camera-path optimizer. Good reference for the smoothing math.
- `NaufalRizqullah/opensource-clipping` — documented deadzone/jitter/scene-cut-reset approach.

Genuinely consider reading AutoClip and building on it rather than from scratch. Your differentiation is in the scoring rubric, the rights registry, and the calibration loop — not in re-deriving a crop smoother.

---

## 7. Captions

Generate **ASS** (not SRT) — SRT can't do word-level highlighting.

- One dialogue line per ~3–5 words (Shorts caption convention), positioned in the lower third but above the UI overlay
- Karaoke `\k` tags for word-by-word reveal, driven straight off the word timestamps
- Emphasis words from the LLM get a colour override (`\c&HXXXXXX&`) and a scale bump
- Heavy outline + shadow so it survives any background
- Burn in: `ffmpeg -vf "ass=captions.ass"`

Keep the style in a config file, not in code — you'll iterate on it constantly.

---

## 8. Music

Precise, because most advice here is wrong:

- **YouTube Audio Library is the only source YouTube itself calls copyright-safe.** It explicitly disclaims responsibility for royalty-free music from other libraries.
- **"Royalty-free" ≠ "Content ID cleared."** A track can be royalty-free and still trigger a claim, because the rights holder or a distributor uploaded a reference file.
- **Paid libraries (Epidemic Sound, Uppbeat, Artlist) work via channel safelisting** — you connect your channel ID and their tracks stop triggering claims *on that channel*. Uppbeat's safelisting is YouTube-only.
- **The Shorts in-app sound picker** licences popular music for Shorts under 60s, but revenue is shared with the rights holder. Reach, not revenue.
- **A Short over 60s with any Content ID claim gets blocked globally**, not just demonetized.

MVP: YouTube Audio Library, downloaded once into `assets/music/`, tagged by mood in a YAML file. Deterministic selection by the clip's `type`. Or no music — for a good talking-head clip it's usually noise.

---

## 9. Cost

Per Short (local rendering, cloud transcription + LLM):

| Item | Cost |
|---|---|
| Transcription (2h episode, amortized over ~3 clips) | ~$0.01 |
| Haiku 4.5 recall pass | ~$0.05 |
| Sonnet 5 rerank | ~$0.03 |
| Render | $0 |
| **Total** | **~$0.10** |

**90 Shorts/month ≈ $9.** Under $5 if you go local on transcription and use the Batch API.

Compare: OpusClip Starter $15/mo (credits burn at 1/minute of source — a 60-min podcast is 60 credits), Klap ~$14–29/mo, Vizard ~$14.50/mo annual.

**Be honest with yourself about why you're building this.** The savings are ~$10/month. The real returns are: full control of the scoring rubric, a calibration loop nobody else has, no per-minute credit meter, and a serious portfolio project that touches ASR, LLM orchestration, CV, and media engineering. Those are good reasons. "It's cheaper" is not one.

---

## 10. Local vs cloud

Everything local except LLM calls and (optionally) transcription. Video never leaves your machine — no upload bandwidth, no storage bill, no vendor limits. The only reason to go cloud is if rendering on your hardware is too slow, and at 1–3 Shorts/day it won't be.

If you have no NVIDIA GPU: use Gemini for transcription, CPU for MediaPipe (fine at 4fps sampling), ffmpeg on CPU. A 45-second Short renders in a couple of minutes. Perfectly acceptable.

---

## 11. Repo layout

Separate repo, sibling to your existing faceless-shorts project. Shared pieces (caption renderer, uploader) can be extracted into a common package later — don't do it upfront.

```
shorts-clipper/
├── README.md
├── pyproject.toml
├── config/
│   ├── settings.yaml           # paths, model names, thresholds
│   ├── watchlist.yaml          # channels + rights_status + permission notes
│   ├── caption_style.yaml
│   └── scoring_weights.yaml    # tuned by the calibration loop
├── src/clipper/
│   ├── cli.py                  # one entrypoint, subcommands per stage
│   ├── state.py                # job state read/write, resumability
│   ├── discover/
│   │   ├── youtube.py          # quota-aware client, playlistItems-first
│   │   ├── podcast_index.py
│   │   ├── velocity.py         # deterministic outlier scoring
│   │   └── summarize.py        # the one LLM call in discovery
│   ├── rights/
│   │   ├── registry.py         # rights_status lookup + gate
│   │   └── permissions/        # saved emails, screenshots of clip policies
│   ├── ingest/
│   │   ├── download.py         # yt-dlp wrapper
│   │   └── audio.py            # ffmpeg extract, normalize
│   ├── transcribe/
│   │   ├── gemini.py
│   │   ├── whisperx_local.py
│   │   └── schema.py           # the one internal transcript format
│   ├── moments/
│   │   ├── recall.py           # pass 1, Haiku
│   │   ├── score.py            # pass 2, Sonnet
│   │   ├── prompts/
│   │   └── snap.py             # deterministic boundary snapping
│   ├── edit/
│   │   ├── cut.py
│   │   ├── reframe.py          # scene detect + face track + smooth
│   │   ├── captions.py         # ASS generation
│   │   ├── audio_mix.py        # loudnorm, ducking
│   │   └── render.py
│   ├── publish/
│   │   └── metadata.py         # title, description w/ credit + source link
│   └── calibrate/
│       └── analytics.py        # pull retention, correlate vs predicted score
├── assets/music/
├── jobs/                       # gitignored
└── tests/
```

CLI shape:

```bash
clipper discover --limit 15
clipper ingest <url|file>
clipper transcribe <job_id>
clipper moments <job_id>
clipper render <job_id> --clip 1
clipper run <url> --auto        # all stages, for when you trust it
```

---

## 12. Build order

**Phase 0 — before writing any pipeline code (one weekend).**
Do it manually, end to end, five times. `yt-dlp` → WhisperX → paste transcript into Claude → cut in ffmpeg by hand → caption in CapCut → upload. Publish them. You will learn more about what makes a clip work in that weekend than in a month of building, and you will discover which pipeline stages actually matter. Skipping this is the most common way this project dies.

**Phase 1 — the vertical slice.** `ingest → transcribe → moments → render`, one video in, one Short out. Hardcode the caption style. No discovery, no reframing (centre-crop is fine). Get one watchable Short out of the CLI.

**Phase 2 — quality.** Face-tracked reframe, karaoke captions with emphasis, loudnorm, silence-snapped boundaries. This is where the output stops looking amateur.

**Phase 3 — discovery + rights.** Watchlist, velocity scoring, the rights gate. Also: send 20 permission emails to mid-tier creators in your niche. Expect several yeses.

**Phase 4 — review loop.** A minimal local web page listing scored candidates with inline video preview and approve/reject buttons. Flask + plain HTML. Rejections are training data.

**Phase 5 — calibration + upload.** Pull retention from the YouTube Analytics API for your own uploads, correlate with predicted scores, retune `scoring_weights.yaml`. Optionally automate upload (`videos.insert`).

---

## 13. Alternative approaches worth considering

Ranked by how well they fit where you actually are.

**A. Rights-clean source pool (recommended starting configuration).** Same pipeline, but the watchlist is CC BY, public domain, and permissioned creators only. Lower baseline virality than Rogan clips, near-zero policy risk, and you can run it at volume without the channel-level enforcement hanging over you. Start here, expand later.

**B. Clipper campaigns.** Platforms where creators and brands post authorized footage and pay per view (typically $1–5 CPM). Copyright is a non-issue by construction, and you get paid without needing YPP at all. Your pipeline becomes a throughput advantage over people editing manually in CapCut. Worth testing in parallel — it validates the pipeline commercially before your channel is monetized.

**C. Commentary format.** The clip is evidence; your framing is the product. Highest policy safety, strongest transformation argument, most work per Short — but it directly reuses the script-generation work from your other project, and it turns a clip farm into a show. This is the version that still exists in two years.

**D. Own-source.** Point the pipeline at your own long-form content, or at footage from people you actually know. Zero risk, and the pipeline works identically.

The honest recommendation: **build once, run on A and B, evolve toward C.** Keep the Red tier out of the channel entirely, or on a separate channel you're willing to lose.

---

## 14. Open questions to resolve before Phase 1

- Do you have an NVIDIA GPU available? Decides local vs cloud transcription.
- One niche or three? A channel with a consistent topic survives the "authentic channel" review far better than a general clip feed.
- Are you willing to add your own voice/commentary, or is it strictly faceless clips? This is the biggest single fork in the design.
- Which 30 channels go on the watchlist, and what's each one's rights status?
