# shorts-clipper

Long-form video in, vertical Shorts out. Watchlist-driven, rights-gated,
resumable.

See `FULL_PLAN.md` for the research and design rationale, and `CLAUDE.md` for
how to work in this repo.

```bash
uv sync
uv run clipper --help
```

## Pipeline

```
discover -> ingest -> transcribe -> moments -> render -> publish -> calibrate
```

The four middle stages are the per-job pipeline. Each reads
`jobs/<video_id>/state.json`, does its work, and writes it back. Rerun a stage,
not the pipeline.

`discover` runs before a job exists and writes a dated shortlist you pick from.
`review` is the human in the loop, between `moments` and `render`.
`publish` writes the upload title, description and credit for a rendered clip,
and `upload` sends it. `calibrate` correlates the rubric scores that predicted a
clip against the retention it actually got, and re-fits the weights.

```bash
clipper discover --limit 15
clipper ingest <url|file>
clipper transcribe <job_id>
clipper moments <job_id>
clipper review <job_id>
clipper render <job_id> --clip 1
clipper publish <job_id> --clip 1
clipper upload <job_id> --clip 1
clipper calibrate --apply
```

## Uploading

`clipper upload` sends one rendered clip through `videos.insert`, resumably,
and records the video id it comes back with - the key `calibrate` joins
retention on. It lands **private**: the last place a bad clip gets caught is
somebody opening it in Studio, and making a good one public is one click.

```bash
clipper auth                                  # once: prints a refresh token
clipper upload <job_id> --clip 1              # private, one clip at a time
clipper upload <job_id> --clip 1 --privacy unlisted
```

It refuses a clip that already has a video id unless you pass `--force`, and
the rights gate fires here as well as at render and publish. A run that dies
part-way resumes into the same upload session rather than sending the file
twice.

One upload costs 1600 of the project's 10,000 YouTube Data API units per day -
the same budget discovery spends one unit at a time, and both charge
`jobs/.discover/quota.json`. Four uploads a day is most of it.

`clipper auth` needs an OAuth client of type **Desktop app** from the Google
Cloud console, with the YouTube Data API v3 and YouTube Analytics API enabled;
put its id and secret in `.env` first. If you would rather upload by hand,
`clipper publish` still writes `upload.json` and
`clipper publish <job> --clip 1 --published <id>` records the result.

## Review

`moments` scores candidates; it does not decide. `clipper review` serves a local
page listing them with the rubric, the words, and an inline preview, and keep /
cut buttons that write `approved` to the job state.

```bash
clipper review <job_id>                       # http://localhost:5151
clipper render <job_id> --approved-only       # then encode only what survived
```

Review happens *before* render, so there is usually nothing encoded to play
yet: the preview seeks into `source.mp4` at the cut points and stops at the
end, and switches to `final.mp4` once the clip has been rendered. Keyboard:
`j`/`k` to move, `a` to keep, `r` to cut, `u` to undo, space to play.

Say why in the note box. Rejections are the training data the calibration loop
has no other source for - `scored.json` keeps the rubric that predicted a clip
was good, and the note is the record of whether it was.

The server binds to `127.0.0.1` and has no authentication, because the media
route streams straight out of the jobs root. `--host` anything else and every
source video on the machine is readable from that address.

## Reframing

16:9 comes out 9:16 with the crop following whoever is talking - MediaPipe face
detection, an EMA and a deadzone so the frame does not jitter, and a hard reset
at each shot boundary. The path is handed to ffmpeg as a `sendcmd` script, so
the encode is the same one the static crop uses.

```bash
clipper render <job_id> --clip 1                  # tracked (the default)
clipper render <job_id> --clip 1 --centre-crop    # static, for one run
```

It needs the detection model in `assets/models/` - see the README there for
the one-line fetch, and `clipper doctor` to check. Without it, or when too few
frames contain a face to be worth trusting, the render falls back to a centre
crop and records why on the clip's `reframe` stage. `edit.reframe: centre` in
`config/settings.yaml` turns tracking off for good.

## Music

A clip can carry a music bed, ducked out from under the speech by a sidechain
compressor keyed on the voice:

```bash
clipper render <job_id> --clip 1 --music lofi-rhodes-bed.mp3   # or any path
clipper render <job_id> --clip 1 --no-music                    # off for one run
```

Tracks live in `assets/music/`, and every one of them needs its licence
recorded in `assets/music/README.md` - a Content ID claim on the audio takes
the revenue for the whole Short. The bed is normalized to
`edit.music_bed_lufs`, so swapping tracks does not change the mix. `--music`
implies it is on for that render; `edit.music_enabled: true` makes it the
default.

## Status

Working end to end, every phase of the build order in `FULL_PLAN.md` section 12
included. 619 tests, none of which touch a network or need ffmpeg.

Keys go in `.env` (copy `.env.example`); the CLI loads it on start.

Before it does anything useful you need `config/watchlist.yaml` filled in — it
is the file that decides both what gets scanned and what may legally be clipped,
and it ships empty. `clipper doctor` will tell you what else is missing.
