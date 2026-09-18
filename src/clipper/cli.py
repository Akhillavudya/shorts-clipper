"""One entrypoint, one subcommand per stage (FULL_PLAN.md section 11).

    clipper discover --limit 15
    clipper ingest <url|file>
    clipper transcribe <job_id>
    clipper moments <job_id>
    clipper render <job_id> --clip 1
    clipper review <job_id>
    clipper run <url> --auto
    clipper publish <job_id> --clip 1
    clipper upload <job_id> --clip 1
    clipper calibrate --apply

Every stage takes a job id and works against ``jobs/<job_id>/state.json``, so
stages compose in any order the state allows and a failed stage is rerun on its
own rather than restarting the pipeline.

This module parses arguments and prints. The work lives in ``pipeline.py``, so
a stage can be driven from a test or the review UI without Typer in the way.

``discover``, ``publish``, ``upload`` and ``calibrate`` sit outside the four
per-job stages: discovery runs before a job exists, and the rest run once a
clip is rendered or a Short is live. ``calibrate`` reads across every job at
once because the sample is the point. ``review`` is outside them too - it runs
a local web server and edits state rather than producing any. ``auth`` is not a
stage at all: it is the one-time browser step that mints the OAuth refresh
token ``upload`` and ``calibrate`` both run on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from clipper import config as config_module
from clipper import media, pipeline
from clipper.state import RightsBlocked, StageStatus, summarize

app = typer.Typer(
    name="clipper",
    help="Long-form video in, vertical Shorts out. One JSON state file per job.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

JobsRoot = Annotated[
    Path | None,
    typer.Option("--jobs-root", help="Directory holding per-job state."),
]
ConfigPath = Annotated[
    Path,
    typer.Option("--config", help="Path to settings.yaml.", show_default=True),
]


def _progress(stage: str, detail: str = "") -> None:
    console.print(f"  [dim]{stage:<22}[/dim] {detail}")


def _context(config: Path, jobs_root: Path | None) -> pipeline.Context:
    try:
        return pipeline.Context.build(config_path=config, jobs_root=jobs_root)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _fail(exc: Exception, code: int = 1) -> typer.Exit:
    console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
    return typer.Exit(code=code)


@app.command()
def discover(
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="How many candidates to shortlist.")
    ] = None,
    per_channel: Annotated[
        int, typer.Option("--per-channel", help="Recent uploads to check per channel.")
    ] = 5,
    no_summary: Annotated[
        bool,
        typer.Option("--no-summary", help="Skip the LLM blurb pass. Pure arithmetic."),
    ] = False,
    include_blocked: Annotated[
        bool,
        typer.Option(
            "--include-blocked",
            help="Also list red/unknown channels. They can never be rendered.",
        ),
    ] = False,
    watchlist: Annotated[
        Path, typer.Option("--watchlist", help="Channel watchlist to scan.")
    ] = Path("config/watchlist.yaml"),
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Scan the watchlist and shortlist candidate source videos.

    Quota-aware and watchlist-driven, never search-driven (section 3). Velocity
    scoring is deterministic; a single LLM call summarizes the shortlist at the
    end. Prints a markdown table and writes a JSON file.
    """
    from clipper.discover import summarize as summarize_module

    ctx = _context(config, jobs_root)
    try:
        result = pipeline.discover(
            ctx,
            limit=limit,
            watchlist_path=watchlist,
            per_channel=per_channel,
            include_blocked=include_blocked,
            summarize=not no_summary,
            on_progress=_progress,
        )
    except Exception as exc:
        raise _fail(exc) from exc

    # The markdown table is printed raw rather than as a rich table: section 3
    # asks for markdown so the shortlist can be pasted into notes, and rich
    # would render box-drawing characters that do not survive the paste.
    console.print(summarize_module.markdown_table(result.candidates))
    console.print()

    quota = result.quota or {}
    console.print(
        f"  [dim]{result.channels_scanned} channels scanned, "
        f"{quota.get('spent_this_run', 0)} quota units this run, "
        f"{quota.get('used_today', 0)}/{quota.get('ceiling', 0)} today[/dim]"
    )
    if result.blocked_hidden:
        console.print(
            f"  [dim]{result.blocked_hidden} red/unknown channels hidden "
            f"(--include-blocked to list them)[/dim]"
        )
    if result.skipped:
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(result.skipped.items()) if v)
        if skipped:
            console.print(f"  [dim]not scored: {skipped}[/dim]")
    if result.episodes:
        console.print(f"  [dim]{len(result.episodes)} recent podcast episodes in the report[/dim]")
    for where, problem in result.errors.items():
        console.print(f"  [yellow]{where}:[/yellow] {problem}")
    if result.report_path:
        console.print(f"  [dim]written to[/dim] {result.report_path}")
    if result.candidates:
        console.print(f"  [dim]then:[/dim] clipper ingest {result.candidates[0].url}")


@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="YouTube URL, or a path to a local video file.")],
    rights: Annotated[
        str | None,
        typer.Option(
            "--rights",
            help="Override rights status for this source: green|amber|red.",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rerun even if the stage is already done.")
    ] = False,
    watchlist: Annotated[
        Path, typer.Option("--watchlist", help="Channel watchlist to classify against.")
    ] = Path("config/watchlist.yaml"),
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Download the source, extract audio, and open a job.

    Creates ``jobs/<video_id>/`` with ``source.mp4``, ``audio.wav`` and an
    initial ``state.json``. Resolves rights status from the watchlist unless
    ``--rights`` is given.
    """
    ctx = _context(config, jobs_root)
    try:
        state = pipeline.ingest(
            ctx,
            source,
            rights=rights,
            watchlist_path=watchlist,
            force=force,
            on_progress=_progress,
        )
    except Exception as exc:
        raise _fail(exc) from exc

    colour = {"green": "green", "amber": "yellow"}.get(state.rights_status.value, "red")
    console.print(
        f"[bold]{state.video_id}[/bold]  rights=[{colour}]{state.rights_status.value}[/{colour}]"
    )
    if state.title:
        console.print(f"  {state.title}")
    console.print(f"  [dim]{state.rights_note or ''}[/dim]")
    if not state.is_exportable:
        console.print(
            "  [yellow]render will refuse this source until it is classified "
            "in config/watchlist.yaml.[/yellow]"
        )


@app.command()
def transcribe(
    job_id: Annotated[str, typer.Argument(help="Job id (the video id).")],
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="gemini | whisperx. Defaults to settings.yaml."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rerun even if the stage is already done.")
    ] = False,
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Produce a word-level, diarized transcript.

    Both backends normalize into the schema in ``transcribe/schema.py``, so
    nothing downstream depends on which one ran. Writes ``transcript.json``.
    """
    ctx = _context(config, jobs_root)
    try:
        transcript = pipeline.transcribe(
            ctx, job_id, backend=backend, force=force, on_progress=_progress
        )
    except Exception as exc:
        raise _fail(exc) from exc

    speakers = ", ".join(transcript.speakers()) or "none detected"
    console.print(
        f"[bold]{job_id}[/bold]  {transcript.word_count} words, "
        f"{len(transcript.segments)} segments, speakers: {speakers}"
    )


@app.command()
def moments(
    job_id: Annotated[str, typer.Argument(help="Job id (the video id).")],
    top: Annotated[
        int | None,
        typer.Option("--top", help="How many candidates to send to the rerank pass."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rerun even if the stage is already done.")
    ] = False,
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Find candidate moments, score them, and snap the boundaries.

    Two LLM passes - recall then precision rerank (section 5) - followed by
    deterministic boundary snapping. Writes ``candidates.json`` and
    ``scored.json``.
    """
    ctx = _context(config, jobs_root)
    try:
        pipeline.moments(ctx, job_id, top=top, force=force, on_progress=_progress)
    except Exception as exc:
        raise _fail(exc) from exc

    state = ctx.store.load(job_id)
    if not state.clips:
        console.print(
            f"[yellow]{job_id}: nothing scored above min_score. "
            f"See jobs/{job_id}/scored.json for the raw rubric.[/yellow]"
        )
        return

    table = Table(title=f"{job_id} - shortlisted clips", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("len", justify="right")
    table.add_column("type")
    table.add_column("title")

    for clip in state.clips:
        table.add_row(
            str(clip.index),
            f"{clip.score:.2f}" if clip.score is not None else "-",
            f"{clip.duration_s:.0f}s" if clip.duration_s else "-",
            clip.type or "-",
            clip.suggested_title or clip.reason or "-",
        )
    console.print(table)
    console.print(f"  [dim]render with:[/dim] clipper render {job_id} --clip 1")


@app.command()
def render(
    job_id: Annotated[str, typer.Argument(help="Job id (the video id).")],
    clip: Annotated[
        int | None,
        typer.Option("--clip", help="Clip index to render. Omit for every clip not rejected."),
    ] = None,
    approved_only: Annotated[
        bool,
        typer.Option(
            "--approved-only",
            help="Render only clips kept in `clipper review`, not merely un-rejected.",
        ),
    ] = False,
    allow_red: Annotated[
        bool,
        typer.Option(
            "--allow-red",
            help="Export despite a red/unknown rights status. Recorded in the job state.",
        ),
    ] = False,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Why the rights override is justified. Recorded."),
    ] = None,
    music: Annotated[
        str | None,
        typer.Option(
            "--music",
            help=(
                "Lay a music bed under the speech: a path, or a filename in "
                "assets/music. Naming one turns the bed on for this render."
            ),
        ),
    ] = None,
    no_music: Annotated[
        bool,
        typer.Option("--no-music", help="Render without a bed even if settings.yaml enables one."),
    ] = False,
    centre_crop: Annotated[
        bool,
        typer.Option(
            "--centre-crop",
            help="Skip face tracking and use a static centre crop.",
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Rerun even if the stage is already done.")
    ] = False,
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Cut, reframe, mix, caption, and encode a clip.

    The rights gate fires here, because this is the only stage that produces an
    uploadable file. Writes ``clips/clip_NN/final.mp4``.
    """
    ctx = _context(config, jobs_root)

    if not media.available():
        console.print(
            "[red]ffmpeg and ffprobe are not on PATH.[/red] Rendering needs both.\n"
            "  Windows: winget install Gyan.FFmpeg"
        )
        raise typer.Exit(code=2)

    try:
        results = pipeline.render(
            ctx,
            job_id,
            clip_index=clip,
            approved_only=approved_only,
            music=music,
            no_music=no_music,
            centre_crop=centre_crop,
            allow_red=allow_red,
            override_reason=reason,
            force=force,
            on_progress=_progress,
        )
    except RightsBlocked as exc:
        console.print(f"[red]blocked by the rights gate[/red]\n{exc}")
        raise typer.Exit(code=3) from exc
    except Exception as exc:
        raise _fail(exc) from exc

    for result in results:
        # The flag is about the *crop*, not the source: a 2560x1440 frame
        # gives a 810x1440 9:16 crop, which is upscaled to 1080x1920 even
        # though the source was well above 1080p. Saying "source was below
        # 1080p" sent one afternoon looking for a download bug that was not
        # there.
        note = (
            " [yellow](upscaled - the 9:16 crop is narrower than 1080px)[/yellow]"
            if result.upscaled
            else ""
        )
        console.print(
            f"[green]{result.clip_id}[/green] {result.path} "
            f"({result.duration_s:.1f}s, {result.width}x{result.height}){note}"
        )


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="YouTube URL, or a path to a local video file.")],
    auto: Annotated[
        bool,
        typer.Option("--auto", help="Run every stage without pausing for review."),
    ] = False,
    rights: Annotated[
        str | None,
        typer.Option("--rights", help="Override rights status: green|amber|red."),
    ] = None,
    allow_red: Annotated[
        bool, typer.Option("--allow-red", help="With --auto, export despite a red status.")
    ] = False,
    music: Annotated[
        str | None,
        typer.Option("--music", help="Lay a music bed under the speech. See `clipper render`."),
    ] = None,
    no_music: Annotated[
        bool,
        typer.Option("--no-music", help="Render without a bed even if settings.yaml enables one."),
    ] = False,
    centre_crop: Annotated[
        bool,
        typer.Option("--centre-crop", help="Skip face tracking and use a static centre crop."),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Rerun every stage from the start.")
    ] = False,
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Run every stage end to end, for when you trust it.

    Resumes from the first incomplete stage, so rerunning after a failure picks
    up where it stopped. Without ``--auto`` it stops after ``moments`` for
    human review, which is the intended default (section 12, Phase 4).
    """
    ctx = _context(config, jobs_root)
    try:
        state = pipeline.run_all(
            ctx,
            source,
            auto=auto,
            rights=rights,
            music=music,
            no_music=no_music,
            centre_crop=centre_crop,
            allow_red=allow_red,
            force=force,
            on_progress=_progress,
        )
    except RightsBlocked as exc:
        console.print(f"[red]blocked by the rights gate[/red]\n{exc}")
        raise typer.Exit(code=3) from exc
    except Exception as exc:
        raise _fail(exc) from exc

    console.print(summarize(state))
    if not auto:
        console.print(
            f"\n  [dim]stopped before render for review. "
            f"Then:[/dim] clipper render {state.video_id} --clip 1"
        )


@app.command()
def review(
    job_id: Annotated[
        str | None,
        typer.Argument(help="Job id to open. Omit to land on the job list."),
    ] = None,
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on.")] = 5151,
    host: Annotated[
        str,
        typer.Option("--host", help="Interface to bind. Anything but localhost exposes media."),
    ] = "127.0.0.1",
    open_browser: Annotated[
        bool,
        typer.Option("--browser/--no-browser", help="Open the page automatically."),
    ] = True,
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Serve the local review page: watch each candidate, keep it or cut it.

    Writes ``approved`` and ``review_note`` to the job state and nothing else -
    no stage runs from the browser. Render what survives with
    ``clipper render <job> --approved-only``.

    Binds to localhost by default on purpose. The page streams media straight
    out of the jobs root, so a bind address other than loopback publishes every
    source video on this machine to whoever can reach it.
    """
    import threading
    import webbrowser

    ctx = _context(config, jobs_root)

    if job_id is not None and not ctx.store.exists(job_id):
        console.print(f"[red]no job {job_id!r} under {ctx.jobs_root}[/red]")
        raise typer.Exit(code=2)

    path = f"/jobs/{job_id}" if job_id else "/"
    url = f"http://{'localhost' if host in {'0.0.0.0', '127.0.0.1'} else host}:{port}{path}"

    console.print(f"[bold]clipper review[/bold]  {url}")
    console.print(f"  [dim]jobs root:[/dim] {ctx.jobs_root}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            f"  [yellow]bound to {host}: every video under {ctx.jobs_root} is "
            f"readable from that address.[/yellow]"
        )
    console.print("  [dim]ctrl-c to stop[/dim]")

    # Imported here, not at module scope: Flask is only needed by this one
    # command, and `clipper moments` should not pay to import it.
    from clipper.review import serve

    if open_browser:
        # After the server is listening, not before - a browser that arrives
        # first gets a connection refused and shows its own error page.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        serve(host=host, port=port, store=ctx.store)
    except KeyboardInterrupt:  # pragma: no cover - ctrl-c is not a traceback
        console.print("  [dim]stopped[/dim]")


@app.command()
def publish(
    job_id: Annotated[str, typer.Argument(help="Job id (the video id).")],
    clip: Annotated[
        int | None, typer.Option("--clip", help="Clip index. Omit for every clip not rejected.")
    ] = None,
    approved_only: Annotated[
        bool,
        typer.Option("--approved-only", help="Only clips kept in `clipper review`."),
    ] = False,
    published: Annotated[
        str | None,
        typer.Option(
            "--published",
            help="YouTube video id of the uploaded Short. Recorded for calibration.",
        ),
    ] = None,
    allow_red: Annotated[
        bool,
        typer.Option("--allow-red", help="Build metadata despite a red/unknown status."),
    ] = False,
    reason: Annotated[
        str | None, typer.Option("--reason", help="Why the rights override is justified.")
    ] = None,
    watchlist: Annotated[
        Path, typer.Option("--watchlist", help="Channel watchlist, for topics and credit.")
    ] = Path("config/watchlist.yaml"),
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Write the upload title, description and credit for a rendered clip.

    Writes ``clips/clip_NN/upload.json``. The rights gate fires here too: the
    description is the credit block, and a credit block for a source that may
    not be clipped is an upload waiting to happen.

    ``--published`` records the live video id on the clip. That id is the join
    key ``clipper calibrate`` needs; without it retention data has nothing to
    correlate against.
    """
    ctx = _context(config, jobs_root)
    try:
        built = pipeline.publish(
            ctx,
            job_id,
            clip_index=clip,
            approved_only=approved_only,
            published_id=published,
            watchlist_path=watchlist,
            allow_red=allow_red,
            override_reason=reason,
        )
    except RightsBlocked as exc:
        console.print(f"[red]blocked by the rights gate[/red]\n{exc}")
        raise typer.Exit(code=3) from exc
    except Exception as exc:
        raise _fail(exc) from exc

    for meta in built:
        console.print(f"[green]{meta.clip_id}[/green] {meta.title}")
        if meta.title_truncated:
            console.print("  [yellow]title was truncated to fit 100 characters[/yellow]")
        if meta.attribution_required:
            console.print("  [dim]attribution is a condition of this source's rights status[/dim]")
        console.print(f"  [dim]{ctx.store.clip_dir(job_id, meta.clip_id) / 'upload.json'}[/dim]")

    if published and built:
        console.print(f"  [green]recorded[/green] youtube_video_id={published}")


@app.command()
def upload(
    job_id: Annotated[str, typer.Argument(help="Job id (the video id).")],
    clip: Annotated[int, typer.Option("--clip", help="Clip index. Required: one at a time.")],
    privacy: Annotated[
        str | None,
        typer.Option(
            "--privacy",
            help="private | unlisted | public. Default comes from settings.yaml.",
        ),
    ] = None,
    notify: Annotated[
        bool,
        typer.Option("--notify", help="Notify subscribers. Off by default."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Upload again even though this clip is already live."),
    ] = False,
    allow_red: Annotated[
        bool,
        typer.Option("--allow-red", help="Upload despite a red/unknown rights status."),
    ] = False,
    reason: Annotated[
        str | None, typer.Option("--reason", help="Why the rights override is justified.")
    ] = None,
    watchlist: Annotated[
        Path, typer.Option("--watchlist", help="Channel watchlist, for topics and credit.")
    ] = Path("config/watchlist.yaml"),
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Upload one rendered clip to YouTube, resumably.

    Sends ``clips/clip_NN/final.mp4`` with the metadata in ``upload.json``,
    records the resulting video id on the clip, and leaves the video
    ``private`` unless told otherwise. A run that dies part-way resumes into
    the same session rather than starting a second upload of the same Short.

    This is the one irreversible stage. It takes a single ``--clip``, it
    refuses a clip that already has a video id without ``--force``, and the
    rights gate fires here as well as at render and publish.
    """
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

    ctx = _context(config, jobs_root)

    with Progress(
        TextColumn("  [dim]uploading[/dim]"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("upload", total=None)

        def on_progress(sent: int, total: int) -> None:
            progress.update(task, completed=sent, total=total)

        try:
            result = pipeline.upload(
                ctx,
                job_id,
                clip_index=clip,
                privacy=privacy,
                notify_subscribers=notify,
                force=force,
                watchlist_path=watchlist,
                allow_red=allow_red,
                override_reason=reason,
                on_progress=on_progress,
            )
        except RightsBlocked as exc:
            console.print(f"[red]blocked by the rights gate[/red]\n{exc}")
            raise typer.Exit(code=3) from exc
        except Exception as exc:
            raise _fail(exc) from exc

    console.print(f"[green]{result.clip_id}[/green] uploaded as {result.video_id}")
    console.print(f"  [dim]{result.watch_url}[/dim]")
    console.print(f"  [dim]privacy:[/dim] {result.privacy_status or 'private'}")
    if result.resumed_from:
        console.print(f"  [dim]resumed from byte {result.resumed_from:,}[/dim]")
    console.print(f"  [dim]review it in Studio:[/dim] {result.studio_url}")


@app.command()
def auth(
    scopes: Annotated[
        str,
        typer.Option("--scopes", help="Comma-separated: upload, analytics."),
    ] = "upload,analytics",
    port: Annotated[
        int, typer.Option("--port", help="Loopback port for the redirect. 0 = any free port.")
    ] = 0,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Print the URL instead of opening it.")
    ] = False,
) -> None:
    """Run the one-time Google consent flow and print a refresh token.

    ``clipper upload`` and ``clipper calibrate`` both need OAuth, and neither
    can ask for it mid-run. This asks once, from a terminal you are sitting at, and
    prints the refresh token to paste into ``.env``.

    Needs ``YOUTUBE_CLIENT_ID`` and ``YOUTUBE_CLIENT_SECRET`` from a Google
    Cloud OAuth client of type **Desktop app** - only that type accepts the
    loopback redirect this uses.
    """
    import os

    from clipper import google_auth

    client_id = os.environ.get(google_auth.CLIENT_ID_ENV)
    client_secret = os.environ.get(google_auth.CLIENT_SECRET_ENV)
    if not (client_id and client_secret):
        console.print(
            f"[red]set {google_auth.CLIENT_ID_ENV} and "
            f"{google_auth.CLIENT_SECRET_ENV} first[/red]\n"
            "  Create an OAuth client (type: Desktop app) at\n"
            "  https://console.cloud.google.com/apis/credentials\n"
            "  and enable the YouTube Data API v3 and YouTube Analytics API."
        )
        raise typer.Exit(code=2)

    try:
        wanted = google_auth.resolve_scopes(scopes)
        console.print("[bold]clipper auth[/bold]  waiting for consent in the browser")
        for scope in wanted:
            console.print(f"  [dim]scope:[/dim] {scope}")
        payload = google_auth.run_consent_flow(
            client_id=client_id,
            client_secret=client_secret,
            scopes=wanted,
            port=port,
            open_browser=not no_browser,
        )
    except google_auth.AuthError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    granted = str(payload.get("scope") or "")
    missing = [s for s in wanted if s not in granted]

    console.print("\n[green]authorized[/green] - add this to your .env:\n")
    console.print(f"{google_auth.REFRESH_TOKEN_ENV}={payload['refresh_token']}")
    if missing:
        console.print(
            "\n[yellow]consent granted fewer scopes than asked for[/yellow]: " + ", ".join(missing)
        )
    console.print(
        "\n  [dim]the refresh token is a long-lived credential - it is not "
        "printed again, and .env is gitignored[/dim]"
    )


@app.command()
def calibrate(
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Write the fitted weights. Default is a dry run."),
    ] = False,
    days: Annotated[int, typer.Option("--days", help="How far back to pull analytics.")] = 90,
    weights: Annotated[Path, typer.Option("--weights", help="Scoring weights file to fit.")] = Path(
        "config/scoring_weights.yaml"
    ),
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Correlate predicted rubric scores against real retention, and re-fit.

    Reads every job at once - the sample is the point. Prints the per-axis
    correlations and the proposed weights; writes nothing unless ``--apply`` is
    passed *and* the fit predicts better than the weights already in force.
    """
    ctx = _context(config, jobs_root)
    try:
        report = pipeline.calibrate(ctx, apply=apply, days=days, weights_path=weights)
    except Exception as exc:
        raise _fail(exc) from exc

    console.print(report.render())


@app.command()
def status(
    job_id: Annotated[str | None, typer.Argument(help="Job id. Omit to list every job.")] = None,
    jobs_root: JobsRoot = None,
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Show stage status for one job, or list every job.

    Reads state.json and prints; it runs nothing. A stage showing ``running``
    means the process died mid-stage - that is the point of the status, not a
    bug in it.
    """
    ctx = _context(config, jobs_root)

    if job_id is not None:
        try:
            console.print(summarize(ctx.store.load(job_id)))
        except FileNotFoundError as exc:
            raise _fail(exc, code=2) from exc
        return

    job_ids = ctx.store.list_jobs()
    if not job_ids:
        console.print(f"[dim]no jobs under {ctx.jobs_root}[/dim]")
        return

    table = Table(title=f"jobs in {ctx.jobs_root}")
    table.add_column("job")
    table.add_column("rights")
    table.add_column("stage")
    table.add_column("clips", justify="right")

    for jid in job_ids:
        state = ctx.store.load(jid)
        pending = state.next_pending_stage()
        stage_label = pending or "[green]complete[/green]"
        if pending and state.stage(pending).status is StageStatus.failed:
            stage_label = f"[red]{pending} failed[/red]"
        table.add_row(jid, state.rights_status.value, stage_label, str(len(state.clips)))
    console.print(table)


@app.command()
def doctor(
    config: ConfigPath = Path("config/settings.yaml"),
) -> None:
    """Check that the things every stage assumes are actually present.

    Cheaper than discovering mid-run that ffmpeg is missing or a key is unset.
    """
    import importlib.util
    import os

    ok = True

    ffmpeg_ok = media.available()
    ok &= ffmpeg_ok
    ffmpeg_note = (
        "[green]found[/green]"
        if ffmpeg_ok
        else "[red]missing[/red] - needed by every stage past ingest"
    )
    console.print(f"  {'ffmpeg + ffprobe':<28} {ffmpeg_note}")

    try:
        settings = config_module.Settings.load(config)
        console.print(f"  {str(config):<28} [green]ok[/green]")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        console.print(f"  {config}       [red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    for name, block in (settings.section("providers") or {}).items():
        env = block.get("api_key_env")
        if not env:
            continue
        present = bool(os.environ.get(env))
        console.print(
            f"  {env:<28} {'[green]set[/green]' if present else '[yellow]unset[/yellow]'} ({name})"
        )

    # OAuth, not an API key: `upload` and `calibrate` are the two stages that
    # act as the channel rather than reading public data, and both fail at the
    # first call without these. Reported together because a refresh token is
    # only usable alongside the client that minted it.
    from clipper import google_auth

    oauth = {
        var: bool(os.environ.get(var))
        for var in (
            google_auth.CLIENT_ID_ENV,
            google_auth.CLIENT_SECRET_ENV,
            google_auth.REFRESH_TOKEN_ENV,
        )
    }
    if all(oauth.values()) or os.environ.get(google_auth.TOKEN_ENV):
        console.print(f"  {'youtube oauth':<28} [green]set[/green] (upload, calibrate)")
    elif any(oauth.values()):
        unset = ", ".join(var for var, present in oauth.items() if not present)
        console.print(f"  {'youtube oauth':<28} [yellow]incomplete[/yellow] - unset: {unset}")
    else:
        console.print(
            f"  {'youtube oauth':<28} [yellow]unset[/yellow] - run `clipper auth` "
            f"before upload or calibrate"
        )

    for path in (
        config_module.WATCHLIST_PATH,
        config_module.CAPTION_STYLE_PATH,
        config_module.SCORING_WEIGHTS_PATH,
    ):
        exists = Path(path).is_file()
        ok &= exists
        console.print(
            f"  {str(path):<28} {'[green]ok[/green]' if exists else '[red]missing[/red]'}"
        )

    # Face tracking is optional by design: without it `render` centre-crops
    # and records why. Reported here so that is a choice rather than a
    # surprise noticed three Shorts later.
    from clipper.edit import track

    edit_cfg = settings.edit
    if not track.tracking_enabled(edit_cfg):
        console.print(f"  {'face tracking':<28} [dim]off (edit.reframe: centre)[/dim]")
    else:
        options = track.options_from_config(edit_cfg)
        missing = [
            name
            for module, name in (("cv2", "opencv-python"), ("mediapipe", "mediapipe"))
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            console.print(
                f"  {'face tracking':<28} [yellow]{', '.join(missing)} not installed"
                f"[/yellow] - renders will centre-crop"
            )
        elif not options.model_path.is_file():
            console.print(
                f"  {'face tracking':<28} [yellow]model missing[/yellow] at "
                f"{options.model_path} - see assets/models/README.md"
            )
        else:
            console.print(f"  {'face tracking':<28} [green]ready[/green]")

    try:
        from clipper.rights.registry import Watchlist

        watchlist = Watchlist.load()
        blocked = sum(1 for c in watchlist.channels if not c.is_exportable)
        console.print(
            f"  watchlist                    [green]{len(watchlist)} channels[/green] "
            f"({blocked} blocked)"
        )
        if len(watchlist) == 0:
            console.print(
                "    [yellow]empty - every source will resolve to 'unknown' and "
                "render will refuse it.[/yellow]"
            )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        ok = False
        console.print(f"  watchlist                    [red]{exc}[/red]")

    raise typer.Exit(code=0 if ok else 1)


def main() -> None:
    """Console-script entrypoint.

    ``.env`` is loaded here rather than at import time so that a test or the
    Phase 4 UI importing ``pipeline`` never picks up the developer's keys by
    accident. Existing environment variables win, so an explicitly exported
    key still overrides the file.
    """
    load_dotenv(override=False)
    app()


if __name__ == "__main__":
    main()
