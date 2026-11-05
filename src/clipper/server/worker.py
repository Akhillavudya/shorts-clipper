"""The worker child process (WEB_UI_PLAN.md section 1).

One process per job, spawned by the supervisor in ``runner.py`` and never by
anything else::

    python -m clipper.server.worker --entry <id> --source <url> \\
        --config <abs path> --jobs-root <abs path> --options <json>

It is a separate process for three reasons, all stated in section 1 of the
plan and all worth repeating where the code is:

**Crash isolation.** ``edit/track.py`` drives MediaPipe and OpenCV in-process.
An access violation there takes the interpreter down with it, and the whole
point of this feature is walking away and coming back - coming back to a dead
server is the failure that matters most. Here it takes down one job.

**Cancellation.** Threads cannot be killed. ``on_progress`` returns ``None``
and the pipeline discards it, so a callback cannot signal stop, and
``media.run`` uses ``subprocess.run`` and so exposes no handle to terminate.
Killing the process tree is the only way to stop a render, and that needs a
tree to kill.

**Contention.** ``detect_faces`` is a pure-Python frame loop and PySceneDetect
does a second full decode. Both hold the GIL, and both would make the page you
are watching crawl.

### The four stages, driven here rather than by ``run_all``

``pipeline.run_all`` is four calls in a row with nothing between them, and this
needs two things between them: a cancellation check, and the rights gate.
Section 5's cheap failure - stop at ``blocked`` *before* ``transcribe``, having
spent no LLM quota and no CPU - is only expressible if something looks at the
state between ingest and transcribe. So the four calls are made here, in the
same order and with the same arguments ``run_all`` uses. ``pipeline.py`` itself
is untouched, which is the rule phase 2 is not allowed to break.

### Exit codes are the contract

``0`` done · ``1`` failed · ``3`` blocked by rights · ``130`` cancelled.

``3`` is reserved so the supervisor can colour a rights block as its own thing
rather than as a generic failure - it is the correct, cheap outcome, and
showing it beside "ffmpeg died" would train you to ignore it. It matches
``clipper run``'s own exit code for :class:`~clipper.state.RightsBlocked`, so
the CLI and the web layer say the same number for the same event.

### What it writes

``jobs/<video_id>/state.json``, through the ordinary stage machinery, and
``jobs/_queue/events/<entry_id>.jsonl``. Nothing else - in particular it never
touches ``queue.json``, which the supervisor owns. That is the whole of the
single-writer discipline in section 3: no file has two writers, so there is no
lock anywhere in this package. The video id travels back to the supervisor as
a ``resolved`` event, and the rights status as a ``rights`` event.

Its stdout and stderr are redirected by the parent into
``jobs/_queue/logs/worker-<entry_id>.log``, which is where yt-dlp's and
ffmpeg's chatter goes. The event log is a progress feed, not a transcript.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from clipper.server.events import EventLog
from clipper.server.queue import QueueStore, RunOptions
from clipper.server.text import clean_error
from clipper.state import JobStore, RightsBlocked

#: ``0`` done · ``1`` failed · ``3`` blocked by rights · ``130`` cancelled.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 3
EXIT_CANCELLED = 130


class Cancelled(RuntimeError):
    """The cancel sentinel appeared between two stages.

    Graceful cancellation, which is the half that can be caught: the
    supervisor writes ``jobs/_queue/<entry_id>.cancel`` and this is raised at
    the next boundary. A cancel arriving three minutes into a ``-preset slow``
    encode cannot be caught here at all, which is why the supervisor also kills
    the tree after a grace period.
    """


def run_entry(
    entry_id: str,
    *,
    source: str,
    options: RunOptions,
    config_path: Path,
    jobs_root: Path,
    log: EventLog,
    cancel_path: Path,
) -> int:
    """Run one queue entry to a terminal exit code. Never raises.

    ``source`` is a URL or a local file path for an ordinary entry, and a
    *video id* for a ``render_only`` one - the job already exists in that case,
    so there is nothing to resolve and the first three stages are skipped.

    Imports ``pipeline`` lazily, inside the function, so that importing this
    module (which ``tests/test_server_worker.py`` and the supervisor both do)
    does not drag in ffmpeg-adjacent code.
    """
    from clipper import pipeline

    def cancelled() -> bool:
        return cancel_path.is_file()

    def checkpoint() -> None:
        if cancelled():
            raise Cancelled(f"{entry_id} was cancelled")

    def note(stage: str, detail: str = "") -> None:
        log.append(entry_id, "stage", stage, detail)

    on_progress = log.progress_callback(entry_id)

    log.append(entry_id, "started", detail=source)

    try:
        checkpoint()
        ctx = pipeline.Context.build(config_path=config_path, jobs_root=jobs_root)

        if options.render_only:
            video_id = source
            log.append(entry_id, "resolved", detail=video_id)
            store = JobStore(jobs_root)
            if store.exists(video_id):
                state = store.load(video_id)
                if state.stage("ingest").detail.get("source_deleted"):
                    note("ingest", "restoring the deleted source")
                    state = pipeline.ingest(
                        ctx,
                        state.source_url or state.source_path,
                        rights=state.rights_status.value,
                        on_progress=on_progress,
                    )
                    state.stage("ingest").detail.pop("source_deleted", None)
                    store.save(state)
        else:
            note("ingest", "resolving the source")
            entry = QueueStore(jobs_root).get(entry_id)
            store = JobStore(jobs_root)
            state = store.load(entry.video_id) if entry and entry.video_id else None
            if state is None or not state.is_done("ingest") or options.force:
                state = pipeline.ingest(
                    ctx,
                    source,
                    rights=options.rights,
                    force=options.force,
                    on_progress=on_progress,
                )
            video_id = state.video_id

            # The supervisor copies both of these onto the queue row. Until the
            # first one lands, the row is addressable only by entry_id - that
            # is the whole reason entry_id is the identity.
            log.append(entry_id, "resolved", detail=video_id)
            log.append(
                entry_id,
                "rights",
                stage=str(state.rights_status),
                detail=state.rights_note or "",
            )

            # Section 5's cheap failure, and the earliest point it can be
            # made: ``resolve()`` has just produced the channel id and the
            # licence, and nothing expensive has happened yet. Stopping here
            # spends no LLM quota and no CPU, where the same job carried to
            # ``render`` would have burned the download and the transcription
            # first.
            if not state.is_exportable and not options.allow_red:
                log.append(
                    entry_id,
                    "blocked",
                    stage=str(state.rights_status),
                    detail=state.rights_note
                    or (
                        f"rights_status is {state.rights_status}; render, publish and "
                        f"upload all refuse it. Classify this job from its page."
                    ),
                )
                return EXIT_BLOCKED

            checkpoint()
            note("transcribe", "word-level timings")
            pipeline.transcribe(ctx, video_id, force=options.force, on_progress=on_progress)

            checkpoint()
            note("moments", "two LLM passes over the transcript")
            pipeline.moments(ctx, video_id, force=options.force, on_progress=on_progress)

        checkpoint()
        note("render", "cut, reframe, mix, caption, encode")
        rendered = pipeline.render(
            ctx,
            video_id,
            clip_index=options.clip_index,
            approved_only=options.approved_only,
            music=options.music,
            no_music=options.no_music,
            centre_crop=options.centre_crop,
            allow_red=options.allow_red,
            override_reason=options.override_reason,
            force=options.force,
            on_progress=on_progress,
        )

        log.append(
            entry_id,
            "finished",
            stage=video_id,
            detail=f"{len(rendered)} clip{'' if len(rendered) == 1 else 's'} rendered",
        )
        return EXIT_OK

    except Cancelled:
        log.append(entry_id, "cancelled", detail="cancelled between stages")
        return EXIT_CANCELLED

    except RightsBlocked as exc:
        # Reachable despite the gate above: ``render`` gates each clip too, and
        # a ``--rights`` override can be narrower than the job. Same exit code
        # either way, because it is the same event.
        log.append(
            entry_id, "blocked", detail=clean_error(str(exc)) or "blocked by the rights gate"
        )
        return EXIT_BLOCKED

    except BaseException as exc:  # noqa: BLE001 - the process boundary is here
        # Everything becomes an exit code. A traceback is printed to stdout,
        # which the parent has redirected into the raw log, and the one-line
        # form goes to the event log for the page to show.
        traceback.print_exc()
        log.error(entry_id, f"{type(exc).__name__}: {exc}")
        return EXIT_CANCELLED if isinstance(exc, KeyboardInterrupt) else EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    """``argparse``, not Typer.

    This entrypoint is machine-driven and is spawned a few hundred
    milliseconds before it does minutes of work; the CLI's dependency stack is
    not worth importing for it, and its arguments are not meant to be typed.
    Running it by hand for debugging still works, which is why it has ``--help``
    at all.
    """
    parser = argparse.ArgumentParser(
        prog="python -m clipper.server.worker",
        description="Run one queued job. Spawned by clipper's job supervisor.",
    )
    parser.add_argument("--entry", required=True, help="Queue entry id.")
    parser.add_argument("--source", required=True, help="URL, file path, or video id.")
    parser.add_argument("--config", required=True, help="Absolute path to settings.yaml.")
    parser.add_argument("--jobs-root", required=True, help="Absolute path to the jobs root.")
    parser.add_argument("--options", default="{}", help="RunOptions as a JSON object.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run one entry, return its exit code.

    Every path that crosses the process boundary is absolute and every
    argument is passed as a list element - never a shell string - because this
    repo lives under ``…\\New systems\\shorts-clipper\\``. Section 7.

    ``load_dotenv`` is called *here* and deliberately not in the server
    process. The worker is the only half that makes LLM calls, and it is
    spawned as ``python -m``, which bypasses ``cli.main()`` where the CLI loads
    it. The parent also copies its environment across, so this is belt and
    braces for running the worker by hand.
    """
    args = build_parser().parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a declared dep
        pass

    jobs_root = Path(args.jobs_root)
    options = RunOptions.model_validate(json.loads(args.options))
    log = EventLog(jobs_root)
    cancel_path = QueueStore(jobs_root).cancel_path(args.entry)

    code = run_entry(
        args.entry,
        source=args.source,
        options=options,
        config_path=Path(args.config),
        jobs_root=jobs_root,
        log=log,
        cancel_path=cancel_path,
    )
    # Flushed explicitly: the parent reads this file while the process is being
    # killed, and buffered output would be the half that never arrives.
    sys.stdout.flush()
    sys.stderr.flush()
    return code


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())


__all__ = [
    "EXIT_BLOCKED",
    "EXIT_CANCELLED",
    "EXIT_FAILED",
    "EXIT_OK",
    "Cancelled",
    "build_parser",
    "main",
    "run_entry",
]
