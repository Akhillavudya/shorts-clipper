"""Stage orchestration - what each CLI subcommand actually runs.

The CLI parses arguments and prints; this module does the work. Keeping them
apart means a stage can be driven from a test, a notebook, or the review UI in
Phase 4 without going through Typer.

Every function here follows the same shape, and the shape is the contract from
``state.py``:

    if store.should_run(state.stage(name), force=force):
        with store.stage(state, name) as st:
            st.detail[...] = ...
            ...work...

Work that is expensive and idempotent is additionally guarded on its output
file existing, so rerunning ``ingest`` after ``transcribe`` failed does not
re-download the video.

There is no orchestration framework and none is wanted. ``run`` below is a
loop over four function calls that stops at the first one that has not
completed - that is the whole scheduler, and it is enough because the state
file, not the process, is what tracks progress.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clipper import config as config_module
from clipper import media
from clipper.ingest import audio as audio_stage
from clipper.ingest import download
from clipper.llm import Router
from clipper.moments import recall, score, snap
from clipper.rights import registry
from clipper.state import JobState, JobStore, StageStatus
from clipper.transcribe import transcribe as run_transcribe
from clipper.transcribe.schema import Transcript, transcript_path

Progress = Callable[[str, str], None] | None


class StageError(RuntimeError):
    """A stage cannot run because a previous one has not."""


def _note(on_progress: Progress, stage: str, detail: str = "") -> None:
    if on_progress is not None:
        on_progress(stage, detail)


@dataclass
class Context:
    """Everything the stages share: config, the store, and the job id.

    Built once per CLI invocation. The settings object is passed around rather
    than re-read per stage so a config edit mid-run cannot produce a job whose
    first half used different thresholds from its second.
    """

    settings: config_module.Settings
    store: JobStore
    jobs_root: Path

    @classmethod
    def build(
        cls,
        *,
        config_path: Path | str = config_module.SETTINGS_PATH,
        jobs_root: Path | str | None = None,
    ) -> Context:
        settings = config_module.Settings.load(config_path)
        root = Path(jobs_root) if jobs_root is not None else settings.jobs_root
        return cls(settings=settings, store=JobStore(root), jobs_root=root)

    def router(self, video_id: str) -> Router:
        """A router scoped to this job, so the cache and log land in its dir."""
        return Router(self.settings.data, job_dir=self.store.job_dir(video_id))

    def router_for_dir(self, directory: Path | str) -> Router:
        """A router for work that is not a job - discovery, chiefly.

        Discovery has no video id, but it still wants the cache and the call log
        for the same reason every stage does: a rerun during development must
        not respend quota.
        """
        return Router(self.settings.data, job_dir=Path(directory))


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def ingest(
    ctx: Context,
    source: str,
    *,
    rights: str | None = None,
    watchlist_path: Path | str = config_module.WATCHLIST_PATH,
    force: bool = False,
    on_progress: Progress = None,
) -> JobState:
    """Resolve the source, download it, extract audio, classify the rights.

    Resolution happens before the job exists - the video id it returns is what
    names the job directory. That one metadata call is also what gives the
    rights gate a channel id to key off, which is why it is not folded into the
    download.
    """
    _note(on_progress, "resolve", source)
    ref = download.resolve(source)

    state = ctx.store.load_or_create(ref.video_id)
    job_dir = ctx.store.job_dir(ref.video_id)

    if not ctx.store.should_run(state.stage("ingest"), force=force):
        return state

    watchlist = registry.Watchlist.load(watchlist_path)

    with ctx.store.stage(state, "ingest") as st:
        download.apply_metadata(state, ref)

        decision = registry.resolve(
            channel_id=ref.channel_id,
            licence=ref.licence,
            watchlist=watchlist,
            override=rights,
        )
        registry.apply(state, decision)
        st.detail["rights_status"] = decision.status.value
        st.detail["rights_source"] = (
            "override" if rights else ("watchlist" if decision.entry else "licence/unclassified")
        )

        _note(on_progress, "download", ref.url or str(ref.path))
        max_height = ctx.settings.get(
            "ingest.max_source_height", download.DEFAULT_MAX_SOURCE_HEIGHT
        )
        source_path = download.download(
            ref,
            job_dir,
            overwrite=force,
            max_height=int(max_height) if max_height else None,
        )
        st.detail["source"] = str(source_path)
        st.detail["max_source_height"] = max_height

        # Duration is only known for certain after the file is on disk; a local
        # source has no metadata at all until it is probed.
        if state.duration_s is None:
            with contextlib.suppress(media.FFmpegMissing, media.FFmpegError):
                state.duration_s = media.video_info(source_path).duration_s

        _note(on_progress, "audio", "16 kHz mono wav")
        transcription = ctx.settings.transcription
        audio_out = audio_stage.extract_for_job(
            source_path,
            job_dir,
            sample_rate=int(transcription.get("sample_rate", 16000)),
            channels=int(transcription.get("channels", 1)),
            overwrite=force,
        )
        st.detail["audio"] = str(audio_out)
        st.detail["duration_s"] = state.duration_s

    return state


# --------------------------------------------------------------------------
# transcribe
# --------------------------------------------------------------------------


def transcribe(
    ctx: Context,
    video_id: str,
    *,
    backend: str | None = None,
    force: bool = False,
    on_progress: Progress = None,
) -> Transcript:
    """Produce ``transcript.json`` with word-level timings."""
    state = ctx.store.load(video_id)
    job_dir = ctx.store.job_dir(video_id)
    out_path = transcript_path(job_dir)

    if not ctx.store.should_run(state.stage("transcribe"), force=force) and out_path.is_file():
        return Transcript.load(out_path)

    if not state.is_done("ingest"):
        raise StageError(
            f"job {video_id} has not been ingested (stage is "
            f"{state.stage('ingest').status.value}); run `clipper ingest` first"
        )

    audio_path = audio_stage.require_audio(job_dir)
    chosen = backend or ctx.settings.transcription.get("backend", "gemini")

    with ctx.store.stage(state, "transcribe") as st:
        st.detail["backend"] = chosen

        def progress(*args: Any) -> None:
            if len(args) == 4:
                index, total, start, end = args
                _note(
                    on_progress,
                    "transcribe",
                    f"window {index}/{total} ({start:.0f}s - {end:.0f}s)",
                )
            else:
                _note(on_progress, "transcribe", " ".join(str(a) for a in args))

        transcript = run_transcribe(
            audio_path, ctx.settings.data, backend=chosen, on_progress=progress
        )
        transcript.save(out_path)

        st.detail["words"] = transcript.word_count
        st.detail["speakers"] = transcript.speakers()
        st.detail["language"] = transcript.language

    return transcript


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------


def moments(
    ctx: Context,
    video_id: str,
    *,
    top: int | None = None,
    force: bool = False,
    on_progress: Progress = None,
) -> list[score.ScoredClip]:
    """Two LLM passes, then deterministic snapping. Populates ``state.clips``."""
    state = ctx.store.load(video_id)
    job_dir = ctx.store.job_dir(video_id)

    if not state.is_done("transcribe"):
        raise StageError(
            f"job {video_id} has no transcript (stage is "
            f"{state.stage('transcribe').status.value}); run `clipper transcribe` first"
        )

    if not ctx.store.should_run(state.stage("moments"), force=force):
        return []

    transcript = Transcript.load(transcript_path(job_dir))
    moments_cfg = ctx.settings.moments
    weights_cfg = config_module.load_scoring_weights()
    router = ctx.router(video_id)

    with ctx.store.stage(state, "moments") as st:
        # -- pass 1 -----------------------------------------------------
        def recall_progress(window: recall.Window, total: int) -> None:
            _note(on_progress, "recall", f"window {window.number}/{total}")

        candidates = recall.run(
            transcript,
            router,
            title=state.title,
            chunk_minutes=float(moments_cfg.get("chunk_minutes", 10)),
            overlap_s=float(moments_cfg.get("chunk_overlap_s", 60)),
            target_candidates=int(moments_cfg.get("target_candidates", 40)),
            on_progress=recall_progress,
        )
        recall.save(candidates, job_dir)
        st.detail["candidates"] = len(candidates)

        if not candidates:
            st.detail["note"] = "pass 1 returned nothing usable"
            state.clips = []
            return []

        # -- pass 2 -----------------------------------------------------
        _note(on_progress, "rerank", f"{len(candidates)} candidates")
        clip_length = moments_cfg.get("clip_length") or {}
        scored = score.run(
            candidates,
            transcript,
            router,
            weights_cfg,
            title=state.title,
            top_n=int(top or moments_cfg.get("rerank_top_n", 20)),
            clip_length=clip_length,
        )
        score.save(scored, job_dir, weights_config=weights_cfg)
        st.detail["scored"] = len(scored)

        # -- snapping ---------------------------------------------------
        shortlisted = score.shortlist(scored, weights_cfg)
        st.detail["shortlisted"] = len(shortlisted)
        st.detail["min_score"] = weights_cfg.get("min_score")

        _note(on_progress, "snap", "detecting silences")
        silences = snap.load_silences(
            audio_stage.audio_path(job_dir),
            threshold_db=float(moments_cfg.get("silence_threshold_db", -30)),
            min_silence_ms=int(moments_cfg.get("min_silence_ms", 250)),
        )
        st.detail["silences"] = len(silences)
        if not silences:
            st.detail["snap_note"] = (
                "no silences detected (ffmpeg missing, or continuous speech); "
                "boundaries fall back to sentence extension"
            )

        state.clips = []
        for index, clip in enumerate(shortlisted, start=1):
            snapped = snap.snap(
                transcript,
                clip.candidate.start_word_index,
                clip.candidate.end_word_index,
                silences=silences,
                pad_ms=int(moments_cfg.get("pad_ms", 150)),
                max_duration_s=float(clip_length.get("max", 55)) + 10.0,
            )
            state.add_clip(
                index=index,
                start_word_index=snapped.start_word_index,
                end_word_index=snapped.end_word_index,
                start=snapped.start,
                end=snapped.end,
                type=clip.candidate.type,
                reason=clip.candidate.one_line_reason,
                rubric=clip.rubric,
                score=clip.score,
                suggested_title=clip.suggested_title,
                emphasis_words=clip.emphasis_words,
                hook_rewrite=clip.hook_rewrite,
            )

    return shortlisted


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


def render(
    ctx: Context,
    video_id: str,
    *,
    clip_index: int | None = None,
    approved_only: bool = False,
    music: str | None = None,
    no_music: bool = False,
    centre_crop: bool = False,
    allow_red: bool = False,
    override_reason: str | None = None,
    force: bool = False,
    on_progress: Progress = None,
) -> list[Any]:
    """Render one clip, or every clip the review stage did not reject.

    ``approved_only`` narrows that to clips explicitly kept in the review UI,
    which is the pairing that makes Phase 4 worth running: review first, then
    encode only what survived.

    Imported lazily: ``edit/render.py`` pulls the whole edit package, and
    ``clipper moments`` on a machine without ffmpeg should not have to.
    """
    from clipper.edit import audio_mix, track
    from clipper.edit import render as render_module

    state = ctx.store.load(video_id)
    job_dir = ctx.store.job_dir(video_id)

    if not state.clips:
        raise StageError(f"job {video_id} has no clips; run `clipper moments` first")

    transcript = Transcript.load(transcript_path(job_dir))
    source = download.source_video_path(ctx.store, state)
    edit_cfg = ctx.settings.edit
    caption_style = config_module.load_caption_style()
    music_plan = audio_mix.plan_from_config(
        edit_cfg,
        music_dir=ctx.settings.music_dir,
        override=music,
        disabled=no_music,
    )
    tracking = track.plan_from_config(edit_cfg, disabled=centre_crop)

    if clip_index is not None:
        selected = [c for c in state.clips if c.index == clip_index]
        if not selected:
            available = ", ".join(str(c.index) for c in state.clips) or "none"
            raise StageError(f"job {video_id} has no clip {clip_index} (have: {available})")
    elif approved_only:
        selected = [c for c in state.clips if c.approved is True]
        if not selected:
            raise StageError(
                f"job {video_id} has no approved clips. Review them with "
                f"`clipper review {video_id}`, or drop --approved-only to render "
                f"everything that was not rejected."
            )
    else:
        # ``approved is None`` means "not reviewed yet", which is still the
        # normal state for a job that went straight from `moments` to `render`.
        # Only an explicit rejection is skipped; --approved-only is the opt-in
        # to the stricter reading.
        selected = [c for c in state.clips if c.approved is not False]

    results = []
    with ctx.store.stage(state, "render") as st:
        st.detail["clips"] = [c.clip_id for c in selected]
        st.detail["allow_red"] = allow_red
        st.detail["music"] = str(music_plan.track) if music_plan else None
        st.detail["reframe"] = "track" if tracking else "centre"

        for clip in selected:

            def clip_progress(stage: str, detail: str, _id: str = clip.clip_id) -> None:
                _note(on_progress, f"{_id}/{stage}", detail)

            results.append(
                render_module.render_clip(
                    ctx.store,
                    state,
                    clip,
                    source=source,
                    transcript=transcript,
                    edit_config=edit_cfg,
                    caption_style=caption_style,
                    fonts_dir=ctx.settings.fonts_dir,
                    music=music_plan,
                    tracking=tracking,
                    allow_red=allow_red,
                    override_reason=override_reason,
                    force=force,
                    on_progress=clip_progress,
                )
            )

        st.detail["rendered"] = len(results)

    return results


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def run_all(
    ctx: Context,
    source: str,
    *,
    auto: bool = False,
    rights: str | None = None,
    music: str | None = None,
    no_music: bool = False,
    centre_crop: bool = False,
    allow_red: bool = False,
    force: bool = False,
    on_progress: Progress = None,
) -> JobState:
    """Every stage end to end, resuming from the first incomplete one.

    Without ``auto`` it stops after ``moments``, which is the intended default:
    section 12 Phase 4 makes human review the norm, and rendering an unreviewed
    clip wastes the most expensive stage on something that will be rejected.
    """
    state = ingest(ctx, source, rights=rights, force=force, on_progress=on_progress)
    video_id = state.video_id

    transcribe(ctx, video_id, force=force, on_progress=on_progress)
    moments(ctx, video_id, force=force, on_progress=on_progress)

    state = ctx.store.load(video_id)
    if not auto:
        state.stage("render").status = StageStatus.pending
        ctx.store.save(state)
        return state

    render(
        ctx,
        video_id,
        music=music,
        no_music=no_music,
        centre_crop=centre_crop,
        allow_red=allow_red,
        force=force,
        on_progress=on_progress,
    )
    return ctx.store.load(video_id)


# --------------------------------------------------------------------------
# discover
# --------------------------------------------------------------------------


@dataclass
class DiscoveryResult:
    """One watchlist scan: what to look at, what it cost, and what it skipped."""

    candidates: list[Any] = dataclass_field(default_factory=list)
    report_path: Path | None = None
    quota: dict[str, Any] = dataclass_field(default_factory=dict)
    skipped: dict[str, int] = dataclass_field(default_factory=dict)
    errors: dict[str, str] = dataclass_field(default_factory=dict)
    episodes: list[Any] = dataclass_field(default_factory=list)
    channels_scanned: int = 0
    blocked_hidden: int = 0


def discover(
    ctx: Context,
    *,
    limit: int | None = None,
    watchlist_path: Path | str = config_module.WATCHLIST_PATH,
    per_channel: int = 5,
    include_blocked: bool = False,
    summarize: bool = True,
    youtube: Any = None,
    podcast_client: Any = None,
    now: datetime | None = None,
    on_progress: Progress = None,
) -> DiscoveryResult:
    """Scan the watchlist and shortlist candidate source videos.

    The order here is the order of FULL_PLAN.md section 3, and the order is the
    design: every cheap deterministic filter runs before the one LLM call, so
    the model only ever sees a pre-filtered shortlist of about fifteen.

    Non-exportable channels are dropped from the shortlist by default. A red
    entry can never be rendered, so surfacing it as a suggestion is an invitation
    to override the gate; ``include_blocked`` shows them anyway, labelled, for
    when the watchlist itself is what is being reviewed.

    Unlike the four per-job stages this writes no job state - discovery happens
    before a job exists. Its output is a dated report you read and pick from.
    """
    from clipper.discover import summarize as summarize_module
    from clipper.discover import velocity
    from clipper.discover import youtube as youtube_module

    now = now or datetime.now(UTC)
    watchlist = registry.Watchlist.load(watchlist_path)
    cfg = ctx.settings.discovery

    channels = watchlist.channels
    hidden = [] if include_blocked else [c for c in channels if not c.is_exportable]
    if hidden:
        channels = [c for c in channels if c.is_exportable]

    result = DiscoveryResult(channels_scanned=len(channels), blocked_hidden=len(hidden))
    directory = youtube_module.cache_dir(ctx.jobs_root)

    if not channels:
        result.errors["watchlist"] = (
            f"{watchlist_path} lists no scannable channels"
            + (f"; {len(hidden)} are red or unknown and were hidden" if hidden else "")
            + ". config/watchlist.yaml is the highest-leverage file in the "
            "project, and discovery does nothing until it has entries."
        )
        return result

    client = youtube or youtube_module.YouTubeClient(
        ledger=youtube_module.ledger_for(
            ctx.jobs_root, ceiling=int(cfg.get("daily_quota_ceiling", 2000))
        ),
        playlist_cache=youtube_module.playlist_cache_for(ctx.jobs_root),
    )

    def scan_progress(channel_id: str, found: int) -> None:
        _note(on_progress, "scan", f"{channel_id}: {found} recent uploads")

    scan = client.scan_channels(
        [c.id for c in channels], per_channel=per_channel, on_progress=scan_progress
    )
    result.quota = scan.quota
    result.errors.update(scan.errors)

    # -- deterministic ranking ------------------------------------------
    scores, skipped = velocity.score_records(
        scan.records,
        now=now,
        min_age_hours=float(cfg.get("min_video_age_hours", 6)),
        min_channel_uploads=int(cfg.get("min_channel_uploads", 5)),
        min_duration_s=float(watchlist.defaults.get("min_duration_s", 900)),
        per_channel_min_duration={c.id: float(watchlist.min_duration_s(c.id)) for c in channels},
    )
    result.skipped = velocity.summarize_skips(skipped)

    scores = velocity.cap_per_channel(
        scores,
        {c.id: watchlist.max_per_day(c.id) for c in channels},
        default=int(watchlist.defaults.get("max_per_day", 2)),
    )

    top_n = int(limit or cfg.get("max_candidates", 15))
    candidates = []
    for item in scores[:top_n]:
        decision = registry.resolve(
            channel_id=item.record.channel_id,
            licence=item.record.licence,
            watchlist=watchlist,
        )
        candidates.append(
            summarize_module.Candidate(
                score=item,
                rights_status=decision.status.value,
                rights_note=decision.note,
            )
        )

    # -- the one LLM call -----------------------------------------------
    if summarize and candidates:
        _note(on_progress, "summarize", f"{len(candidates)} candidates")
        try:
            summarize_module.run(
                candidates, ctx.router_for_dir(directory), channel_count=len(channels)
            )
        except Exception as exc:  # noqa: BLE001 - the blurb is optional, the list is not
            result.errors["summarize"] = f"{type(exc).__name__}: {exc}"

    # -- RSS side, when the watchlist names feeds ------------------------
    feed_ids = [
        str(c.raw.get("podcast_feed_id"))
        for c in channels
        if c.raw and c.raw.get("podcast_feed_id")
    ]
    if feed_ids:
        from clipper.discover.podcast_index import PodcastIndexClient, PodcastIndexError

        podcast = podcast_client or PodcastIndexClient()
        if podcast.configured:
            _note(on_progress, "podcast-index", f"{len(feed_ids)} feeds")
            try:
                result.episodes = podcast.scan_feeds(
                    feed_ids,
                    min_duration_s=float(watchlist.defaults.get("min_duration_s", 900)),
                    now=now,
                )
            except PodcastIndexError as exc:
                result.errors["podcast_index"] = str(exc)

    result.candidates = candidates
    result.report_path = summarize_module.save(
        candidates,
        directory,
        quota=result.quota,
        skipped=result.skipped,
        errors=result.errors,
        episodes=[e.to_dict() for e in result.episodes],
        now=now,
    )
    return result


# --------------------------------------------------------------------------
# publish
# --------------------------------------------------------------------------


def publish(
    ctx: Context,
    video_id: str,
    *,
    clip_index: int | None = None,
    approved_only: bool = False,
    published_id: str | None = None,
    watchlist_path: Path | str = config_module.WATCHLIST_PATH,
    allow_red: bool = False,
    override_reason: str | None = None,
) -> list[Any]:
    """Build ``upload.json`` for rendered clips, and record a published id.

    The rights gate fires again here. ``edit/render.py`` gates the file; this
    gates the title, description and credit that would accompany it, which is
    the other half of what an upload actually is.
    """
    from clipper.publish import metadata as metadata_module

    state = ctx.store.load(video_id)
    if not state.clips:
        raise StageError(f"job {video_id} has no clips; run `clipper moments` first")

    if clip_index is not None:
        selected = [c for c in state.clips if c.index == clip_index]
        if not selected:
            available = ", ".join(str(c.index) for c in state.clips) or "none"
            raise StageError(f"job {video_id} has no clip {clip_index} (have: {available})")
    elif approved_only:
        selected = [c for c in state.clips if c.approved is True]
        if not selected:
            raise StageError(
                f"job {video_id} has no approved clips; review them with "
                f"`clipper review {video_id}`"
            )
    else:
        selected = [c for c in state.clips if c.approved is not False]

    if published_id and len(selected) != 1:
        raise StageError(
            "--published records one YouTube video id against one clip; pass "
            "--clip N to say which one."
        )

    watchlist = registry.Watchlist.load(watchlist_path)
    publish_cfg = ctx.settings.section("publish")

    built: list[Any] = []
    for clip in selected:
        meta = metadata_module.build(
            state,
            clip,
            config=publish_cfg,
            watchlist=watchlist,
            allow_red=allow_red,
            override_reason=override_reason,
        )
        metadata_module.save(meta, ctx.store.clip_dir(video_id, clip.clip_id))
        if published_id:
            metadata_module.record_published(clip, published_id)
        built.append(meta)

    ctx.store.save(state)
    return built


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------


def upload(
    ctx: Context,
    video_id: str,
    *,
    clip_index: int,
    watchlist_path: Path | str = config_module.WATCHLIST_PATH,
    allow_red: bool = False,
    override_reason: str | None = None,
    privacy: str | None = None,
    notify_subscribers: bool = False,
    force: bool = False,
    uploader: Any = None,
    ledger: Any = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Any:
    """Send one rendered clip to YouTube, and record the id it comes back with.

    One clip per call, named explicitly. Everything else in this pipeline is
    reversible by deleting a file and rerunning the stage; this is not, so it
    does not take a set.

    The order is deliberate: gate, then refuse a duplicate, then charge quota,
    then send. Each check is cheaper than the one after it, and the last thing
    to happen is the only thing that cannot be undone.
    """
    from clipper.discover import youtube as youtube_module
    from clipper.publish import metadata as metadata_module
    from clipper.publish import upload as upload_module

    state = ctx.store.load(video_id)
    clip = next((c for c in state.clips if c.index == clip_index), None)
    if clip is None:
        available = ", ".join(str(c.index) for c in state.clips) or "none"
        raise StageError(f"job {video_id} has no clip {clip_index} (have: {available})")

    # The gate, for the third and last time. render gated the file, publish
    # gated the text, and this is the step that is actually irreversible.
    registry.check_export(state, allow_red=allow_red, reason=override_reason)
    ctx.store.save(state)

    if clip.youtube_video_id and not force:
        raise StageError(
            f"{clip.clip_id} is already published as {clip.youtube_video_id} "
            f"({upload_module.WATCH_URL.format(video_id=clip.youtube_video_id)}). "
            f"Pass --force to upload it a second time."
        )

    clip_dir = ctx.store.clip_dir(video_id, clip.clip_id)
    final_path = clip_dir / "final.mp4"
    if not final_path.is_file():
        raise StageError(f"{final_path} does not exist; run `clipper render {video_id}` first")
    file_size = final_path.stat().st_size
    if file_size == 0:
        raise StageError(f"{final_path} is empty; re-render before uploading it")

    # The metadata on disk wins. `publish` wrote it, and a title edited by hand
    # between publish and upload is an edit, not a stale file to overwrite.
    if (clip_dir / metadata_module.UPLOAD_FILENAME).is_file():
        raw = upload_module.read_upload_json(clip_dir)
    else:
        meta = metadata_module.build(
            state,
            clip,
            config=ctx.settings.section("publish"),
            watchlist=registry.Watchlist.load(watchlist_path),
            allow_red=allow_red,
            override_reason=override_reason,
        )
        metadata_module.save(meta, clip_dir)
        raw = meta.to_dict()

    body = upload_module.insert_body(raw)
    if privacy:
        body["status"]["privacyStatus"] = privacy

    upload_cfg = config_module.get(ctx.settings.data, "publish.upload", {}) or {}
    quota_cost = int(upload_cfg.get("quota_cost", upload_module.UPLOAD_QUOTA_COST))

    if ledger is None:
        ledger = youtube_module.ledger_for(
            ctx.jobs_root,
            ceiling=int(upload_cfg.get("daily_quota_ceiling", 8000)),
        )

    owns_uploader = uploader is None
    if owns_uploader:
        uploader = upload_module.ResumableUpload(
            chunk_bytes=int(upload_cfg.get("chunk_mb", 8)) * 1024 * 1024,
            max_attempts=int(upload_cfg.get("max_attempts", 5)),
            timeout_s=float(upload_cfg.get("timeout_s", 300.0)),
        )

    try:
        with ctx.store.stage(state, "upload", clip_id=clip.clip_id) as st:
            st.detail["bytes"] = file_size
            st.detail["privacy_status"] = body["status"].get("privacyStatus")

            # Charged before the session opens, not after it succeeds: Google
            # bills the attempt, and a failure that leaves the ledger untouched
            # is how a retry loop quietly spends the day's whole budget.
            try:
                ledger.spend(quota_cost, resource="videos.insert")
            except youtube_module.QuotaExceeded as exc:
                raise StageError(
                    f"one upload costs {quota_cost} of the project's 10,000 "
                    f"Data API units/day, shared with discovery, and the ceiling "
                    f"is already reached: {exc}"
                ) from exc
            st.detail["quota_units"] = quota_cost

            session, start_at = _upload_session(
                uploader,
                upload_module,
                clip_dir,
                final_path,
                body,
                file_size,
                notify_subscribers=notify_subscribers,
            )
            st.detail["resumed_from"] = start_at

            try:
                resource = uploader.send(
                    final_path, session.uri, start_at=start_at, on_progress=on_progress
                )
            except upload_module.SessionExpired:
                # The session died mid-transfer. One clean restart, because the
                # alternative is leaving a rendered clip that can never upload.
                upload_module.clear_session(clip_dir)
                session, start_at = _upload_session(
                    uploader,
                    upload_module,
                    clip_dir,
                    final_path,
                    body,
                    file_size,
                    notify_subscribers=notify_subscribers,
                    fresh=True,
                )
                resource = uploader.send(
                    final_path, session.uri, start_at=start_at, on_progress=on_progress
                )

            upload_module.clear_session(clip_dir)

            youtube_video_id = str(resource.get("id") or "")
            metadata_module.record_published(clip, youtube_video_id)
            st.detail["youtube_video_id"] = youtube_video_id

            result = upload_module.UploadResult(
                video_id=youtube_video_id,
                clip_id=clip.clip_id,
                privacy_status=str(
                    (resource.get("status") or {}).get("privacyStatus")
                    or body["status"].get("privacyStatus", "")
                ),
                title=str((resource.get("snippet") or {}).get("title") or ""),
                bytes_sent=file_size - start_at,
                file_size=file_size,
                resumed_from=start_at,
                attempts=getattr(uploader, "attempts", 0),
                response=resource,
            )
    finally:
        if owns_uploader:
            uploader.close()

    return result


def _upload_session(
    uploader: Any,
    upload_module: Any,
    clip_dir: Path,
    final_path: Path,
    body: dict[str, Any],
    file_size: int,
    *,
    notify_subscribers: bool = False,
    fresh: bool = False,
) -> tuple[Any, int]:
    """Resume the session saved beside the clip, or open a new one.

    Resuming asks the server for the offset rather than trusting anything we
    wrote down: the only byte count that matters is the one the server has.
    """
    if not fresh:
        saved = upload_module.load_session(clip_dir)
        if saved is not None and saved.matches(final_path, now=time.time()):
            try:
                where = uploader.offset(saved.uri, file_size)
            except upload_module.SessionExpired:
                upload_module.clear_session(clip_dir)
            else:
                if isinstance(where, dict):
                    # Already complete - a previous run died after the last
                    # chunk landed. Nothing left to send.
                    return saved, file_size
                return saved, where

    uri = uploader.start(body, file_size=file_size, notify_subscribers=notify_subscribers)
    session = upload_module.Session.opening(uri, final_path, now=time.time())
    upload_module.save_session(clip_dir, session)
    return session, 0


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


def calibrate(
    ctx: Context,
    *,
    apply: bool = False,
    days: int = 90,
    weights_path: Path | str = config_module.SCORING_WEIGHTS_PATH,
    client: Any = None,
) -> Any:
    """Correlate predicted rubric scores against real retention.

    Reads across every job rather than one - the sample is the whole point.
    Returns a report; it writes ``scoring_weights.yaml`` only when ``apply`` is
    set and the fit actually predicts better than the weights in force.
    """
    from clipper.calibrate import analytics

    return analytics.run(
        ctx.store,
        config_module.load_scoring_weights(weights_path),
        client=client,
        weights_path=weights_path,
        apply=apply,
        days=days,
    )


__all__ = [
    "Context",
    "DiscoveryResult",
    "StageError",
    "calibrate",
    "discover",
    "ingest",
    "moments",
    "publish",
    "render",
    "run_all",
    "transcribe",
]
