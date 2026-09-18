"""Speaker-tracked crop path (FULL_PLAN.md section 6, step 2 - Phase 2).

``reframe.py`` decides the *shape* of the crop: a 9:16 rectangle, full height,
as wide as the source allows. This module decides where that rectangle sits
horizontally over time, so a talking head that sits to one side of a 16:9
frame does not sit outside the Short.

The pipeline the plan asks for, in order:

1. Shot boundaries from PySceneDetect. Tracking state resets hard at a cut -
   smoothing across one produces a pan that does not exist in the source.
2. Face detection every ``detect_interval_s`` (0.25s by default). MediaPipe's
   BlazeFace is enough for talking heads.
3. Pick the active speaker: biggest face, biased toward the centre, with
   hysteresis so two similarly sized faces do not make the frame flap.
4. Smooth the centre path - EMA plus a deadzone, so a head that moves by a few
   pixels does not move the camera at all.
5. Emit the path as an ffmpeg ``sendcmd`` script driving the crop filter.

**Why sendcmd rather than piping frames through OpenCV.** Piping means
decoding, cropping and re-encoding in Python at 2560x1440; sendcmd means
ffmpeg does what it already does and the tracked path costs one extra text
file. It also keeps the tracked and untracked paths on the same encode, so a
regression between them can only come from the geometry.

Every value written into that script is a *constant*, never an expression.
``crop``'s ``eval`` option is gone in ffmpeg 8 and defaulted to ``init`` in 7,
so an expression in ``t`` would be evaluated once on one build and per frame
on another. A constant re-applied on a fine grid behaves identically on both.

Detection is imported lazily and its absence is not fatal: ``plan_path``
raises :class:`TrackingUnavailable`, and ``reframe`` falls back to the centre
crop and records why. That mirrors ``moments``, which degrades to sentence
extension without ffmpeg rather than refusing to run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from clipper.edit.reframe import CropPlan

COMMAND_FILENAME = "crop_path.cmd"

# The crop filter is given an instance name so sendcmd has something to
# address. Anything unique works; this one says what it is in a -vf string
# that is otherwise hard to read.
FILTER_NAME = "crop@dyn"

DEFAULT_MODEL = Path("assets/models/blaze_face_full_range.tflite")
DEFAULT_INTERVAL_S = 0.25
DEFAULT_ALPHA = 0.15
DEFAULT_DEADZONE_PX = 40
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_SCENE_THRESHOLD = 27.0
DEFAULT_MIN_COVERAGE = 0.35

# Frames are downscaled before detection. BlazeFace resamples its input to
# 192x192 regardless, so full-resolution frames buy nothing; 640 keeps a 250px
# face at ~60px, well inside range.
DETECT_WIDTH = 640

# Active-speaker weighting. A face is worth picking for being big - the near
# speaker in a two-shot is usually the one being listened to - with a centre
# bias layered on top.
CENTRE_BIAS = 0.35
# ...and worth more again for being the one picked last time. Without this,
# two faces of near-equal size trade the frame back and forth every sample.
STICKINESS = 0.25
STICKY_FRACTION = 0.08  # "near the previous pick", as a fraction of the width


class TrackingUnavailable(RuntimeError):
    """Face tracking cannot run here. Carries the reason for the state file."""


@dataclass(frozen=True)
class Face:
    """One detection, in source-frame pixels."""

    x: float
    y: float
    width: float
    height: float
    score: float = 1.0

    @property
    def centre_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def centre_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class FaceSample:
    """What was visible at one instant. ``faces`` may be empty."""

    t: float
    faces: tuple[Face, ...] = ()


@dataclass(frozen=True)
class CropPath:
    """A horizontal crop path, and enough provenance to distrust it.

    ``keys`` is ``(time, crop_x)`` in seconds and source pixels, already
    clamped and made even. ``coverage`` is the fraction of samples that had a
    usable face: a path built from three detections across forty seconds is
    not a tracked camera, it is noise, and the caller checks this before
    using it.
    """

    keys: tuple[tuple[float, int], ...]
    samples: int = 0
    detections: int = 0
    scene_cuts: tuple[float, ...] = ()
    interval_s: float = DEFAULT_INTERVAL_S

    @property
    def coverage(self) -> float:
        return self.detections / self.samples if self.samples else 0.0

    @property
    def travel_px(self) -> int:
        """Total horizontal distance the crop travels. 0 is a static crop."""
        return sum(abs(b - a) for (_, a), (_, b) in zip(self.keys, self.keys[1:], strict=False))

    def sendcmd(self, *, filter_name: str = FILTER_NAME) -> str:
        """The path as a sendcmd script."""
        return to_sendcmd(self.keys, filter_name=filter_name)


@dataclass(frozen=True)
class TrackingOptions:
    """The ``edit:`` block's tracking settings, read once per render."""

    model_path: Path = DEFAULT_MODEL
    interval_s: float = DEFAULT_INTERVAL_S
    alpha: float = DEFAULT_ALPHA
    deadzone_px: int = DEFAULT_DEADZONE_PX
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    min_coverage: float = DEFAULT_MIN_COVERAGE


def options_from_config(edit_config: dict[str, Any]) -> TrackingOptions:
    """Read the ``edit:`` block. Every key falls back to the constant above."""
    return TrackingOptions(
        model_path=Path(edit_config.get("face_model") or DEFAULT_MODEL),
        interval_s=float(edit_config.get("detect_interval_s", DEFAULT_INTERVAL_S)),
        alpha=float(edit_config.get("smoothing_alpha", DEFAULT_ALPHA)),
        deadzone_px=int(edit_config.get("deadzone_px", DEFAULT_DEADZONE_PX)),
        min_confidence=float(edit_config.get("face_min_confidence", DEFAULT_MIN_CONFIDENCE)),
        scene_threshold=float(edit_config.get("scene_threshold", DEFAULT_SCENE_THRESHOLD)),
        min_coverage=float(edit_config.get("min_face_coverage", DEFAULT_MIN_COVERAGE)),
    )


def tracking_enabled(edit_config: dict[str, Any]) -> bool:
    """Whether ``edit.reframe`` asks for the tracked path.

    Only the exact word ``centre`` turns it off. An unrecognised value tracks,
    because the alternative is a typo silently downgrading every render to a
    centre crop, which nobody would notice.
    """
    return str(edit_config.get("reframe", "track")).strip().lower() != "centre"


def plan_from_config(
    edit_config: dict[str, Any], *, disabled: bool = False
) -> TrackingOptions | None:
    """Read the ``edit:`` block into options, or ``None`` for a centre crop.

    ``disabled`` is the CLI's ``--centre-crop`` and wins over the config, the
    same way ``--no-music`` does.
    """
    if disabled or not tracking_enabled(edit_config):
        return None
    return options_from_config(edit_config)


# --------------------------------------------------------------------------
# Picking a face
# --------------------------------------------------------------------------


def face_weight(
    face: Face,
    *,
    frame_width: int,
    previous_centre_x: float | None = None,
) -> float:
    """How much this face deserves the frame.

    Size is ``sqrt(area)``, not area, so a face twice as wide is worth twice
    as much rather than four times: squared units make the winner change on a
    lean toward the camera.
    """
    if frame_width <= 0:
        return 0.0
    relative_size = math.sqrt(max(0.0, face.area)) / frame_width
    mid = frame_width / 2.0
    centrality = max(0.0, 1.0 - abs(face.centre_x - mid) / mid)
    weight = max(0.0, face.score) * relative_size * (1.0 + CENTRE_BIAS * centrality)
    if (
        previous_centre_x is not None
        and abs(face.centre_x - previous_centre_x) <= STICKY_FRACTION * frame_width
    ):
        weight *= 1.0 + STICKINESS
    return weight


def pick_face(
    faces: tuple[Face, ...] | list[Face],
    *,
    frame_width: int,
    previous_centre_x: float | None = None,
) -> Face | None:
    """The active speaker, or ``None`` when nothing was detected."""
    best: Face | None = None
    best_weight = 0.0
    for face in faces:
        weight = face_weight(face, frame_width=frame_width, previous_centre_x=previous_centre_x)
        if best is None or weight > best_weight:
            best, best_weight = face, weight
    return best


# --------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------


def smooth_centres(
    samples: list[FaceSample],
    *,
    frame_width: int,
    scene_cuts: tuple[float, ...] | list[float] = (),
    alpha: float = DEFAULT_ALPHA,
    deadzone_px: int = DEFAULT_DEADZONE_PX,
    start_centre_x: float | None = None,
) -> list[tuple[float, float]]:
    """Turn per-sample detections into a smoothed centre path.

    Four rules, in the order they fire:

    * **First sighting.** The very first face found is adopted outright rather
      than eased toward. There is no earlier frame for the crop to be
      consistent with, and easing in from frame centre instead opens every
      Short with a two-second drift onto the speaker.
    * **Scene cut.** The camera is allowed to teleport, because the source
      already did. State resets: the new face is adopted outright and the
      previous pick stops influencing which face is chosen.
    * **Deadzone.** A target within ``deadzone_px`` of where the crop already
      is moves the crop not at all. This is what stops a talking head's normal
      sway from producing a permanently drifting frame.
    * **EMA.** Outside the deadzone the crop moves ``alpha`` of the way to the
      target per sample. Lower is smoother and laggier.

    A sample with no face holds the current position rather than recentring. A
    face lost for a few frames is the common case - a turn of the head, a hand
    in front of the mouth - and recentring on each one is worse than sitting
    still.

    ``start_centre_x`` seeds the path, and counts as a first sighting: give it
    and the opening frames ease from there instead of snapping.
    """
    alpha = min(1.0, max(0.0, alpha))
    cuts = sorted(float(c) for c in scene_cuts)
    default_centre = frame_width / 2.0
    current = default_centre if start_centre_x is None else float(start_centre_x)
    settled = start_centre_x is not None
    last_pick: float | None = None
    cut_index = 0
    previous_t: float | None = None

    path: list[tuple[float, float]] = []
    for sample in samples:
        crossed = False
        while cut_index < len(cuts) and cuts[cut_index] <= sample.t:
            # A cut at or before the first sample is the start of the file,
            # not a change of shot.
            if previous_t is not None and cuts[cut_index] > previous_t:
                crossed = True
            cut_index += 1
        if crossed:
            last_pick = None

        face = pick_face(sample.faces, frame_width=frame_width, previous_centre_x=last_pick)
        if face is not None:
            target = face.centre_x
            last_pick = target
            if crossed or not settled:
                current = target  # hard reset: no pan across a cut, or into one
            elif abs(target - current) > deadzone_px:
                current += alpha * (target - current)
            settled = True
        elif crossed:
            # New shot, nobody in it. Centre is the only defensible guess.
            current = default_centre

        path.append((sample.t, current))
        previous_t = sample.t

    return path


# --------------------------------------------------------------------------
# Turning a centre path into crop x commands
# --------------------------------------------------------------------------


def crop_x_for(centre_x: float, *, crop_width: int, src_width: int) -> int:
    """Clamp a centre of interest into a legal, even crop offset.

    Odd offsets are rejected outright by ``yuv420p``, and here the failure
    would arrive mid-encode as a command that ffmpeg refuses.
    """
    x = int(round(centre_x - crop_width / 2.0))
    x = max(0, min(max(0, src_width - crop_width), x))
    return x - (x % 2)


def build_keys(
    centres: list[tuple[float, float]],
    *,
    crop_width: int,
    src_width: int,
    fps: float,
    duration_s: float | None = None,
    scene_cuts: tuple[float, ...] | list[float] = (),
) -> tuple[tuple[float, int], ...]:
    """Resample the centre path onto the frame grid and emit crop offsets.

    Detections are 0.25s apart; frames are 33ms apart. One command per
    detection would move the crop in 250ms steps, which reads as a stutter
    exactly when the subject is moving fastest. Linear interpolation between
    samples turns each step into a pan.

    ``scene_cuts`` is what keeps that interpolation honest. ``smooth_centres``
    snaps the crop at a cut, but the snap lands on the 0.25s sample grid, and
    interpolating through it would smear an instant cut into a quarter-second
    whip pan. A segment containing a cut steps at the cut instead of ramping
    across it, which also puts the move on the exact frame the shot changes
    rather than up to 0.25s late.

    Consecutive identical offsets are dropped - the filter is already holding
    that value - so a static shot costs one line, not twelve hundred.
    """
    if not centres:
        return ()
    if fps <= 0:
        fps = 30.0

    cuts = sorted(float(c) for c in scene_cuts)
    end = duration_s if duration_s and duration_s > 0 else centres[-1][0]
    step = 1.0 / fps
    keys: list[tuple[float, int]] = []
    last_x: int | None = None

    index = 0
    frame = 0
    while True:
        t = frame * step
        if t > end + step / 2:
            break
        while index + 1 < len(centres) and centres[index + 1][0] <= t:
            index += 1
        if index + 1 < len(centres):
            t0, c0 = centres[index]
            t1, c1 = centres[index + 1]
            cut = next((c for c in cuts if t0 < c <= t1), None)
            if cut is not None:
                centre = c0 if t < cut else c1
            else:
                span = t1 - t0
                ratio = 0.0 if span <= 0 else min(1.0, max(0.0, (t - t0) / span))
                centre = c0 + (c1 - c0) * ratio
        else:
            centre = centres[index][1]
        x = crop_x_for(centre, crop_width=crop_width, src_width=src_width)
        if x != last_x:
            keys.append((round(t, 3), x))
            last_x = x
        frame += 1

    return tuple(keys)


def to_sendcmd(
    keys: tuple[tuple[float, int], ...] | list[tuple[float, int]],
    *,
    filter_name: str = FILTER_NAME,
) -> str:
    """Render crop keys as an ffmpeg sendcmd script.

    One command per line, ``TIME TARGET COMMAND ARG;``. The argument is always
    an integer; see the module docstring on why it is never an expression.
    """
    lines = [f"{t:.3f} {filter_name} x {x};" for t, x in keys]
    return "\n".join(lines) + ("\n" if lines else "")


# --------------------------------------------------------------------------
# Detection (the parts that need the heavy dependencies)
# --------------------------------------------------------------------------


def _require(module: str, package: str) -> Any:
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise TrackingUnavailable(
            f"face tracking needs {package}, which is not importable ({exc}). "
            f"Install it, or set edit.reframe: centre in config/settings.yaml."
        ) from exc


def scene_cuts(
    video: Path | str,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len_s: float = 0.5,
    fps: float = 30.0,
) -> tuple[float, ...]:
    """Shot-boundary times in seconds, from PySceneDetect's content detector.

    Returns the *start* of every scene after the first. A clip with no cuts
    tracks exactly as well, it just never resets, so the caller treats an
    empty tuple and a failure the same way.

    This is the expensive half of tracking, because it decodes the clip a
    second time. Measured on a 43s 2560x1440 clip: 16s here against 8s for
    every face detection in the same file, on top of a 15s static reframe.
    Sharing one decode with :func:`detect_faces` is the obvious win if that
    ever matters.

    ContentDetector cannot report a cut in the first couple of frames - it has
    no previous frame to compare against - so a clip whose cut point lands
    almost immediately gets eased through rather than snapped. The EMA absorbs
    it in about half a second.
    """
    scenedetect = _require("scenedetect", "scenedetect")
    detector = scenedetect.ContentDetector(
        threshold=threshold, min_scene_len=max(1, int(min_scene_len_s * fps))
    )
    scenes = scenedetect.detect(str(video), detector)
    return tuple(start.get_seconds() for start, _ in scenes[1:])


def detect_faces(
    video: Path | str,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    model_path: Path | str = DEFAULT_MODEL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[list[FaceSample], int]:
    """Sample ``video`` every ``interval_s`` and detect faces.

    Returns the samples and the source frame width, which is what the rest of
    this module measures everything in.

    Frames are walked with ``grab()`` and only decoded at sample points.
    Seeking per sample is the obvious implementation and is several times
    slower on a long-GOP h264 file, because every seek decodes forward from a
    keyframe anyway.
    """
    model = Path(model_path)
    if not model.is_file():
        raise TrackingUnavailable(
            f"face model not found at {model}. Fetch it with:\n"
            f"  curl -L -o {model.as_posix()} https://storage.googleapis.com/"
            f"mediapipe-models/face_detector/blaze_face_full_range/float16/1/"
            f"blaze_face_full_range.tflite"
        )

    cv2 = _require("cv2", "opencv-python")
    mp = _require("mediapipe", "mediapipe")
    mp_vision = _require("mediapipe.tasks.python.vision", "mediapipe")
    mp_python = _require("mediapipe.tasks.python", "mediapipe")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise TrackingUnavailable(f"OpenCV could not open {video}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        src_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        stride = max(1, int(round(interval_s * fps)))

        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        detector = mp_vision.FaceDetector.create_from_options(options)

        samples: list[FaceSample] = []
        try:
            index = 0
            while capture.grab():
                if index % stride == 0:
                    ok, frame = capture.retrieve()
                    if ok:
                        samples.append(_detect_one(detector, mp, cv2, frame, t=index / fps))
                index += 1
        finally:
            detector.close()
    finally:
        capture.release()

    return samples, src_width


def _detect_one(detector: Any, mp: Any, cv2: Any, frame: Any, *, t: float) -> FaceSample:
    """Detect on one frame, scaling the boxes back to source pixels."""
    height, width = frame.shape[:2]
    scale = 1.0
    if width > DETECT_WIDTH:
        scale = DETECT_WIDTH / width
        frame = cv2.resize(
            frame,
            (DETECT_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # detect_for_video insists on monotonically increasing integer milliseconds.
    result = detector.detect_for_video(image, int(round(t * 1000)))

    faces: list[Face] = []
    for detection in result.detections or []:
        box = detection.bounding_box
        score = detection.categories[0].score if detection.categories else 1.0
        faces.append(
            Face(
                x=box.origin_x / scale,
                y=box.origin_y / scale,
                width=box.width / scale,
                height=box.height / scale,
                score=float(score),
            )
        )
    return FaceSample(t=t, faces=tuple(faces))


def plan_path(
    video: Path | str,
    plan: CropPlan,
    *,
    options: TrackingOptions | None = None,
    fps: float = 30.0,
    duration_s: float | None = None,
) -> CropPath:
    """Detect, pick, smooth and key one clip's crop path.

    Raises :class:`TrackingUnavailable` if the dependencies or the model are
    missing. Returns a path whose ``coverage`` the caller must check: this
    function will happily build a path out of two detections, and deciding
    that is not good enough belongs to the render stage.
    """
    options = options or TrackingOptions()
    samples, src_width = detect_faces(
        video,
        interval_s=options.interval_s,
        model_path=options.model_path,
        min_confidence=options.min_confidence,
    )
    if src_width <= 0:
        src_width = plan.src_width

    try:
        cuts = scene_cuts(video, threshold=options.scene_threshold, fps=fps)
    except Exception:  # noqa: BLE001 - a missed reset is a worse pan, not a failure
        cuts = ()

    centres = smooth_centres(
        samples,
        frame_width=src_width,
        scene_cuts=cuts,
        alpha=options.alpha,
        deadzone_px=options.deadzone_px,
    )
    keys = build_keys(
        centres,
        crop_width=plan.crop_width,
        src_width=src_width,
        fps=fps,
        duration_s=duration_s,
        scene_cuts=cuts,
    )
    return CropPath(
        keys=keys,
        samples=len(samples),
        detections=sum(1 for s in samples if s.faces),
        scene_cuts=cuts,
        interval_s=options.interval_s,
    )


__all__ = [
    "COMMAND_FILENAME",
    "DEFAULT_ALPHA",
    "DEFAULT_DEADZONE_PX",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_MODEL",
    "FILTER_NAME",
    "CropPath",
    "Face",
    "FaceSample",
    "TrackingOptions",
    "TrackingUnavailable",
    "build_keys",
    "crop_x_for",
    "detect_faces",
    "face_weight",
    "options_from_config",
    "plan_from_config",
    "pick_face",
    "plan_path",
    "scene_cuts",
    "smooth_centres",
    "to_sendcmd",
    "tracking_enabled",
]
