# Models

Detection weights used by `edit/track.py`. Not code, not config - binary
assets that are fetched rather than written, so they get their own directory
and their own provenance note, the same way `assets/music/README.md` records a
licence for every track.

Nothing here is required to run the pipeline. Without the model the render
falls back to a centre crop and says so on the clip's `reframe` stage.

## blaze_face_full_range.tflite

- **What:** MediaPipe BlazeFace, full-range variant. Face detection only - no
  landmarks, no identity, no recognition.
- **Used by:** `edit/track.py`, sampled every `edit.detect_interval_s`.
- **Licence:** Apache License 2.0, as published by Google in the MediaPipe
  model zoo. Redistribution is permitted; the weights are not modified here.
- **Source:**
  <https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite>
- **Fetch:**

  ```bash
  curl -L -o assets/models/blaze_face_full_range.tflite \
    https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite
  ```

**Full-range, not short-range, and it matters.** The short-range model is
tuned for a face within about two metres of the lens. Measured against a real
podcast two-shot at 2560x1440, twelve samples one second apart:

| model       | confidence | dropouts |
|-------------|------------|----------|
| short-range | 0.41-0.52  | 2 of 12  |
| full-range  | 0.84-0.95  | 0 of 12  |

Short-range is smaller (230 KB against 1.1 MB) and faster, and is the better
pick if your sources are all close-up single-speaker framing. Point
`edit.face_model` at it and fetch it from the same path with
`blaze_face_short_range/float16/1/blaze_face_short_range.tflite`.
