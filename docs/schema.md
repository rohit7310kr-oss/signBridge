# signBridge recording schema (v1)

One recording = one JSON file. Produced by `static/js/recorder.js`, checked by `static/js/validate.js`,
expanded for models by `static/js/sequence.js` (`resolveFrames`).

```jsonc
{
  "schemaVersion": 1,
  "recording": {
    "recordingId": "uuid",
    "startedAt": "ISO-8601",          // wall clock at start()
    "endedAt": "ISO-8601",            // wall clock at stop()
    "durationMs": 2500.1,             // start() -> stop(), >= last frame t
    "frameCount": 30,
    "stopReason": "user | limit | camera_stopped",
    "limits": { "maxFrames": 1800 },  // ?maxFrames=N; hitting it stops the recording explicitly
    "droppedFrames": { "nonMonotonic": 0, "beforeStart": 0 },
    "coordinateSystem": { ... },      // see below
    "handednessNote": "...",
    "topology": { "hand": {...}, "pose": {...}, "face": {...} },  // names + entry layout, written once
    "source": { "mediapipeVersion", "delegate", "camera", "swapHandedness", "detectors": { ... } }
  },
  "frames": [ /* strictly increasing t */ ]
}
```

## Frame

```jsonc
{
  "t": 123.4,                       // ms since start(); capture time of the video frame
  "hands": {
    "left":  { "detected": false },
    "right": {
      "detected": true,
      "rawLabel": "Left",           // MediaPipe's own label (before anatomical mapping)
      "handednessScore": 0.97,      // null if the API gave none
      "landmarks": [[x, y, z], ...] // 21, order = topology.hand.names
    },
    "unassigned": 0,                // only when >0 detections could not get a slot
    "swapped": true
  },
  "pose": { "detected": true, "ageMs": 0, "landmarks": [[x, y, z, visibility], ...] }, // 33
  "face": { "detected": true, "ageMs": 66.7, "sourceIndex": 12 },                     // 478 via reference
  "invalidValues": 0                // only when >0; those values were written as null
}
```

### Three kinds of "no data"

| Representation | Meaning |
|---|---|
| `pose` / `face` is `null` | Detector unavailable (failed to load) or has not produced a result yet |
| `{ "detected": false }` (hand slot) / `{ "detected": false, "ageMs": n }` | Detector ran, nothing found |
| `null` inside a landmark array | Invalid value (NaN/Infinity from the model). Always also counted in `invalidValues`; the validator reports it as an error |

Missing hands are never zero-filled.

### Pose / face freshness (async inference)

Pose runs every `poseEvery` frames (default 3) and face every `faceEvery` (default 3, offset by one frame
from pose). Between runs the last result is reused.

- `ageMs` = capture time of this frame − capture time of the frame the result was computed from.
  `0` means computed on this exact frame; `>0` means reused.
- The landmarks are stored once, on the first frame the result appears in. Later frames that reuse it carry
  `{ detected, ageMs, sourceIndex }` pointing at that frame. `resolveFrames()` expands these.
- A result computed before recording started is stored in full on its first frame with `ageMs > 0`.
- `t` is never adjusted to hide staleness. The validator warns when `ageMs > 500` (`stale_pose` / `stale_face`).

## Coordinates

- `x`, `y`: normalized 0..1 in the **raw, un-mirrored** camera frame (0,0 = top-left). The on-screen preview
  is mirrored with CSS only.
- `z`: MediaPipe's relative depth (smaller = closer), not metric. Rounded to 6 decimals along with x/y.
- Pose `visibility` is MediaPipe's per-landmark in-view score (0..1), `null` if the API omits it.
  The JS API does not expose per-landmark presence.
- Pose names `left_*`/`right_*` follow MediaPipe's convention (the subject's sides).

## Handedness

MediaPipe assumes a mirrored (selfie) image. signBridge feeds the raw frame, so by default the label is
swapped: `hands.right` is the signer's anatomical right hand. `rawLabel` always keeps the original.
**Verify once on your camera**: raise your right hand and check "Hand slots" in the debug panel; if it says
`left`, load the page with `?swapHands=0`.

## What is and isn't captured

| Modality | Captured | Confidence available from API |
|---|---|---|
| Hands | 21 × (x,y,z), up to 2 hands | handedness score only |
| Pose | all 33 × (x,y,z,visibility) | visibility per landmark |
| Face | all 478 × (x,y,z) incl. iris, 1 face | none (no face score in the JS API) |

Not captured by design: blendshapes / expression data, world (metric) landmarks, video frames.

## Validation (`validateRecording`)

Errors (recording is corrupt): schema version, frame count / duration mismatch, bad or non-increasing `t`,
wrong hand landmark count, non-finite/missing coordinates, confidence or visibility outside [0,1],
bad `ageMs`, bad `sourceIndex`, nulled values.
Warnings: stale pose/face, pose/face landmark count ≠ 33/478, hand slot conflicts, a modality never detected,
empty recording. Results are aggregated per code with up to three examples each.

## URL options

`?delegate=CPU` · `?poseEvery=N` · `?faceEvery=N` · `?maxFrames=N` · `?swapHands=0`

## Tests

`node --test tests/phase1.test.mjs` (no dependencies; the browser/MediaPipe layer is not covered by them).
