/**
 * signBridge recording schema: constants shared by the recorder, validator and UI.
 * Human-readable description lives in docs/schema.md; keep the two in sync.
 *
 * v2 = hand-only. v1 (hands + pose + face) is not accepted; see migrate.js.
 */

export const SCHEMA_VERSION = 2;

export const HAND_LANDMARK_COUNT = 21;

// MediaPipe hand topology (index = position in `landmarks`).
export const HAND_NAMES = [
  "WRIST",
  "THUMB_CMC",
  "THUMB_MCP",
  "THUMB_IP",
  "THUMB_TIP",
  "INDEX_FINGER_MCP",
  "INDEX_FINGER_PIP",
  "INDEX_FINGER_DIP",
  "INDEX_FINGER_TIP",
  "MIDDLE_FINGER_MCP",
  "MIDDLE_FINGER_PIP",
  "MIDDLE_FINGER_DIP",
  "MIDDLE_FINGER_TIP",
  "RING_FINGER_MCP",
  "RING_FINGER_PIP",
  "RING_FINGER_DIP",
  "RING_FINGER_TIP",
  "PINKY_MCP",
  "PINKY_PIP",
  "PINKY_DIP",
  "PINKY_TIP",
];

/** Written once into every recording so frames can stay compact arrays. */
export function buildTopology() {
  return {
    hand: {
      count: HAND_LANDMARK_COUNT,
      entry: ["x", "y", "z"],
      names: HAND_NAMES,
    },
  };
}

export function buildUnits() {
  return {
    timestamps: "milliseconds since the start of the recording (frame.t)",
    durationMs: "milliseconds from start() to stop()",
  };
}

export function buildCoordinateSystem() {
  return {
    imageX:
      "normalized horizontal image coordinate, 0 = left edge, 1 = right edge of the RAW camera frame",
    imageY:
      "normalized vertical image coordinate, 0 = top edge, 1 = bottom edge of the RAW camera frame",
    z: "MediaPipe relative depth, wrist is the origin; smaller = closer to the camera. Not metric.",
    range:
      "Values are NOT clamped. x/y can fall slightly outside [0, 1] when a hand is partly out of frame.",
    mirrored: false,
    mirroredNote:
      "Landmarks are in the un-mirrored camera frame. The on-screen preview is mirrored with CSS only.",
    coordinatePrecision: "rounded to 6 decimal places",
  };
}

export const HANDEDNESS_NOTE =
  "hands.left / hands.right are the signer's anatomical hands. MediaPipe labels handedness assuming a " +
  "mirrored (selfie) image; signBridge feeds the raw frame, so the slot is the OPPOSITE of MediaPipe's " +
  "label when source.swapHandedness is true. Each detected hand's `handedness` field keeps MediaPipe's " +
  "original label; `confidence` is MediaPipe's score for that label.";
