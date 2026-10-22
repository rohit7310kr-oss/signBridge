/**
 * Schema-driven feature extraction for the three.ws ASL fingerspelling API.
 * Pure functions, no DOM, unit-tested in Node (tests/asl.test.mjs).
 *
 * The API publishes an ordered `columns` list (e.g. "x_face_0", "y_left_hand_20", "z_pose_5").
 * That list is the single source of truth: we never hardcode an order or a count. Each column
 * becomes one extractor, and a row is produced by running the extractors in column order, so a
 * row always has exactly `columns.length` entries.
 *
 * Missing landmarks and non-finite coordinates become `null` (never 0), which is the API's
 * documented way to express "not detected".
 *
 * Detector sourcing (schema v2 app, see asl.js):
 *   hand columns  <- the dedicated HandLandmarker, the same instance the recorder uses
 *   face/pose     <- HolisticLandmarker, and only when the schema actually asks for them
 * `mergeHandResults()` puts both into one HolisticLandmarkerResult-shaped object, so
 * extractRow()/hasHands() and the wire format are untouched.
 */

export const COLUMN_RE = /^([xyz])_(face|left_hand|right_hand|pose)_(\d+)$/;

/** Column group name -> HolisticLandmarkerResult property. */
export const GROUP_KEYS = {
  face: "faceLandmarks",
  left_hand: "leftHandLandmarks",
  right_hand: "rightHandLandmarks",
  pose: "poseLandmarks",
};

/**
 * Column groups that ONLY the Holistic task can produce. The hand-only model has no face and
 * no pose output, so a schema mentioning these forces Holistic to be loaded as well.
 */
export const HOLISTIC_ONLY_GROUPS = ["face", "pose"];

const PRECISION = 1e6; // 6 decimals keeps the payload small without losing meaningful precision

export class SchemaError extends Error {}

/**
 * Check the API's schema response and normalise it.
 * @returns {{columns:string[], minFrames:number, maxFrames:number}}
 */
export function validateSchema(schema) {
  if (!schema || typeof schema !== "object") {
    throw new SchemaError("schema is not an object");
  }
  const { columns, min_frames: minFrames, max_frames: maxFrames } = schema;
  if (!Array.isArray(columns) || columns.length === 0) {
    throw new SchemaError("schema.columns is missing or empty");
  }
  if (!columns.every((c) => typeof c === "string")) {
    throw new SchemaError("schema.columns contains a non-string entry");
  }
  if (!Number.isInteger(minFrames) || minFrames < 1) {
    throw new SchemaError(`schema.min_frames is invalid: ${minFrames}`);
  }
  if (!Number.isInteger(maxFrames) || maxFrames < minFrames) {
    throw new SchemaError(`schema.max_frames is invalid: ${maxFrames}`);
  }
  return { columns, minFrames, maxFrames };
}

/**
 * Parse every column name into an extractor. Throws if any column does not match the
 * expected pattern, because silently skipping one would shift every later feature.
 */
export function buildExtractors(columns) {
  if (!Array.isArray(columns) || columns.length === 0) {
    throw new SchemaError("columns must be a non-empty array");
  }
  return columns.map((name) => {
    const m = COLUMN_RE.exec(name);
    if (!m) throw new SchemaError(`unrecognised feature column: "${name}"`);
    return { name, axis: m[1], key: GROUP_KEYS[m[2]], index: Number(m[3]) };
  });
}

/**
 * True when `columns` names a face or pose feature, i.e. when the hand-only model cannot
 * satisfy the schema on its own and HolisticLandmarker has to be loaded too.
 * A hands-only schema (the usual case for fingerspelling) returns false, and capture then
 * runs the recording path's detector and nothing else.
 */
export function needsHolistic(columns) {
  return (columns ?? []).some((name) => {
    const m = COLUMN_RE.exec(name);
    return m ? HOLISTIC_ONLY_GROUPS.includes(m[2]) : false;
  });
}

/**
 * Holistic results are arrays of landmark lists (one per person) in current MediaPipe builds,
 * but a flat landmark list is tolerated too. Returns a landmark array or null.
 */
export function pickSet(value) {
  if (!Array.isArray(value) || value.length === 0) return null;
  const first = value[0];
  if (Array.isArray(first)) return first.length ? first : null;
  return first && typeof first.x === "number" ? value : null;
}

function num(v) {
  return typeof v === "number" && Number.isFinite(v)
    ? Math.round(v * PRECISION) / PRECISION
    : null;
}

/**
 * Split a HandLandmarkerResult into anatomical left/right landmark lists.
 *
 * MediaPipe's handedness assumes a mirrored (selfie) image, but we feed the raw camera frame,
 * so the labels are flipped — the same convention the recorder applies in landmarks.js, which
 * keeps the left_hand/right_hand columns meaning the same thing in both modes.
 *
 * The first detection of a side wins; a duplicate claim on the same side is dropped rather
 * than overwriting it, so a frame never silently gains a hand it did not have.
 *
 * @returns {{left: Array|null, right: Array|null}} lists of 21 {x,y,z} landmarks
 */
export function splitHands(handResult, { swapHandedness = true } = {}) {
  const out = { left: null, right: null };
  const lists = handResult?.landmarks ?? [];
  const cats = handResult?.handedness ?? [];
  lists.forEach((points, i) => {
    if (!Array.isArray(points) || points.length === 0) return;
    let label = cats?.[i]?.[0]?.categoryName ?? null;
    if (swapHandedness) {
      if (label === "Left") label = "Right";
      else if (label === "Right") label = "Left";
    }
    const side = label === "Left" ? "left" : label === "Right" ? "right" : null;
    if (side && !out[side]) out[side] = points;
  });
  return out;
}

/**
 * Combine the two tasks into ONE HolisticLandmarkerResult-shaped object, so extractRow(),
 * hasHands() and the JSON payload keep working exactly as before.
 *
 *   hands     -> the dedicated HandLandmarker's, whenever it saw that hand. Holistic's hand
 *                landmarks are derived from a low-resolution pose crop and are the reason the
 *                capture path used to look worse than the recording path; they are now only a
 *                fallback for a side the hand model missed in that frame (never a regression
 *                against the old behaviour).
 *   face/pose -> Holistic's, unchanged.
 *
 * Each hand slot is a `NormalizedLandmark[][]` (one landmark list per person), which is the
 * shape Holistic uses and the shape pickSet() expects.
 *
 * @param {object|null} holisticResult HolisticLandmarkerResult, or null when not used
 * @param {object|null} handResult HandLandmarkerResult from the shared detector
 * @param {object} [opts]
 * @param {boolean} [opts.swapHandedness] recorder convention for handedness labels
 * @param {boolean} [opts.mirrorHands] mirror hand x back onto a mirrored row (?aslMirror=1);
 *   also flips the handedness test, because mirroring an image swaps which side a hand is on
 */
export function mergeHandResults(
  holisticResult,
  handResult,
  { swapHandedness = true, mirrorHands = false } = {},
) {
  const holistic = holisticResult ?? null;
  const split = handResult
    ? splitHands(handResult, {
        swapHandedness: mirrorHands ? !swapHandedness : swapHandedness,
      })
    : null;

  const toRowFrame = (points) =>
    mirrorHands
      ? points.map((p) =>
          p && typeof p.x === "number" ? { ...p, x: 1 - p.x } : p,
        )
      : points;

  const slot = (side) => {
    const dedicated = split?.[side] ?? null;
    if (dedicated) return [toRowFrame(dedicated)];
    const fallback = pickSet(holistic?.[GROUP_KEYS[`${side}_hand`]]);
    return fallback ? [fallback] : [];
  };

  return {
    ...(holistic ?? {}),
    faceLandmarks: holistic?.faceLandmarks ?? [],
    poseLandmarks: holistic?.poseLandmarks ?? [],
    leftHandLandmarks: slot("left"),
    rightHandLandmarks: slot("right"),
  };
}

/** One API row from one (merged) result: exactly `extractors.length` numbers/nulls, in order. */
export function extractRow(result, extractors) {
  const sets = {};
  for (const key of Object.values(GROUP_KEYS))
    sets[key] = pickSet(result?.[key]);
  return extractors.map(({ axis, key, index }) =>
    num(sets[key]?.[index]?.[axis]),
  );
}

/** True if at least one hand was detected in this result. */
export function hasHands(result) {
  return (
    pickSet(result?.leftHandLandmarks) !== null ||
    pickSet(result?.rightHandLandmarks) !== null
  );
}
