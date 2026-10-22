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
 */

export const COLUMN_RE = /^([xyz])_(face|left_hand|right_hand|pose)_(\d+)$/;

/** Column group name -> HolisticLandmarkerResult property. */
export const GROUP_KEYS = {
  face: "faceLandmarks",
  left_hand: "leftHandLandmarks",
  right_hand: "rightHandLandmarks",
  pose: "poseLandmarks",
};

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

/** One API row from one Holistic result: exactly `extractors.length` numbers/nulls, in order. */
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
