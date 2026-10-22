/**
 * Adapter boundary: converts a MediaPipe HandLandmarker result into the signBridge hand frame.
 * Pure functions, no DOM, so they can be unit-tested in Node.
 *
 * Hand slot representation (identical keys whether detected or not, easy to load in Python):
 *   detected:   { detected: true,  handedness: "Left"|"Right"|null, confidence: number|null, landmarks: [[x,y,z] x21] }
 *   undetected: { detected: false, handedness: null,                 confidence: null,        landmarks: null }
 *
 * Coordinates are rounded to 6 decimals and are NOT clamped. Non-finite values become null and
 * are counted in `issues.nonFiniteValues`; a wrong landmark count is counted in
 * `issues.wrongLandmarkCount`. Nothing is silently dropped or zero-filled.
 */

import { HAND_LANDMARK_COUNT } from "./schema.js";

const PRECISION = 1e6;

export function emptyHand() {
  return {
    detected: false,
    handedness: null,
    confidence: null,
    landmarks: null,
  };
}

function makeIssues() {
  return { nonFiniteValues: 0, wrongLandmarkCount: 0 };
}

function num(v, issues) {
  if (typeof v === "number" && Number.isFinite(v)) {
    return Math.round(v * PRECISION) / PRECISION;
  }
  issues.nonFiniteValues++;
  return null;
}

function flipLabel(label) {
  if (label === "Left") return "Right";
  if (label === "Right") return "Left";
  return label;
}

/**
 * Map MediaPipe hands onto anatomical left/right slots.
 * If two detections claim the same side, the higher-confidence one is kept and the
 * other is counted in `unassigned` rather than overwriting or vanishing silently.
 */
export function normalizeHands(
  handResult,
  { swapHandedness = true } = {},
  issues = makeIssues(),
) {
  const slots = { left: emptyHand(), right: emptyHand() };
  let unassigned = 0;

  (handResult?.landmarks ?? []).forEach((points, i) => {
    const cat = handResult.handedness?.[i]?.[0];
    const handedness = cat?.categoryName ?? null;
    const label = swapHandedness ? flipLabel(handedness) : handedness;
    const side = label === "Left" ? "left" : label === "Right" ? "right" : null;
    const confidence =
      cat?.score === undefined || cat?.score === null
        ? null
        : num(cat.score, issues);

    if (side === null) {
      unassigned++;
      return;
    }
    const existing = slots[side];
    if (existing.detected && (existing.confidence ?? 0) >= (confidence ?? 0)) {
      unassigned++;
      return;
    }
    if (existing.detected) unassigned++; // the previous occupant is displaced
    if (points.length !== HAND_LANDMARK_COUNT) issues.wrongLandmarkCount++;
    slots[side] = {
      detected: true,
      handedness,
      confidence,
      landmarks: points.map((p) => [
        num(p?.x, issues),
        num(p?.y, issues),
        num(p?.z, issues),
      ]),
    };
  });

  const out = { ...slots };
  if (unassigned > 0) out.unassigned = unassigned;
  return out;
}

/**
 * @param {object} args
 * @param {number} args.frameMs        capture time of this frame (performance.now() timeline)
 * @param {object} args.handResult     HandLandmarkerResult for this very frame
 * @param {boolean} [args.swapHandedness]
 * @returns {{frameMs:number, hands:object, issues:{nonFiniteValues:number, wrongLandmarkCount:number}}}
 */
export function buildRecordFrame({
  frameMs,
  handResult,
  swapHandedness = true,
}) {
  const issues = makeIssues();
  const hands = normalizeHands(handResult, { swapHandedness }, issues);
  return { frameMs, hands, issues };
}
