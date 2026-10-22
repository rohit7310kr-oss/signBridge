/**
 * Verifies the fix for "MediaPipe works worse while observing/sending to ASL than while
 * exporting JSON": the capture path must carry the SAME hand landmarks as the export path,
 * Holistic's (pose-derived) hand landmarks must only ever be the fallback, and the 390-value
 * row must stay schema-shaped.
 *
 * Pure Node, no DOM, no MediaPipe download. Run: node tests/asl-merge.test.mjs
 */

import fs from "node:fs";
import { fileURLToPath } from "node:url";

import { normalizeHands } from "../landmarks.js";
import {
  buildExtractors,
  extractRow,
  hasHands,
  mergeHandResults,
  needsHolistic,
  splitHands,
  validateSchema,
} from "../asl-features.js";

const R = (v) => Math.round(v * 1e6) / 1e6;
const AXIS = { x: 0, y: 1, z: 2 };
const failures = [];
const check = (name, cond, detail = "") => {
  if (!cond) failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
  console.log(`${cond ? "ok  " : "FAIL"}  ${name}${detail && !cond ? ` — ${detail}` : ""}`);
};

// ---- fixtures ----
const pt = (x, y, z) => ({ x, y, z });
const leftPts = Array.from({ length: 21 }, (_, i) =>
  pt(0.11 + i / 1000, 0.22 + i / 1000, -0.33 - i / 1000),
);
const rightPts = Array.from({ length: 21 }, (_, i) =>
  pt(0.44 + i / 1000, 0.55 + i / 1000, 0.66 + i / 1000),
);
const facePts = Array.from({ length: 478 }, (_, i) =>
  pt(i / 1000, 0.5 + i / 10000, -i / 5000),
);
const posePts = Array.from({ length: 33 }, (_, i) => pt(0.7 + i / 10000, 0.8, 0.01));
const HOLISTIC_HAND = 0.9; // deliberately wrong: must never reach the payload

const handResult = {
  landmarks: [leftPts, rightPts],
  worldLandmarks: [],
  handedness: [
    [{ categoryName: "Left", score: 0.91 }],
    [{ categoryName: "Right", score: 0.88 }],
  ],
};
const holisticResult = {
  faceLandmarks: [facePts],
  faceBlendshapes: [],
  poseLandmarks: [posePts],
  poseWorldLandmarks: [],
  poseSegmentationMasks: [],
  leftHandLandmarks: [Array.from({ length: 21 }, () => pt(HOLISTIC_HAND, HOLISTIC_HAND, HOLISTIC_HAND))],
  rightHandLandmarks: [Array.from({ length: 21 }, () => pt(HOLISTIC_HAND, HOLISTIC_HAND, HOLISTIC_HAND))],
};

// ---- real live schema ----
const livePath = fileURLToPath(new URL("../live_schema.json", import.meta.url));
const schema = validateSchema(JSON.parse(fs.readFileSync(livePath, "utf8")));
const extractors = buildExtractors(schema.columns);

check(
  "schema: 390 columns, min 8 / max 1500 frames",
  schema.columns.length === 390 && schema.minFrames === 8 && schema.maxFrames === 1500,
  `got ${schema.columns.length}/${schema.minFrames}/${schema.maxFrames}`,
);
check(
  "schema: face+pose present, so Holistic is still required",
  needsHolistic(schema.columns) === true,
);
check(
  "schema: hands-only columns need no Holistic",
  needsHolistic(["x_left_hand_0", "z_right_hand_20"]) === false,
);

// ---- capture path: merged row ----
const merged = mergeHandResults(holisticResult, handResult, { swapHandedness: true });
const row = extractRow(merged, extractors);
check("row width equals column count", row.length === schema.columns.length);
check("row has no undefined entries", row.every((v) => v !== undefined));
check("hands reported in the frame", hasHands(merged) === true);

// Same handedness convention as the recorder, so left/right mean the same thing in both modes.
const exported = normalizeHands(handResult, { swapHandedness: true });

let handMismatch = null;
for (let i = 0; i < schema.columns.length; i++) {
  const name = schema.columns[i];
  const m = /^([xyz])_(left_hand|right_hand)_(\d+)$/.exec(name);
  if (!m) continue;
  const [, axis, side, idx] = m;
  const want = R(exported[side === "left_hand" ? "left" : "right"].landmarks[Number(idx)][AXIS[axis]]);
  if (row[i] !== want)
    handMismatch ??= `${name}: capture ${row[i]} vs export ${want}`;
}
check(
  "every hand value in the ASL row equals the exported recording's value",
  handMismatch === null,
  handMismatch ?? "",
);
check(
  "no Holistic hand landmark leaked into the payload",
  !row.some((v) => v === HOLISTIC_HAND),
);

// ---- Holistic still supplies face/pose, value for value ----
// Expectations are derived from the column NAME, so this stays correct whatever subset of
// landmark indices the published schema happens to use.
let facePoseMismatch = null;
let faceSeen = 0;
let poseSeen = 0;
for (let i = 0; i < schema.columns.length; i++) {
  const m = /^([xyz])_(face|pose)_(\d+)$/.exec(schema.columns[i]);
  if (!m) continue;
  const [, axis, group, idx] = m;
  const src = group === "face" ? facePts : posePts;
  group === "face" ? faceSeen++ : poseSeen++;
  const want = R(src[Number(idx)][AXIS[axis]]);
  if (row[i] !== want) facePoseMismatch ??= `${schema.columns[i]}: ${row[i]} vs ${want}`;
}
check(
  "every face/pose value comes from Holistic, unchanged",
  facePoseMismatch === null && faceSeen === 228 && poseSeen === 36,
  facePoseMismatch ?? `face ${faceSeen} pose ${poseSeen}`,
);
check(
  "all face/pose columns are finite",
  schema.columns.every((c, i) => !/_(face|pose)_/.test(c) || typeof row[i] === "number"),
);

// ---- fallback: a side the hand model missed is filled by Holistic, never left null ----
// handResult's first detection carries the raw label "Left", which swapHandedness flips to the
// anatomical RIGHT slot — hence "right_hand" here (see the mapping check above).
const handModel = { ...handResult, landmarks: [leftPts], handedness: [handResult.handedness[0]] };
const fallback = mergeHandResults(holisticResult, handModel, { swapHandedness: true });
const fallbackRow = extractRow(fallback, extractors);
const detectedIdx = schema.columns.indexOf("x_right_hand_0");
const missedIdx = schema.columns.indexOf("x_left_hand_0");
check(
  "the side the hand model saw still comes from the hand model",
  detectedIdx >= 0 && fallbackRow[detectedIdx] === R(leftPts[0].x),
  `${fallbackRow[detectedIdx]} vs ${R(leftPts[0].x)}`,
);
check(
  "the side the hand model missed falls back to Holistic",
  missedIdx >= 0 && fallbackRow[missedIdx] === HOLISTIC_HAND,
  `got ${fallbackRow[missedIdx]}`,
);

// ---- mirroring: hands mirrored like the face/pose image ----
const mirrored = mergeHandResults(null, { ...handResult, landmarks: [rightPts], handedness: [[{ categoryName: "Right", score: 0.9 }]] }, {
  swapHandedness: false,
  mirrorHands: true,
});
const mirrorRow = extractRow(mirrored, extractors);
const mirrorLeft = schema.columns.indexOf("x_left_hand_0");
const mirrorRight = schema.columns.indexOf("x_right_hand_0");
check(
  "?aslMirror=1 mirrors the hand columns",
  mirrorRow[mirrorLeft] === R(1 - rightPts[0].x) || mirrorRow[mirrorRight] === R(1 - rightPts[0].x),
  `left ${mirrorRow[mirrorLeft]} right ${mirrorRow[mirrorRight]} want ${R(1 - rightPts[0].x)}`,
);

// ---- hand-only fallback path (no HandLandmarker at all) ----
const holisticOnly = mergeHandResults(holisticResult, null, { swapHandedness: true });
check(
  "with no hand model at all, Holistic's hands are still used",
  hasHands(holisticOnly) === true &&
    extractRow(holisticOnly, extractors)[schema.columns.indexOf("x_left_hand_0")] ===
      HOLISTIC_HAND,
);
check("splitHands keeps one detection per side", (() => {
  const s = splitHands(handResult, { swapHandedness: true });
  return s.left === rightPts && s.right === leftPts; // labels are swapped: raw frame
})());

console.log(
  failures.length === 0
    ? "\nAll checks passed."
    : `\n${failures.length} check(s) failed:\n- ${failures.join("\n- ")}`,
);
process.exit(failures.length === 0 ? 0 : 1);
