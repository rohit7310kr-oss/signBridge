/**
 * Validation of an exported (schema v2, hand-only) recording. Never mutates the input.
 * Issues are aggregated per code (count + first few examples) so a bad recording
 * cannot produce thousands of lines.
 *
 * Note: x/y are deliberately NOT range-checked. MediaPipe can legitimately return
 * values slightly outside [0, 1] when a hand is partly out of frame.
 */

import { SCHEMA_VERSION, HAND_LANDMARK_COUNT } from "./schema.js";

export const MAX_GAP_MS = 500; // warn when consecutive frames are further apart than this

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const isCount = (v) => Number.isInteger(v) && v >= 0;

export function validateRecording(doc, { maxGapMs = MAX_GAP_MS } = {}) {
  const issues = new Map(); // code -> { severity, count, examples }
  const add = (severity, code, message, frameIndex) => {
    let e = issues.get(code);
    if (!e) issues.set(code, (e = { severity, code, count: 0, examples: [] }));
    e.count++;
    if (e.examples.length < 3) {
      e.examples.push(
        frameIndex === undefined ? message : `frame ${frameIndex}: ${message}`,
      );
    }
  };
  const error = (...a) => add("error", ...a);
  const warn = (...a) => add("warning", ...a);

  const stats = {
    frames: 0,
    leftDetected: 0,
    rightDetected: 0,
    bothDetected: 0,
    noHands: 0,
    framesWithIssues: 0,
    largeGaps: 0,
  };

  const finish = () => {
    const all = [...issues.values()];
    return {
      valid: !all.some((i) => i.severity === "error"),
      errors: all.filter((i) => i.severity === "error"),
      warnings: all.filter((i) => i.severity === "warning"),
      stats,
    };
  };

  if (!doc || typeof doc !== "object") {
    error("not_an_object", "recording is not an object");
    return finish();
  }
  if (doc.schemaVersion !== SCHEMA_VERSION) {
    error(
      "bad_schema_version",
      doc.schemaVersion === 1
        ? "schema v1 (hands + pose + face); convert it with migrateV1ToV2() in js/migrate.js"
        : `expected ${SCHEMA_VERSION}, got ${doc.schemaVersion}`,
    );
    return finish();
  }
  const meta = doc.recording;
  if (!meta || typeof meta !== "object") {
    error("missing_recording", "missing `recording` metadata");
    return finish();
  }
  if (!Array.isArray(doc.frames)) {
    error("missing_frames", "`frames` is not an array");
    return finish();
  }
  const frames = doc.frames;
  stats.frames = frames.length;

  if (!meta.recordingId)
    error("missing_recording_id", "recordingId is missing");
  if (Number.isNaN(Date.parse(meta.startedAt)))
    error("bad_started_at", `startedAt: ${meta.startedAt}`);
  if (Number.isNaN(Date.parse(meta.endedAt)))
    error("bad_ended_at", `endedAt: ${meta.endedAt}`);
  if (!meta.stopReason) error("missing_stop_reason", "stopReason is missing");
  if (meta.frameCount !== frames.length) {
    error(
      "frame_count_mismatch",
      `frameCount ${meta.frameCount} vs ${frames.length} frames`,
    );
  }
  if (!isNum(meta.durationMs) || meta.durationMs < 0) {
    error("bad_duration", `durationMs: ${meta.durationMs}`);
  }
  for (const group of ["droppedFrames", "counters"]) {
    const g = meta[group];
    if (!g || typeof g !== "object" || !Object.values(g).every(isCount)) {
      error(
        "bad_counters",
        `recording.${group} missing or contains a non-count value`,
      );
    }
  }
  if (frames.length === 0) warn("empty_recording", "recording has no frames");

  let prevT = -Infinity;
  frames.forEach((f, i) => {
    if (!f || typeof f !== "object") {
      error("bad_frame", "frame is not an object", i);
      return;
    }

    // ---- index + timestamp ----
    if (f.frameIndex !== i)
      error("bad_frame_index", `frameIndex ${f.frameIndex}, expected ${i}`, i);
    if (!isNum(f.t) || f.t < 0) {
      error("bad_timestamp", `t = ${f.t}`, i);
    } else {
      if (f.t <= prevT)
        error("non_monotonic_timestamp", `t ${f.t} <= previous ${prevT}`, i);
      else if (prevT !== -Infinity && f.t - prevT > maxGapMs) {
        stats.largeGaps++;
        warn(
          "large_time_gap",
          `${Math.round(f.t - prevT)} ms since previous frame`,
          i,
        );
      }
      prevT = f.t;
    }

    // ---- flagged capture issues ----
    if (f.issues) {
      stats.framesWithIssues++;
      if (f.issues.nonFiniteValues > 0) {
        error(
          "non_finite_at_capture",
          `${f.issues.nonFiniteValues} non-finite value(s) were written as null`,
          i,
        );
      }
      if (f.issues.wrongLandmarkCount > 0) {
        error(
          "wrong_landmark_count_at_capture",
          `${f.issues.wrongLandmarkCount} hand(s) had a wrong landmark count`,
          i,
        );
      }
    }

    // ---- hands ----
    let detectedHere = 0;
    for (const side of ["left", "right"]) {
      const h = f.hands?.[side];
      if (!h || typeof h.detected !== "boolean") {
        error("bad_hand_slot", `hands.${side} missing or malformed`, i);
        continue;
      }
      if (!h.detected) {
        if (
          h.landmarks !== null ||
          h.handedness !== null ||
          h.confidence !== null
        ) {
          error(
            "inconsistent_undetected_hand",
            `hands.${side} is not detected but carries data`,
            i,
          );
        }
        continue;
      }
      detectedHere++;
      stats[`${side}Detected`]++;

      if (
        h.handedness !== null &&
        h.handedness !== "Left" &&
        h.handedness !== "Right"
      ) {
        error("bad_handedness", `handedness ${h.handedness}`, i);
      }
      if (
        h.confidence !== null &&
        (!isNum(h.confidence) || h.confidence < 0 || h.confidence > 1)
      ) {
        error("bad_confidence", `confidence ${h.confidence}`, i);
      }
      const lms = h.landmarks;
      if (!Array.isArray(lms)) {
        error("bad_landmarks", `hands.${side}.landmarks is not an array`, i);
        continue;
      }
      if (lms.length !== HAND_LANDMARK_COUNT) {
        error(
          "bad_hand_landmark_count",
          `${lms.length} landmarks, expected ${HAND_LANDMARK_COUNT}`,
          i,
        );
      }
      for (const lm of lms) {
        if (!Array.isArray(lm) || lm.length !== 3) {
          error("bad_landmark_shape", "landmark should be [x, y, z]", i);
          break;
        }
        if (!lm.every(isNum)) {
          error(
            "non_finite_coordinate",
            `landmark has invalid value(s) ${JSON.stringify(lm)}`,
            i,
          );
          break;
        }
      }
    }
    if (detectedHere === 0) stats.noHands++;
    if (detectedHere === 2) stats.bothDetected++;
    if (f.hands?.unassigned > 0) {
      warn(
        "hand_slot_conflict",
        `${f.hands.unassigned} hand detection(s) could not be assigned a side`,
        i,
      );
    }
  });

  // ---- recording-level consistency ----
  if (
    frames.length &&
    isNum(prevT) &&
    isNum(meta.durationMs) &&
    meta.durationMs < prevT
  ) {
    error(
      "duration_shorter_than_frames",
      `durationMs ${meta.durationMs} < last frame t ${prevT}`,
    );
  }
  if (
    meta.counters &&
    isCount(meta.counters.framesWithIssues) &&
    meta.counters.framesWithIssues !== stats.framesWithIssues
  ) {
    error(
      "counter_mismatch",
      `counters.framesWithIssues ${meta.counters.framesWithIssues} vs ${stats.framesWithIssues} flagged frames`,
    );
  }
  const d = meta.droppedFrames;
  if (d && Object.values(d).some((n) => n > 0)) {
    warn("frames_dropped", `dropped/skipped frames: ${JSON.stringify(d)}`);
  }
  if (frames.length && stats.leftDetected + stats.rightDetected === 0) {
    warn("hands_never_detected", "no hand was detected in the whole recording");
  }

  return finish();
}

/** One-line human summary for the UI. */
export function summarizeValidation(result) {
  const e = result.errors.reduce((n, i) => n + i.count, 0);
  const w = result.warnings.reduce((n, i) => n + i.count, 0);
  return `${result.valid ? "VALID" : "INVALID"}: ${e} error(s), ${w} warning(s)`;
}
