/**
 * Convert a schema v1 recording (hands + pose + face) to schema v2 (hands only).
 * Pose and face are dropped; nothing else is invented.
 *
 *   v1 hand { detected, rawLabel, handednessScore, landmarks }
 *   v2 hand { detected, handedness, confidence, landmarks }   (undetected -> all null)
 *
 * v1 had no frameIndex, no `counters`, no `droppedFrames.cameraSkipped`; these are
 * derived from the frames or set to 0. The result is marked with `migratedFrom: 1`.
 */

import {
  SCHEMA_VERSION,
  buildCoordinateSystem,
  buildTopology,
  buildUnits,
  HANDEDNESS_NOTE,
} from "./schema.js";
import { emptyHand } from "./landmarks.js";

function migrateHand(h) {
  if (!h?.detected) return emptyHand();
  return {
    detected: true,
    handedness: h.rawLabel ?? null,
    confidence: h.handednessScore ?? null,
    landmarks: h.landmarks,
  };
}

export function migrateV1ToV2(v1) {
  if (!v1 || v1.schemaVersion !== 1)
    throw new Error("not a schema v1 recording");

  const counters = {
    framesWithIssues: 0,
    nonFiniteValues: 0,
    wrongLandmarkCount: 0,
    unassignedHands: 0,
  };
  const frames = v1.frames.map((f, frameIndex) => {
    const out = {
      frameIndex,
      t: f.t,
      hands: {
        left: migrateHand(f.hands?.left),
        right: migrateHand(f.hands?.right),
      },
    };
    if (f.hands?.unassigned > 0) {
      out.hands.unassigned = f.hands.unassigned;
      counters.unassignedHands += f.hands.unassigned;
    }
    if (f.invalidValues > 0) {
      out.issues = { nonFiniteValues: f.invalidValues, wrongLandmarkCount: 0 };
      counters.framesWithIssues++;
      counters.nonFiniteValues += f.invalidValues;
    }
    return out;
  });

  const {
    topology,
    source,
    droppedFrames,
    coordinateSystem,
    handednessNote,
    ...rest
  } = v1.recording;
  return {
    schemaVersion: SCHEMA_VERSION,
    recording: {
      ...rest,
      frameCount: frames.length,
      droppedFrames: {
        nonMonotonic: droppedFrames?.nonMonotonic ?? 0,
        beforeStart: droppedFrames?.beforeStart ?? 0,
        invalidTimestamp: 0,
        cameraSkipped: 0,
      },
      counters,
      units: buildUnits(),
      coordinateSystem: buildCoordinateSystem(), // same convention as v1, restated for v2
      handednessNote: HANDEDNESS_NOTE,
      topology: buildTopology(),
      source: {
        ...(source ?? {}),
        detectors: { hands: source?.detectors?.hands ?? null },
      },
      migratedFrom: 1,
    },
    frames,
  };
}
