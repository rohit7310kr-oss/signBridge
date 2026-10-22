/**
 * Small recording buffer. Holds frames only between start() and stop()/discard();
 * nothing accumulates while idle. Pure JS (no DOM), unit-tested in Node.
 *
 * Lifecycle:  idle --start()--> recording --stop()--> stopped --discard()/start()--> idle/recording
 *             discard() also works mid-recording (cancels it; nothing is kept or exportable).
 *
 * Temporal rules:
 *   - frame.frameIndex = 0,1,2,... contiguous; rejected frames never consume an index.
 *   - frame.t = capture time relative to recording start (ms), finite and strictly increasing.
 *   - Frames that are not strictly newer than the previous one are rejected and counted,
 *     never re-stamped.
 */

import {
  SCHEMA_VERSION,
  buildCoordinateSystem,
  buildTopology,
  buildUnits,
  HANDEDNESS_NOTE,
} from "./schema.js";

export const DEFAULT_MAX_FRAMES = 1800; // ~2 minutes at 15 FPS

function defaultId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return (
    "rec-" +
    Date.now().toString(36) +
    "-" +
    Math.random().toString(36).slice(2, 10)
  );
}

export class Recorder {
  constructor({
    maxFrames = DEFAULT_MAX_FRAMES,
    now = () => performance.now(),
    clock = () => new Date(),
    idGen = defaultId,
  } = {}) {
    this.maxFrames = maxFrames;
    this._now = now;
    this._clock = clock;
    this._idGen = idGen;
    this._reset();
  }

  _reset() {
    this.state = "idle"; // idle | recording | stopped
    this.frames = [];
    this.meta = null;
    this.recordingId = null;
    this.startedAt = null;
    this.endedAt = null;
    this.stopReason = null;
    this.dropped = {
      nonMonotonic: 0,
      beforeStart: 0,
      invalidTimestamp: 0,
      cameraSkipped: 0,
    };
    this.counters = {
      framesWithIssues: 0,
      nonFiniteValues: 0,
      wrongLandmarkCount: 0,
      unassignedHands: 0,
    };
    this.exported = false;
    this._startPerf = 0;
    this._endPerf = 0;
    this._lastT = -Infinity;
  }

  get frameCount() {
    return this.frames.length;
  }

  get elapsedMs() {
    if (this.state === "recording") return this._now() - this._startPerf;
    if (this.state === "stopped") return this._endPerf - this._startPerf;
    return 0;
  }

  get lastFrame() {
    return this.frames.length ? this.frames[this.frames.length - 1] : null;
  }

  get hasUnexportedData() {
    return this.frames.length > 0 && !this.exported;
  }

  get canExport() {
    return this.state === "stopped" && this.frames.length > 0;
  }

  /** Begin a new recording. Replaces any previous one. No-op (returns false) if already recording. */
  start(meta = {}) {
    if (this.state === "recording") return false;
    const { maxFrames } = this;
    this._reset();
    this.maxFrames = maxFrames;
    this.meta = meta;
    this.recordingId = this._idGen();
    this.startedAt = this._clock().toISOString();
    this._startPerf = this._now();
    this.state = "recording";
    return true;
  }

  /** Stop recording. Idempotent: a second call keeps the first stop time and reason. */
  stop(reason = "user") {
    if (this.state !== "recording") return false;
    this._endPerf = this._now();
    this.endedAt = this._clock().toISOString();
    this.stopReason = reason;
    this.state = "stopped";
    return true;
  }

  /** Drop all buffered data and return to idle. Safe in any state, including mid-recording. */
  discard() {
    const { maxFrames } = this;
    this._reset();
    this.maxFrames = maxFrames;
  }

  markExported() {
    if (this.canExport) this.exported = true;
  }

  /** Count camera frames the detection loop could not keep up with (only while recording). */
  addCameraSkipped(n) {
    if (this.state === "recording" && Number.isFinite(n) && n > 0)
      this.dropped.cameraSkipped += n;
  }

  /**
   * @param {object} rf record frame from buildRecordFrame()
   * @returns {{ok:boolean, reason?:string, index?:number}}
   */
  addFrame(rf) {
    if (this.state !== "recording")
      return { ok: false, reason: "not_recording" };

    if (this.frames.length >= this.maxFrames) {
      this.stop("limit");
      return { ok: false, reason: "limit" };
    }

    if (!Number.isFinite(rf?.frameMs)) {
      this.dropped.invalidTimestamp++;
      return { ok: false, reason: "invalid_timestamp" };
    }
    const t = Math.round((rf.frameMs - this._startPerf) * 1000) / 1000;
    if (t < 0) {
      this.dropped.beforeStart++;
      return { ok: false, reason: "before_start" };
    }
    if (t <= this._lastT) {
      this.dropped.nonMonotonic++;
      return { ok: false, reason: "non_monotonic" };
    }

    const frameIndex = this.frames.length;
    const frame = { frameIndex, t, hands: rf.hands };

    const issues = rf.issues;
    if (
      issues &&
      (issues.nonFiniteValues > 0 || issues.wrongLandmarkCount > 0)
    ) {
      frame.issues = {
        nonFiniteValues: issues.nonFiniteValues,
        wrongLandmarkCount: issues.wrongLandmarkCount,
      };
      this.counters.framesWithIssues++;
      this.counters.nonFiniteValues += issues.nonFiniteValues;
      this.counters.wrongLandmarkCount += issues.wrongLandmarkCount;
    }
    if (rf.hands?.unassigned > 0)
      this.counters.unassignedHands += rf.hands.unassigned;

    this.frames.push(frame);
    this._lastT = t;
    return { ok: true, index: frameIndex };
  }

  /** Build the exportable object. Frames are shared by reference (no deep copy). */
  toRecording() {
    if (this.state === "idle") return null;
    const durationMs = Math.round(this.elapsedMs * 1000) / 1000;
    return {
      schemaVersion: SCHEMA_VERSION,
      recording: {
        recordingId: this.recordingId,
        startedAt: this.startedAt,
        endedAt: this.endedAt,
        durationMs,
        frameCount: this.frames.length,
        stopReason: this.stopReason,
        limits: { maxFrames: this.maxFrames },
        droppedFrames: { ...this.dropped },
        counters: { ...this.counters },
        units: buildUnits(),
        coordinateSystem: buildCoordinateSystem(),
        handednessNote: HANDEDNESS_NOTE,
        topology: buildTopology(),
        source: this.meta,
      },
      frames: this.frames,
    };
  }
}

/**
 * Serialize without freezing the UI: stringifies in slices and awaits `yieldFn`
 * between them. Returns string parts suitable for `new Blob(parts)`.
 * Throws if the number of serialized frames differs from recording.frameCount.
 */
export async function serializeRecordingParts(
  doc,
  {
    sliceSize = 100,
    yieldFn = () => new Promise((r) => setTimeout(r, 0)),
  } = {},
) {
  const { frames } = doc;
  if (doc.recording.frameCount !== frames.length) {
    throw new Error(
      `frameCount ${doc.recording.frameCount} does not match ${frames.length} frames`,
    );
  }
  const parts = [
    `{"schemaVersion":${JSON.stringify(doc.schemaVersion)},"recording":${JSON.stringify(doc.recording)},"frames":[`,
  ];
  let serialized = 0;
  for (let i = 0; i < frames.length; i += sliceSize) {
    const slice = frames.slice(i, i + sliceSize).map((f) => JSON.stringify(f));
    serialized += slice.length;
    parts.push((i > 0 ? "," : "") + slice.join(","));
    if (i + sliceSize < frames.length) await yieldFn();
  }
  parts.push("]}");
  if (serialized !== frames.length)
    throw new Error("serialized frame count mismatch");
  return parts;
}

export function exportFilename(doc) {
  const id = String(doc.recording.recordingId).slice(0, 8);
  const stamp = String(doc.recording.startedAt).replace(/[:.]/g, "-");
  return `signbridge-${stamp}-${id}.json`;
}
