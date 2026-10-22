/**
 * Capture session for ASL fingerspelling frames. Pure JS (no DOM), unit-tested in Node.
 *
 *   idle --begin()--> capturing --finish()--> submitting --endSubmit()--> idle
 *                         |  \--finish() fails (too few frames / no hands)--> idle
 *                         \--cancel()--> idle
 *
 * Sampling: the caller offers every camera frame it processes; the session keeps one per
 * 1/fps seconds, judged by the frame timestamps, so a slow machine simply yields fewer
 * frames instead of the session assuming a rate it cannot sustain.
 *
 * Limits: capture ends when either `maxFrames` (never above the API's max_frames) or
 * `maxSeconds` is reached. offer() then returns "limit" and the caller should call finish().
 */

export const DEFAULT_FPS = 15;
export const DEFAULT_MAX_SECONDS = 15;

export function resolveLimits({
  fps = DEFAULT_FPS,
  maxSeconds = DEFAULT_MAX_SECONDS,
  minFrames,
  apiMaxFrames,
}) {
  const maxFrames = Math.min(apiMaxFrames, Math.floor(fps * maxSeconds));
  if (maxFrames < minFrames) {
    throw new Error(
      `capture limits allow ${maxFrames} frames but the API needs at least ${minFrames}`,
    );
  }
  return { fps, maxSeconds, minFrames, maxFrames };
}

export class CaptureSession {
  /**
   * @param {object} cfg
   * @param {number} cfg.fps target sampling rate
   * @param {number} cfg.maxSeconds capture duration limit
   * @param {number} cfg.minFrames API min_frames
   * @param {number} cfg.apiMaxFrames API max_frames
   * @param {number} [cfg.rowLength] if set, rows of any other length are rejected
   */
  constructor(cfg) {
    this.limits = resolveLimits(cfg);
    this.rowLength = cfg.rowLength ?? null;
    this._reset();
  }

  _reset() {
    this.state = "idle";
    this.frames = [];
    this.handFrames = 0;
    this.rejected = 0;
    this._startTs = null;
    this._lastTs = null;
  }

  get frameCount() {
    return this.frames.length;
  }

  elapsedMs(nowTs) {
    return this._startTs === null || !Number.isFinite(nowTs)
      ? 0
      : Math.max(0, nowTs - this._startTs);
  }

  /** Start a new capture. Returns false if one is already running or being submitted. */
  begin() {
    if (this.state !== "idle") return false;
    this._reset();
    this.state = "capturing";
    return true;
  }

  /**
   * @param {Array<number|null>} row one feature row
   * @param {boolean} hasHand whether this frame contained a detected hand
   * @param {number} ts frame timestamp in ms (any monotonic timeline)
   * @returns {"added"|"skipped"|"rejected"|"limit"|"ignored"}
   *   "limit": the frame WAS added and the session is now full.
   */
  offer(row, hasHand, ts) {
    if (this.state !== "capturing") return "ignored";
    if (!Number.isFinite(ts) || (this._lastTs !== null && ts <= this._lastTs)) {
      this.rejected++;
      return "rejected";
    }
    if (
      !Array.isArray(row) ||
      (this.rowLength !== null && row.length !== this.rowLength)
    ) {
      this.rejected++;
      return "rejected";
    }
    const interval = 1000 / this.limits.fps;
    // 1 ms of slack so a 30 FPS camera still yields 15 FPS despite timestamp jitter.
    if (this._lastTs !== null && ts - this._lastTs < interval - 1)
      return "skipped";

    if (this._startTs === null) this._startTs = ts;
    this.frames.push(row);
    if (hasHand) this.handFrames++;
    this._lastTs = ts;

    const timeUp = ts - this._startTs >= this.limits.maxSeconds * 1000;
    return this.frames.length >= this.limits.maxFrames || timeUp
      ? "limit"
      : "added";
  }

  /**
   * End capture and hand back the frames to submit.
   * Refuses (and returns to idle) when the sequence is too short, or when hands were seen in
   * fewer than `minFrames` frames: an all-null sequence is not a signing attempt.
   * @returns {{ok:true, frames:Array[], stats:object}|{ok:false, reason:string, message:string}}
   */
  finish() {
    if (this.state !== "capturing") {
      return {
        ok: false,
        reason: "not_capturing",
        message: "No capture is running.",
      };
    }
    const { minFrames } = this.limits;
    const n = this.frames.length;
    if (n < minFrames) {
      this._reset();
      return {
        ok: false,
        reason: "insufficient_frames",
        message: `Only ${n} frame(s) captured; at least ${minFrames} are needed. Hold the capture a little longer.`,
      };
    }
    if (this.handFrames < minFrames) {
      const seen = this.handFrames;
      this._reset();
      return {
        ok: false,
        reason: "no_hands",
        message: `Hands were detected in only ${seen} of ${n} frames (need at least ${minFrames}). Keep your hand fully in view and try again.`,
      };
    }
    this.state = "submitting";
    return {
      ok: true,
      frames: this.frames,
      stats: {
        frames: n,
        handFrames: this.handFrames,
        handFraction: this.handFrames / n,
        rejected: this.rejected,
      },
    };
  }

  /** Abort at any point; buffered frames are dropped. */
  cancel() {
    this._reset();
  }

  /** Call when the submission settles (success or failure). */
  endSubmit() {
    if (this.state === "submitting") this._reset();
  }
}
