/**
 * ASL fingerspelling controller. Glue between app.js's camera loop, the pure capture/feature
 * modules and the local proxy. It owns no camera and no animation loop: app.js calls
 * processFrame() once per camera frame while `isCapturing` is true.
 *
 * Detectors. Hand landmarks always come from the shared HandLandmarker that app.js injects as
 * `detectHands` — the very same call the landmark recorder uses — so capture mode and record
 * mode see identical hands. HolisticLandmarker is loaded lazily and used ONLY to fill the
 * face/pose columns the schema asks for; its hand landmarks (a low-resolution pose-derived crop)
 * are never sent unless the dedicated hand model missed that hand in that frame.
 * Consequence: with the hands-only schema, capture runs exactly the recording path.
 *
 * States: idle -> preparing -> capturing -> recognizing -> idle
 *   cancel() works in preparing / capturing / recognizing and always returns to idle.
 *   A monotonically increasing token makes late async results from a cancelled run harmless.
 *
 * Note: this recognises ASL FINGERSPELLING (letters spelled out by hand) only.
 */

import {
  buildExtractors,
  extractRow,
  hasHands,
  mergeHandResults,
  needsHolistic,
} from "./asl-features.js";
import { CaptureSession } from "./asl-capture.js";
import {
  AslApiError,
  confidenceLevel,
  fetchSchema,
  recognize,
} from "./asl-client.js";

const fmtSec = (ms) => `${(ms / 1000).toFixed(1)} s`;

export class AslController {
  /**
   * @param {object} deps
   * @param {(id:string)=>HTMLElement} deps.$ element lookup
   * @param {()=>Promise<object>} deps.loadHolistic resolves to an initialised HolisticLandmarker
   * @param {()=>boolean} deps.hasHandSource true once the shared HandLandmarker exists. When it
   *   does, the hand columns come from it and Holistic is only needed for face/pose.
   * @param {boolean} [deps.swapHandedness] handedness convention (see landmarks.js)
   * @param {()=>boolean} deps.isCameraRunning
   * @param {()=>string|null} deps.blockReason non-null text if capture must not start now
   * @param {()=>void} deps.onChange called after any state change (so app.js can refresh its UI)
   * @param {{fps:number, maxSeconds:number, mirrorInput:boolean, mirrorHands:boolean}} deps.options
   */
  constructor({
    $,
    loadHolistic,
    hasHandSource,
    swapHandedness = true,
    isCameraRunning,
    blockReason,
    onChange,
    options,
  }) {
    this.$ = $;
    this.loadHolistic = loadHolistic;
    this.hasHandSource = hasHandSource;
    this.swapHandedness = swapHandedness;
    this.isCameraRunning = isCameraRunning;
    this.blockReason = blockReason;
    this.onChange = onChange;
    this.options = options;

    this.state = "idle";
    this.holistic = null;
    this.holisticInUse = false; // decided from the schema, on start()
    this.schema = null; // { columns, minFrames, maxFrames }
    this.extractors = null;
    this.session = null;
    this._token = 0;
    this._abort = null;
    this._mirrorCanvas = null;
    this._lastTs = 0;
    this._uiThrottle = 0;
    this._lastWarning = "";

    $("aslStartBtn").addEventListener("click", () => this.start());
    $("aslStopBtn").addEventListener("click", () => this.stopAndRecognize());
    $("aslCancelBtn").addEventListener("click", () =>
      this.cancel("Cancelled."),
    );
  }

  get isCapturing() {
    return this.state === "capturing";
  }
  get busy() {
    return this.state !== "idle";
  }

  // ---- UI ----
  _status(msg, kind = "") {
    const el = this.$("aslStatus");
    el.textContent = msg;
    el.className = kind;
  }

  _setState(state) {
    this.state = state;
    this.refresh();
    this.onChange();
  }

  refresh(force = true) {
    const now = performance.now();
    if (!force && now - this._uiThrottle < 200) return;
    this._uiThrottle = now;
    const cam = this.isCameraRunning();
    this.$("aslStartBtn").disabled = !(cam && this.state === "idle");
    this.$("aslStopBtn").disabled = this.state !== "capturing";
    this.$("aslCancelBtn").disabled = this.state === "idle";
    this.$("aslState").textContent = this.state;
    const s = this.session;
    const capturing = this.state === "capturing" && s;
    this.$("aslFrames").textContent = capturing ? String(s.frameCount) : "0";
    this.$("aslHandFrames").textContent = capturing
      ? String(s.handFrames)
      : "0";
    this.$("aslElapsed").textContent = capturing
      ? fmtSec(s.elapsedMs(this._lastTs))
      : "0.0 s";
    const lim = this.schema && this.session ? this.session.limits : null;
    this.$("aslLimit").textContent = lim
      ? `${lim.maxFrames} frames / ${lim.maxSeconds} s at ${lim.fps} FPS`
      : `${this.options.maxSeconds} s at ${this.options.fps} FPS (API limit confirmed on start)`;
    const det = this.$("aslDetector");
    if (det)
      det.textContent = this.holisticInUse
        ? "HandLandmarker for hands + Holistic for face/pose"
        : this.state === "idle"
          ? "HandLandmarker for hands (face/pose not needed so far)"
          : "HandLandmarker only — schema needs no face/pose";
  }

  _showResult(result, stats) {
    const level = confidenceLevel(result);
    const text = this.$("aslText");
    const meta = this.$("aslMeta");
    // textContent only: response data is never interpreted as HTML.
    text.textContent =
      level === "none" ? "(no letters recognised)" : result.text;
    text.className = `asl-text ${level}`;
    const parts = [];
    if (result.confidence !== null)
      parts.push(`confidence ${(result.confidence * 100).toFixed(1)}%`);
    if (result.ms !== null) parts.push(`${result.ms.toFixed(0)} ms processing`);
    parts.push(`${stats.frames} frames sent, hands in ${stats.handFrames}`);
    if (result.raw !== null && result.raw !== result.text)
      parts.push(`raw: ${result.raw}`);
    let line = parts.join(" · ");
    if (level === "low") {
      line += " — LOW CONFIDENCE: treat this as a guess, not a transcription.";
    }
    meta.textContent = line;
  }

  _clearResult() {
    this.$("aslText").textContent = "–";
    this.$("aslText").className = "asl-text";
    this.$("aslMeta").textContent = "";
  }

  _fail(message) {
    this._status(message, "error");
  }

  // ---- lifecycle ----
  async _ensureSchema(token) {
    if (this.schema) return;
    this._abort = new AbortController();
    const schema = await fetchSchema({ signal: this._abort.signal });
    if (token !== this._token) return;
    this.extractors = buildExtractors(schema.columns); // throws if any column is unrecognised
    this.schema = schema;
  }

  async start() {
    if (this.state !== "idle") return;
    if (!this.isCameraRunning())
      return this._status("Start the camera first.", "warn");
    const blocked = this.blockReason();
    if (blocked) return this._status(blocked, "warn");

    const token = ++this._token;
    this._clearResult();
    this._setState("preparing");
    this._status("Preparing: loading the recognition schema…");
    try {
      await this._ensureSchema(token);
      // The schema decides whether Holistic is needed at all. A hands-only schema (the usual
      // case for fingerspelling) means capture runs the recording path and nothing else: one
      // detector, one inference per frame, no second GPU context.
      this.holisticInUse =
        needsHolistic(this.schema.columns) || !this.hasHandSource?.();
      if (this.holisticInUse) {
        this._status(
          "Preparing: the schema needs face/pose, loading the Holistic model…",
        );
        const holistic = await this.loadHolistic();
        if (token !== this._token) return; // cancelled while loading
        if (!this.isCameraRunning()) throw new Error("The camera was stopped.");
        this.holistic = holistic;
      } else {
        this.holistic = null;
      }
      if (token !== this._token) return;
      this.session = new CaptureSession({
        fps: this.options.fps,
        maxSeconds: this.options.maxSeconds,
        minFrames: this.schema.minFrames,
        apiMaxFrames: this.schema.maxFrames,
        rowLength: this.extractors.length,
      });
      this.session.begin();
      this._lastTs = 0;
      this._setState("capturing");
      this._status(
        `Capturing with the hand detector${this.holisticInUse ? " (plus Holistic for face/pose)" : ""}. Spell your letters clearly, then press “Stop & recognize” (auto-stops after ${this.session.limits.maxSeconds} s).`,
      );
    } catch (err) {
      if (token !== this._token) return;
      console.error(err);
      this._setState("idle");
      this._fail(
        err instanceof AslApiError
          ? `Could not prepare recognition: ${err.message}`
          : `Could not start capture: ${err.message}`,
      );
    }
  }

  /**
   * Called by app.js once per camera frame while isCapturing.
   * @param {HTMLVideoElement} video
   * @param {number} ts frame timestamp, identical to the one the recorder uses
   * @param {object|null} handResult HandLandmarkerResult for THIS frame, produced by the one
   *   hand-model call in app.js's loop. Supplies the hand columns.
   * @returns {object|null} the merged result (nothing is drawn from it any more: app.js draws
   *   the recording overlay from the same HandLandmarkerResult)
   */
  processFrame(video, ts, handResult = null) {
    this._lastTs = ts;

    // Holistic contributes face/pose only, and only when the schema asked for them.
    let holisticResult = null;
    if (this.holistic) {
      try {
        holisticResult = this.holistic.detectForVideo(this._source(video), ts);
      } catch (err) {
        console.error(err);
        this.cancel(`Holistic detection failed: ${err.message}`, "error");
        return null;
      }
    }

    const hands = mergeHandResults(holisticResult, handResult, {
      swapHandedness: this.swapHandedness,
      mirrorHands: !!this.options.mirrorHands,
    });

    const outcome = this.session.offer(
      extractRow(hands, this.extractors),
      hasHands(hands),
      ts,
    );
    if (outcome === "limit") {
      this._status("Capture limit reached; recognizing…", "warn");
      this.stopAndRecognize();
    } else {
      this.refresh(false);
      this._warnIfNoHands();
    }
    return hands;
  }

  _warnIfNoHands() {
    const s = this.session;
    if (!s || s.frameCount < 15) return;
    const warn = s.handFrames / s.frameCount < 0.3;
    if (warn && this._lastWarning !== "nohands") {
      this._lastWarning = "nohands";
      this._status(
        "Few hands detected so far. Bring your hand fully into view.",
        "warn",
      );
    } else if (!warn && this._lastWarning === "nohands") {
      this._lastWarning = "";
      this._status(
        "Capturing. Spell your letters clearly, then press “Stop & recognize”.",
      );
    }
  }

  async stopAndRecognize() {
    if (this.state !== "capturing") return;
    const token = this._token;
    const fin = this.session.finish();
    if (!fin.ok) {
      this._setState("idle");
      return this._status(fin.message, "warn");
    }
    this._setState("recognizing");
    this._status(`Recognizing ${fin.stats.frames} frames…`);
    this._abort = new AbortController();
    try {
      const result = await recognize(fin.frames, {
        signal: this._abort.signal,
      });
      if (token !== this._token) return; // cancelled meanwhile
      this._showResult(result, fin.stats);
      const lowHands = fin.stats.handFraction < 0.5;
      this._status(
        lowHands
          ? "Done, but hands were visible in under half of the frames, so the result is unreliable."
          : "Done. Fingerspelling recognition only; this is not general sign-language translation.",
        lowHands ? "warn" : "",
      );
    } catch (err) {
      if (token !== this._token || err?.kind === "aborted") return;
      console.error(err);
      this._fail(`Recognition failed: ${err.message}`);
    } finally {
      if (token === this._token) {
        this.session?.endSubmit();
        this._abort = null;
        this._setState("idle");
      }
    }
  }

  cancel(message = "Cancelled.", kind = "") {
    if (this.state === "idle") return;
    this._token++; // invalidates any in-flight start()/recognize()
    this._abort?.abort();
    this._abort = null;
    this.session?.cancel();
    this._lastWarning = "";
    this._setState("idle");
    this._status(message, kind);
  }

  // ---- detection helpers ----
  /**
   * Optionally mirror the image fed to Holistic (?aslMirror=1) to match selfie-view training
   * data. This affects the face/pose columns only: the hand columns come from the dedicated
   * hand model, which is always fed the raw frame.
   */
  _source(video) {
    if (!this.options.mirrorInput) return video;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!this._mirrorCanvas)
      this._mirrorCanvas = document.createElement("canvas");
    const c = this._mirrorCanvas;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const g = c.getContext("2d");
    g.save();
    g.translate(w, 0);
    g.scale(-1, 1);
    g.drawImage(video, 0, 0, w, h);
    g.restore();
    return c;
  }
}
