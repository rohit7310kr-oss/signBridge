// signBridge: webcam -> MediaPipe HandLandmarker -> recordable, validated JSON (schema v2, hand-only).
// This app only collects hand landmarks. It does NOT recognize signs yet.

import { buildRecordFrame } from "./landmarks.js";
import {
  Recorder,
  DEFAULT_MAX_FRAMES,
  serializeRecordingParts,
  exportFilename,
} from "./recorder.js";
import { validateRecording, summarizeValidation } from "./validate.js";
import { AslController } from "./asl.js";

// Pinned version. If this 404s, check https://www.npmjs.com/package/@mediapipe/tasks-vision
// for a current release and update BOTH URLs below together.
const MP_VERSION = "0.10.21";
const MP_BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL_NAME = "hand_landmarker/float16/1";
const HAND_MODEL = `https://storage.googleapis.com/mediapipe-models/hand_landmarker/${HAND_MODEL_NAME}/hand_landmarker.task`;
// Holistic (face + hands + pose) is only used while an ASL fingerspelling capture is running.
const HOLISTIC_MODEL = `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`;

// Tuning knobs via URL, so you can compare runs without editing code:
//   http://localhost:8000/?delegate=CPU&maxFrames=900&swapHands=0
const params = new URLSearchParams(location.search);
const DELEGATE = params.get("delegate") === "CPU" ? "CPU" : "GPU";
const MAX_FRAMES = Math.max(
  1,
  Number(params.get("maxFrames")) || DEFAULT_MAX_FRAMES,
);
// MediaPipe labels handedness assuming a mirrored image; we feed the raw frame, so swap.
// Use ?swapHands=0 if your own test shows the labels are already anatomical.
const SWAP_HANDEDNESS = params.get("swapHands") !== "0";
// ASL fingerspelling capture: ?aslFps=15&aslSeconds=15&aslMirror=1
//   aslMirror=1 feeds Holistic a horizontally mirrored image. Try it if recognition is poor, since
//   the API's training-data orientation is not documented.
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const ASL_OPTIONS = {
  fps: clamp(Number(params.get("aslFps")) || 15, 5, 30),
  maxSeconds: clamp(Number(params.get("aslSeconds")) || 15, 2, 60),
  mirrorInput: params.get("aslMirror") === "1",
};

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const video = $("video"),
  canvas = $("overlay"),
  ctx = canvas.getContext("2d");
const startBtn = $("startBtn"),
  stopBtn = $("stopBtn");
const recStartBtn = $("recStartBtn"),
  recStopBtn = $("recStopBtn"),
  exportBtn = $("exportBtn"),
  discardBtn = $("discardBtn");
const scrubber = $("scrubber");

function setStatus(msg, kind = "") {
  $("status").textContent = msg;
  $("status").className = kind;
}

// Fail visibly if the CDN is unreachable, instead of a silently dead button.
let FilesetResolver, HandLandmarker, HolisticLandmarker, DrawingUtils;
try {
  ({ FilesetResolver, HandLandmarker, HolisticLandmarker, DrawingUtils } =
    await import(MP_BUNDLE));
} catch (err) {
  console.error(err);
  setStatus(
    "Could not load MediaPipe from the CDN. Check your internet connection or the MP_VERSION value.",
    "error",
  );
  startBtn.disabled = true;
  throw err;
}
// MediaPipe's JS API does not report which delegate actually executes, so we only state what we asked for.
$("delegate").textContent =
  `${DELEGATE} requested (actual delegate not reported by MediaPipe)`;
$("limitInfo").textContent = `${MAX_FRAMES} frames`;

// ---- Stage 1: camera capture ----
let stream = null;
let cameraInfo = null;

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "Camera API unavailable. Use http://localhost or HTTPS in a modern browser.",
    );
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    });
  } catch (err) {
    const messages = {
      NotAllowedError:
        "Camera permission was denied. Allow camera access in the browser's site settings and try again.",
      NotFoundError: "No camera found on this device.",
      NotReadableError: "The camera is in use by another application.",
    };
    throw new Error(
      messages[err.name] ?? `Could not open camera: ${err.message}`,
    );
  }
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const s = stream.getVideoTracks()[0]?.getSettings?.() ?? {};
  cameraInfo = {
    width: video.videoWidth,
    height: video.videoHeight,
    nominalFrameRate: s.frameRate ?? null, // what the driver says, not a measurement
    facingMode: s.facingMode ?? null,
  };
}

function stopCamera() {
  asl?.cancel("Capture cancelled because the camera stopped.", "warn");
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ---- Stage 2: hand landmark extraction ----
let handLandmarker = null;

async function createHandLandmarker() {
  if (handLandmarker) return; // created once, reused across camera restarts
  const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: DELEGATE },
    runningMode: "VIDEO",
    numHands: 2, // ISL has many two-handed signs
  });
}

// Holistic is created lazily on the first ASL capture and then reused.
let holistic = null;
let holisticPromise = null;

function loadHolistic() {
  if (holistic) return Promise.resolve(holistic);
  if (!HolisticLandmarker) {
    return Promise.reject(
      new Error(
        "This MediaPipe build does not export HolisticLandmarker. Check MP_VERSION.",
      ),
    );
  }
  holisticPromise ??= (async () => {
    const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
    holistic = await HolisticLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HOLISTIC_MODEL, delegate: DELEGATE },
      runningMode: "VIDEO",
    });
    return holistic;
  })().catch((err) => {
    holisticPromise = null; // allow a retry after a network failure
    throw new Error(`Could not load the Holistic model: ${err.message}`);
  });
  return holisticPromise;
}

// ---- Recording ----
const recorder = new Recorder({ maxFrames: MAX_FRAMES });
let timestampSource = "performance.now()";
let ignoreNextGapForRecorder = false;

function recordingMeta() {
  return {
    app: "signBridge",
    mediapipeVersion: MP_VERSION,
    model: HAND_MODEL_NAME,
    numHands: 2,
    delegateRequested: DELEGATE, // MediaPipe does not report the delegate actually used
    swapHandedness: SWAP_HANDEDNESS,
    timestampSource, // where frame.t comes from (see docs/schema.md)
    camera: cameraInfo,
  };
}

/** Shorten landmark arrays for on-screen display only; exports are never abbreviated. */
function abbreviate(frame) {
  const brief = (lms) =>
    Array.isArray(lms) ? { count: lms.length, first: lms.slice(0, 2) } : lms;
  const hand = (h) =>
    h?.detected ? { ...h, landmarks: brief(h.landmarks) } : h;
  return {
    ...frame,
    hands: {
      ...frame.hands,
      left: hand(frame.hands.left),
      right: hand(frame.hands.right),
    },
  };
}

const fmtSec = (ms) => `${(ms / 1000).toFixed(1)} s`;

let recUiThrottle = 0;
function updateRecUI(force = false) {
  const now = performance.now();
  if (!force && now - recUiThrottle < 250) return;
  recUiThrottle = now;

  const state = recorder.state;
  const badge = $("recBadge");
  badge.textContent =
    state === "recording"
      ? "● REC"
      : state === "stopped"
        ? "stopped"
        : "not recording";
  badge.className = state === "recording" ? "badge on" : "badge";

  $("recState").textContent =
    state === "stopped" ? `stopped (${recorder.stopReason})` : state;
  $("recFrames").textContent = String(recorder.frameCount);
  $("recElapsed").textContent = fmtSec(recorder.elapsedMs);
  const d = recorder.dropped;
  $("recRejected").textContent = String(
    d.nonMonotonic + d.beforeStart + d.invalidTimestamp,
  );
  $("recSkipped").textContent = String(d.cameraSkipped);

  recStartBtn.disabled = !running || state === "recording" || asl.busy;
  recStopBtn.disabled = state !== "recording";
  exportBtn.disabled = !recorder.canExport;
  discardBtn.disabled = state === "idle";

  if (state === "recording" && recorder.lastFrame) {
    $("lastFrame").textContent = JSON.stringify(
      abbreviate(recorder.lastFrame),
      null,
      2,
    );
  }
}

function showScrubbedFrame() {
  const f = recorder.frames[Number(scrubber.value)];
  $("scrubInfo").textContent = f
    ? `frame ${f.frameIndex} / ${recorder.frameCount - 1}  (t = ${f.t} ms)`
    : "–";
  $("scrubFrame").textContent = f
    ? JSON.stringify(abbreviate(f), null, 2)
    : "(no frame)";
}

// Called for every way a recording can end (user, camera stop, or the Recorder's own limit stop).
function stopRecording(reason) {
  if (recorder.state === "recording") recorder.stop(reason);
  if (recorder.state !== "stopped") return;
  if (recorder.stopReason === "limit") {
    setStatus(
      `Recording stopped: the ${MAX_FRAMES}-frame limit was reached. Nothing was truncated; raise it with ?maxFrames=N.`,
      "warn",
    );
  }
  scrubber.max = String(Math.max(0, recorder.frameCount - 1));
  scrubber.value = "0";
  scrubber.disabled = recorder.frameCount === 0;
  showScrubbedFrame();
  updateRecUI(true);
  // Validate off the render path so stopping never stalls the preview.
  $("validation").textContent = "validating…";
  const id = recorder.recordingId;
  setTimeout(() => {
    if (recorder.recordingId !== id || recorder.state !== "stopped") return; // discarded meanwhile
    const result = validateRecording(recorder.toRecording());
    $("validation").textContent =
      summarizeValidation(result) +
      "\n" +
      JSON.stringify(
        {
          errors: result.errors,
          warnings: result.warnings,
          stats: result.stats,
        },
        null,
        2,
      );
  }, 0);
}

function resetInspector(text = "(no recording)") {
  scrubber.disabled = true;
  scrubber.value = "0";
  scrubber.max = "0";
  $("lastFrame").textContent = "(nothing yet)";
  $("scrubFrame").textContent = text;
  $("scrubInfo").textContent = "–";
  $("validation").textContent = "(no recording)";
}

recStartBtn.addEventListener("click", () => {
  if (!running) return;
  if (
    recorder.hasUnexportedData &&
    !confirm("Replace the previous recording? It has not been exported.")
  ) {
    return;
  }
  recorder.start(recordingMeta());
  ignoreNextGapForRecorder = true;
  resetInspector("(stop recording to inspect)");
  $("validation").textContent = "(recording…)";
  updateRecUI(true);
});

recStopBtn.addEventListener("click", () => stopRecording("user"));

exportBtn.addEventListener("click", async () => {
  if (!recorder.canExport) return;
  const doc = recorder.toRecording();
  exportBtn.disabled = true;
  try {
    // Serialization yields to the event loop between slices so the preview stays live.
    const parts = await serializeRecordingParts(doc);
    if (
      recorder.recordingId !== doc.recording.recordingId ||
      recorder.state !== "stopped"
    ) {
      setStatus(
        "Export cancelled: the recording was discarded or replaced while exporting.",
        "warn",
      );
      return;
    }
    const blob = new Blob(parts, { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = exportFilename(doc);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    recorder.markExported();
    setStatus(
      `Exported ${doc.frames.length} frames (${(blob.size / 1024).toFixed(0)} KB).`,
    );
  } catch (err) {
    console.error(err);
    setStatus(`Export failed: ${err.message}`, "error");
  } finally {
    updateRecUI(true);
  }
});

discardBtn.addEventListener("click", () => {
  if (
    recorder.state === "recording" &&
    !confirm("Discard the recording in progress?")
  )
    return;
  recorder.discard();
  resetInspector();
  setStatus("Recording discarded.");
  updateRecUI(true);
});

scrubber.addEventListener("input", showScrubbedFrame);

// ---- ASL fingerspelling (see asl.js) ----
const asl = new AslController({
  $,
  loadHolistic,
  isCameraRunning: () => running,
  blockReason: () =>
    recorder.state === "recording"
      ? "Stop the landmark recording first; ASL capture uses a different detector."
      : null,
  onChange: () => updateRecUI(true),
  options: ASL_OPTIONS,
});

// ---- Render / debug (not part of the pipeline) ----
let drawer = null;
let previewThrottle = 0;

function describeHand(slot) {
  if (!slot.detected) return "not detected";
  const c = slot.confidence === null ? "n/a" : slot.confidence.toFixed(2);
  return `detected (confidence ${c})`;
}

function render(handResult, rf) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawer ??= new DrawingUtils(ctx);
  for (const points of handResult.landmarks) {
    drawer.drawConnectors(points, HandLandmarker.HAND_CONNECTIONS, {
      color: "#00e5a0",
      lineWidth: 3,
    });
    drawer.drawLandmarks(points, { color: "#ff4d6d", radius: 3 });
  }
  $("leftHand").textContent = describeHand(rf.hands.left);
  $("rightHand").textContent = describeHand(rf.hands.right);
}

/** Overlay for the Holistic path (ASL capture). Hands only; Holistic gives no handedness score. */
function renderAsl(hands) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawer ??= new DrawingUtils(ctx);
  for (const points of [hands.left, hands.right]) {
    if (!points) continue;
    drawer.drawConnectors(points, HandLandmarker.HAND_CONNECTIONS, {
      color: "#4da3ff",
      lineWidth: 3,
    });
    drawer.drawLandmarks(points, { color: "#ffd166", radius: 3 });
  }
  $("leftHand").textContent = hands.left
    ? "detected (Holistic)"
    : "not detected";
  $("rightHand").textContent = hands.right
    ? "detected (Holistic)"
    : "not detected";
  const now = performance.now();
  if (now - previewThrottle > 250) {
    previewThrottle = now;
    $("frameJson").textContent =
      "ASL capture running with the Holistic detector.\nThe hand JSON preview is paused until it ends.";
  }
}

function renderPreviewJson(rf) {
  // Rewriting a <pre> every frame is wasteful; the caller refreshes ~4x/second.
  if (recorder.state === "recording") return; // the recording panel shows the captured frame
  $("frameJson").textContent = JSON.stringify(
    abbreviate({ note: "live preview, not being recorded", hands: rf.hands }),
    null,
    2,
  );
}

// ---- Main loop ----
// Preferred: requestVideoFrameCallback, one callback per new camera frame, with the browser's own
// frame timing and a presentedFrames counter (lets us measure camera FPS and frames we skipped).
// Fallback: requestAnimationFrame + currentTime polling, timestamped with performance.now().
const HAS_RVFC = typeof video.requestVideoFrameCallback === "function";
let running = false;
let callbackHandle = null;
let lastVideoTime = -1;
let lastDetectTs = -Infinity;
let lastPresented = null;

const perf = {
  windowStart: performance.now(),
  processed: 0,
  presented: 0,
  handMsSum: 0,
  loopMsSum: 0,
};
let skippedTotal = 0;

function frameTimestamp(now, metadata) {
  if (metadata && Number.isFinite(metadata.captureTime)) {
    timestampSource = "requestVideoFrameCallback.captureTime";
    return metadata.captureTime;
  }
  if (metadata && Number.isFinite(metadata.presentationTime)) {
    timestampSource = "requestVideoFrameCallback.presentationTime";
    return metadata.presentationTime;
  }
  timestampSource = "performance.now()";
  return now;
}

function processFrame(ts, presentedFrames) {
  const loopStart = performance.now();

  // MediaPipe requires strictly increasing timestamps; never feed it (or the recorder) a stale one.
  if (!(ts > lastDetectTs)) return;
  lastDetectTs = ts;

  if (presentedFrames != null) {
    if (lastPresented != null) {
      const gap = presentedFrames - lastPresented - 1;
      if (gap > 0) {
        skippedTotal += gap;
        if (ignoreNextGapForRecorder) ignoreNextGapForRecorder = false;
        else recorder.addCameraSkipped(gap);
      } else {
        ignoreNextGapForRecorder = false;
      }
      perf.presented += presentedFrames - lastPresented;
    }
    lastPresented = presentedFrames;
  }

  // One detector per frame: Holistic while an ASL capture runs, otherwise HandLandmarker.
  const h0 = performance.now();
  let handResult = null;
  let aslHands = null;
  if (asl.isCapturing) aslHands = asl.processFrame(video, ts);
  else handResult = handLandmarker.detectForVideo(video, ts);
  const handMs = performance.now() - h0;

  const now = performance.now();
  if (handResult) {
    const rf = buildRecordFrame({
      frameMs: ts,
      handResult,
      swapHandedness: SWAP_HANDEDNESS,
    });
    if (recorder.state === "recording") {
      const res = recorder.addFrame(rf);
      if (!res.ok && res.reason === "limit") stopRecording("limit"); // Recorder already stopped itself
    }
    render(handResult, rf);
    if (now - previewThrottle > 250) {
      previewThrottle = now;
      renderPreviewJson(rf);
    }
  } else if (aslHands) {
    renderAsl(aslHands);
  }
  updateRecUI();

  perf.processed++;
  perf.handMsSum += handMs;
  perf.loopMsSum += now - loopStart;
  const dt = now - perf.windowStart;
  if (dt >= 1000) {
    $("detectFps").textContent = (perf.processed / (dt / 1000)).toFixed(1);
    $("cameraFps").textContent = HAS_RVFC
      ? (perf.presented / (dt / 1000)).toFixed(1)
      : "n/a (no requestVideoFrameCallback)";
    $("handMs").textContent =
      `${(perf.handMsSum / perf.processed).toFixed(1)} ms avg`;
    $("loopMs").textContent =
      `${(perf.loopMsSum / perf.processed).toFixed(1)} ms avg`;
    $("skipped").textContent = String(skippedTotal);
    Object.assign(perf, {
      windowStart: now,
      processed: 0,
      presented: 0,
      handMsSum: 0,
      loopMsSum: 0,
    });
  }
}

function scheduleNext() {
  if (!running) return;
  if (HAS_RVFC) {
    callbackHandle = video.requestVideoFrameCallback((now, metadata) => {
      if (!running) return;
      processFrame(frameTimestamp(now, metadata), metadata.presentedFrames);
      scheduleNext();
    });
  } else {
    callbackHandle = requestAnimationFrame(() => {
      if (!running) return;
      if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        processFrame(frameTimestamp(performance.now(), null), null);
      }
      scheduleNext();
    });
  }
}

function cancelLoop() {
  if (callbackHandle === null) return;
  if (HAS_RVFC) video.cancelVideoFrameCallback?.(callbackHandle);
  else cancelAnimationFrame(callbackHandle);
  callbackHandle = null;
}

// ---- Wiring ----
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  try {
    setStatus("Loading hand model…");
    await createHandLandmarker();
    setStatus("Requesting camera…");
    await startCamera();
    running = true;
    lastVideoTime = -1;
    lastDetectTs = -Infinity;
    lastPresented = null;
    skippedTotal = 0;
    Object.assign(perf, {
      windowStart: performance.now(),
      processed: 0,
      presented: 0,
      handMsSum: 0,
      loopMsSum: 0,
    });
    stopBtn.disabled = false;
    asl.refresh();
    setStatus(
      "Running. Show one or both hands to the camera. This only detects hand landmarks; it does not recognize signs yet.",
    );
    updateRecUI(true);
    scheduleNext();
  } catch (err) {
    console.error(err);
    setStatus(err.message, "error");
    stopCamera();
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener("click", () => {
  if (recorder.state === "recording") stopRecording("camera_stopped");
  running = false;
  cancelLoop();
  stopCamera();
  stopBtn.disabled = true;
  startBtn.disabled = false;
  asl.refresh();
  setStatus(
    recorder.canExport
      ? "Camera stopped. Your recording is still available to export."
      : "Stopped.",
  );
  updateRecUI(true);
});

// Release the camera if the page is closed or navigated away from.
window.addEventListener("pagehide", () => {
  running = false;
  cancelLoop();
  stopCamera();
});

asl.refresh();
updateRecUI(true);
