// Run: node --test tests/
import test from "node:test";
import assert from "node:assert/strict";

import { buildRecordFrame, normalizeHands } from "../static/js/landmarks.js";
import { Recorder, serializeRecordingParts } from "../static/js/recorder.js";
import { validateRecording } from "../static/js/validate.js";
import { resolveFrames } from "../static/js/sequence.js";

// ---------- fixtures ----------
const pts = (n, f = (i) => i / 100) => Array.from({ length: n }, (_, i) => ({ x: f(i), y: f(i) + 0.1, z: -0.01 * i }));
const hand = (label, score = 0.9, n = 21) => ({ points: pts(n), cat: [{ categoryName: label, score }] });
const handResult = (...hs) => ({ landmarks: hs.map((h) => h.points), handedness: hs.map((h) => h.cat) });
const poseLms = () => pts(33).map((p) => ({ ...p, visibility: 0.8 }));
const faceLms = () => pts(478);

function makeClock() {
  let t = 1000;
  return { now: () => t, set: (v) => (t = v), advance: (d) => (t += d) };
}
function newRecorder(opts = {}) {
  const clock = makeClock();
  const rec = new Recorder({
    now: clock.now,
    clock: () => new Date("2026-10-22T07:00:00.000Z"),
    idGen: () => "test-id-0001",
    ...opts,
  });
  return { rec, clock };
}
const frameAt = (ms, { hands = handResult(), pose, face } = {}) =>
  buildRecordFrame({ frameMs: ms, handResult: hands, pose, face });
const poseSrc = (id, frameMs, landmarks = poseLms()) => ({ available: true, resultId: id, frameMs, landmarks });
const faceSrc = (id, frameMs, landmarks = faceLms()) => ({ available: true, resultId: id, frameMs, landmarks });

// ---------- landmarks adapter ----------
test("handedness is swapped to anatomical sides and raw label kept", () => {
  const h = normalizeHands(handResult(hand("Left")), { swapHandedness: true });
  assert.equal(h.right.detected, true);
  assert.equal(h.right.rawLabel, "Left");
  assert.equal(h.left.detected, false);
  assert.equal(h.right.landmarks.length, 21);

  const h2 = normalizeHands(handResult(hand("Left")), { swapHandedness: false });
  assert.equal(h2.left.detected, true);
});

test("missing hand is an explicit {detected:false}, never zero landmarks", () => {
  const h = normalizeHands(handResult(), {});
  assert.deepEqual(h.left, { detected: false });
  assert.deepEqual(h.right, { detected: false });
});

test("two detections with the same side keep the higher score and report the extra", () => {
  const h = normalizeHands(handResult(hand("Left", 0.6), hand("Left", 0.95)), { swapHandedness: false });
  assert.equal(h.left.handednessScore, 0.95);
  assert.equal(h.unassigned, 1);
});

test("non-finite coordinates become null and are counted", () => {
  const bad = hand("Left");
  bad.points[3] = { x: NaN, y: Infinity, z: 0.1 };
  const rf = frameAt(0, { hands: handResult(bad) });
  assert.equal(rf.invalidValues, 2);
  assert.deepEqual(rf.hands.right.landmarks[3], [null, null, 0.1]);
  assert.ok(!JSON.stringify(rf).includes("NaN"));
});

test("pose/face are null when no result exists, {detected:false} when run and empty", () => {
  const none = frameAt(0, { pose: { available: false }, face: { available: true, resultId: null } });
  assert.equal(none.pose, null);
  assert.equal(none.face, null);
  const empty = frameAt(10, { pose: poseSrc(1, 10, null) });
  assert.equal(empty.pose.detected, false);
  assert.equal(empty.pose.ageMs, 0);
});

test("pose visibility missing from the API is null, not invented", () => {
  const lms = pts(33); // no visibility
  const rf = frameAt(0, { pose: poseSrc(1, 0, lms) });
  assert.equal(rf.pose.landmarks[0][3], null);
  assert.equal(rf.invalidValues, 0);
});

// ---------- recorder ----------
test("records relative, strictly increasing timestamps", () => {
  const { rec, clock } = newRecorder();
  rec.start({});
  for (const dt of [10, 70, 140]) {
    assert.equal(rec.addFrame(frameAt(clock.now() + dt)).ok, true);
  }
  assert.deepEqual(rec.frames.map((f) => f.t), [10, 70, 140]);
});

test("non-monotonic and pre-start frames are rejected, counted, never re-stamped", () => {
  const { rec, clock } = newRecorder();
  rec.start({});
  rec.addFrame(frameAt(1100));
  assert.equal(rec.addFrame(frameAt(1100)).reason, "non_monotonic");
  assert.equal(rec.addFrame(frameAt(1050)).reason, "non_monotonic");
  assert.equal(rec.addFrame(frameAt(500)).reason, "before_start");
  assert.equal(rec.frameCount, 1);
  assert.deepEqual(rec.dropped, { nonMonotonic: 2, beforeStart: 1 });
});

test("reused pose/face results keep true age and reference the source frame", () => {
  const { rec, clock } = newRecorder();
  rec.start({});
  const pose = poseSrc(1, 1100);
  const face = faceSrc(2, 1100);
  rec.addFrame(frameAt(1100, { pose, face })); // fresh
  rec.addFrame(frameAt(1170, { pose, face })); // reused
  rec.addFrame(frameAt(1240, { pose, face })); // reused
  const [f0, f1, f2] = rec.frames;
  assert.equal(f0.pose.ageMs, 0);
  assert.ok(f0.pose.landmarks);
  assert.equal(f1.pose.ageMs, 70);
  assert.equal(f1.pose.sourceIndex, 0);
  assert.equal(f1.pose.landmarks, undefined); // no duplicated payload
  assert.equal(f2.face.ageMs, 140);
  assert.equal(f2.face.sourceIndex, 0);
});

test("a result carried in from before recording started keeps its landmarks and true age", () => {
  const { rec } = newRecorder();
  rec.start({});
  rec.addFrame(frameAt(1050, { pose: poseSrc(7, 1000) })); // computed 50ms before this frame
  rec.addFrame(frameAt(1120, { pose: poseSrc(7, 1000) }));
  assert.equal(rec.frames[0].pose.ageMs, 50);
  assert.ok(rec.frames[0].pose.landmarks);
  assert.equal(rec.frames[1].pose.sourceIndex, 0);
});

test("limit stops explicitly instead of truncating silently", () => {
  const { rec } = newRecorder({ maxFrames: 3 });
  rec.start({});
  for (let i = 1; i <= 3; i++) assert.equal(rec.addFrame(frameAt(1000 + i * 10)).ok, true);
  const r = rec.addFrame(frameAt(1040));
  assert.equal(r.reason, "limit");
  assert.equal(rec.state, "stopped");
  assert.equal(rec.stopReason, "limit");
  assert.equal(rec.toRecording().recording.limits.maxFrames, 3);
  assert.equal(rec.toRecording().recording.frameCount, 3);
});

test("repeated start/stop is safe and exports contain only the current recording", () => {
  const { rec, clock } = newRecorder();
  assert.equal(rec.stop(), false); // stop while idle
  assert.equal(rec.start({}), true);
  assert.equal(rec.start({}), false); // double start ignored
  rec.addFrame(frameAt(1010));
  rec.addFrame(frameAt(1020));
  clock.advance(100);
  assert.equal(rec.stop("user"), true);
  assert.equal(rec.stop("user"), false); // double stop ignored
  assert.equal(rec.addFrame(frameAt(1030)).reason, "not_recording");
  const first = rec.toRecording();
  assert.equal(first.frames.length, 2);

  clock.set(5000);
  rec.start({});
  rec.addFrame(frameAt(5010));
  rec.stop();
  const second = rec.toRecording();
  assert.equal(second.frames.length, 1);
  assert.equal(second.frames[0].t, 10);
  assert.equal(first.frames.length, 2); // earlier export object untouched
});

test("discard releases buffers", () => {
  const { rec } = newRecorder();
  rec.start({});
  rec.addFrame(frameAt(1010, { pose: poseSrc(1, 1010) }));
  rec.stop();
  rec.discard();
  assert.equal(rec.frameCount, 0);
  assert.equal(rec.state, "idle");
  assert.equal(rec.toRecording(), null);
});

// ---------- serialization + validation ----------
async function fullRecording() {
  const { rec, clock } = newRecorder();
  rec.start({ detectors: { hands: { available: true } } });
  const pose = poseSrc(1, 1100);
  const face = faceSrc(2, 1100);
  rec.addFrame(frameAt(1100, { hands: handResult(hand("Left"), hand("Right")), pose, face }));
  rec.addFrame(frameAt(1170, { hands: handResult(hand("Right")), pose, face }));
  rec.addFrame(frameAt(1240, { hands: handResult(), pose, face }));
  clock.advance(300);
  rec.stop();
  return rec.toRecording();
}

test("export is valid JSON with no invalid tokens and validates clean", async () => {
  const doc = await fullRecording();
  const parts = await serializeRecordingParts(doc, { sliceSize: 2 });
  const text = parts.join("");
  assert.ok(!/NaN|Infinity|undefined/.test(text));
  const parsed = JSON.parse(text);
  assert.equal(parsed.schemaVersion, 1);
  assert.equal(parsed.recording.frameCount, 3);
  assert.deepEqual(parsed, JSON.parse(JSON.stringify(doc))); // chunked == plain stringify

  const v = validateRecording(parsed);
  assert.equal(v.valid, true, JSON.stringify(v.errors));
  assert.equal(v.stats.reusedPose, 2);
  assert.equal(v.stats.framesWithNoHands, 1);
});

test("resolveFrames expands references without mutating the document", async () => {
  const doc = await fullRecording();
  const before = JSON.stringify(doc);
  const resolved = resolveFrames(doc);
  assert.equal(resolved[2].pose.landmarks.length, 33);
  assert.equal(resolved[2].pose.ageMs, 140);
  assert.equal(JSON.stringify(doc), before);
});

const clone = (o) => JSON.parse(JSON.stringify(o));
const codes = (r) => [...r.errors, ...r.warnings].map((i) => i.code);

test("validator catches wrong hand landmark count", async () => {
  const d = clone(await fullRecording());
  d.frames[0].hands.left.landmarks.pop();
  assert.ok(codes(validateRecording(d)).includes("bad_hand_landmark_count"));
  assert.equal(validateRecording(d).valid, false);
});

test("validator catches non-finite / missing coordinates", async () => {
  const d = clone(await fullRecording());
  d.frames[0].hands.left.landmarks[0][1] = null;
  d.frames[1].face.landmarks;
  assert.ok(codes(validateRecording(d)).includes("non_finite_coordinate"));
});

test("validator catches invalid confidence and visibility", async () => {
  const d = clone(await fullRecording());
  d.frames[0].hands.left.handednessScore = 1.5;
  d.frames[0].pose.landmarks[0][3] = -0.2;
  const c = codes(validateRecording(d));
  assert.ok(c.includes("bad_confidence"));
  assert.ok(c.includes("bad_visibility"));
});

test("validator catches bad and out-of-order timestamps", async () => {
  const d = clone(await fullRecording());
  d.frames[1].t = d.frames[0].t; // equal
  d.frames[2].t = -5;
  const c = codes(validateRecording(d));
  assert.ok(c.includes("non_monotonic_timestamp"));
  assert.ok(c.includes("bad_timestamp"));
});

test("validator flags stale results and bad references", async () => {
  const d = clone(await fullRecording());
  d.frames[2].pose.ageMs = 900;
  const v = validateRecording(d, { staleMs: 500 });
  assert.ok(v.warnings.some((w) => w.code === "stale_pose"));

  d.frames[1].face.sourceIndex = 5;
  assert.ok(codes(validateRecording(d)).includes("bad_source_index"));
});

test("validator flags frame count / duration mismatches and missing modalities", async () => {
  const d = clone(await fullRecording());
  d.recording.frameCount = 9;
  d.recording.durationMs = 1;
  const c = codes(validateRecording(d));
  assert.ok(c.includes("frame_count_mismatch"));
  assert.ok(c.includes("duration_shorter_than_frames"));

  const e = clone(await fullRecording());
  e.frames.forEach((f) => {
    f.face = null;
    f.pose = null;
  });
  const v = validateRecording(e);
  assert.ok(v.warnings.some((w) => w.code === "face_never_detected"));
  assert.ok(v.warnings.some((w) => w.code === "pose_never_detected"));
  assert.equal(v.stats.faceUnavailable, 3);
  assert.equal(v.valid, true); // missing modalities are warnings, not corruption
});

test("validator handles garbage input without throwing", () => {
  assert.equal(validateRecording(null).valid, false);
  assert.equal(validateRecording({ schemaVersion: 1 }).valid, false);
  assert.equal(validateRecording({ schemaVersion: 1, recording: {}, frames: "x" }).valid, false);
});
