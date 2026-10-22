/**
 * Client for the local ASL fingerspelling proxy (main.py): GET /api/asl/schema and
 * POST /api/asl/recognize. The browser talks to its own origin; main.py talks to three.ws.
 * No DOM; `fetchFn` is injectable so the error paths are unit-tested in Node.
 *
 * Every failure is an AslApiError whose `kind` is one of:
 *   "network" | "timeout" | "aborted" | "api" | "invalid_response"
 */

import { validateSchema } from "./asl-features.js";

export const API_BASE = "/api/asl";
export const LOW_CONFIDENCE = 0.5;

export class AslApiError extends Error {
  constructor(kind, message, status = null) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

async function request(path, init, { fetchFn, signal, timeoutMs, base }) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onAbort = () => controller.abort();
  if (signal?.aborted) controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    let res;
    try {
      res = await fetchFn(base + path, { ...init, signal: controller.signal });
    } catch (err) {
      if (timedOut)
        throw new AslApiError(
          "timeout",
          "The recognition service took too long to respond.",
        );
      if (signal?.aborted || err?.name === "AbortError")
        throw new AslApiError("aborted", "Request cancelled.");
      throw new AslApiError(
        "network",
        "Could not reach the local server. Is `python main.py` still running?",
      );
    }
    let body = null;
    let parseFailed = false;
    try {
      body = await res.json();
    } catch {
      parseFailed = true;
    }
    if (!res.ok) {
      const detail =
        typeof body?.error === "string" ? body.error : `HTTP ${res.status}`;
      throw new AslApiError("api", detail, res.status);
    }
    if (parseFailed)
      throw new AslApiError(
        "invalid_response",
        "The server did not return valid JSON.",
      );
    return body;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

export async function fetchSchema({
  fetchFn = fetch,
  signal,
  timeoutMs = 15_000,
  base = API_BASE,
} = {}) {
  const body = await request(
    "/schema",
    { method: "GET" },
    { fetchFn, signal, timeoutMs, base },
  );
  try {
    return validateSchema(body);
  } catch (err) {
    throw new AslApiError(
      "invalid_response",
      `Invalid schema from server: ${err.message}`,
    );
  }
}

/** Validate and normalise a recognition response before anything is rendered. */
export function validateRecognition(body) {
  const bad = (why) =>
    new AslApiError(
      "invalid_response",
      `Unexpected recognition response: ${why}`,
    );
  if (!body || typeof body !== "object") throw bad("not an object");
  if (typeof body.text !== "string") throw bad("`text` is not a string");
  const optNum = (key, lo = -Infinity, hi = Infinity) => {
    const v = body[key];
    if (v === undefined || v === null) return null;
    if (typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi)
      throw bad(`\`${key}\` is invalid`);
    return v;
  };
  const optStr = (key) => (typeof body[key] === "string" ? body[key] : null);
  return {
    text: body.text,
    confidence: optNum("confidence", 0, 1),
    ms: optNum("ms", 0),
    frames: optNum("frames", 0),
    raw: optStr("raw"),
    cleaned: optStr("cleaned"),
  };
}

export async function recognize(
  frames,
  { fetchFn = fetch, signal, timeoutMs = 45_000, base = API_BASE } = {},
) {
  const body = await request(
    "/recognize",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    },
    { fetchFn, signal, timeoutMs, base },
  );
  return validateRecognition(body);
}

/** "none" (empty text) | "low" (< LOW_CONFIDENCE or unknown) | "ok". Never "certain". */
export function confidenceLevel(result) {
  if (!result.text.trim()) return "none";
  if (result.confidence === null || result.confidence < LOW_CONFIDENCE)
    return "low";
  return "ok";
}
