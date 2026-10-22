"""
signBridge: static file server for the hand-landmark recorder.

Python only serves files here. Camera capture and MediaPipe hand landmark extraction
run in the browser, so no video ever reaches this process.

Why stdlib instead of Flask: this server has no routes or state, so an extra
dependency buys nothing. The real backend (Express + Socket.IO) comes later.

ASL fingerspelling proxy: the browser never calls three.ws directly. It calls two fixed
routes on this server, which forward to one hard-wired upstream endpoint:
    GET  /api/asl/schema      -> upstream schema (cached briefly)
    POST /api/asl/recognize   -> upstream recognition, after validating the payload
Only landmark numbers are forwarded (never video), and request bodies are never logged.
This is not a generic URL proxy, and it refuses requests whose Host/Origin is not loopback.

Run:   python main.py
Open:  http://localhost:8000
Note:  getUserMedia() only works on a secure context. http://localhost counts
       as secure; http://192.168.x.x does NOT, so test on the same machine.
"""

import json
import math
import mimetypes
import socket
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HOST = "127.0.0.1"  # loopback only; do not expose the camera page to the LAN
PORT = 8000
STATIC_DIR = Path(__file__).parent / "static"

# --- ASL fingerspelling proxy (the upstream URL is fixed on purpose) ---
UPSTREAM_URL = "https://three.ws/api/asl-recognition"
MAX_BODY_BYTES = 12 * 1024 * 1024  # 1500 frames x 390 values is roughly 6 MB of JSON
MAX_UPSTREAM_BYTES = 2 * 1024 * 1024
UPSTREAM_TIMEOUT_S = 30
SCHEMA_TTL_S = 600
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1"}

# Some platforms (notably Windows) map .js/.mjs/.wasm to wrong or missing types,
# and browsers refuse to run ES modules served with a non-JS MIME type.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


class ProxyError(Exception):
    """An error that maps directly to an HTTP status and a JSON {"error": ...} body."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def _reject_constant(name):
    raise ValueError(f"non-finite number {name} is not valid JSON")


def _upstream(method, body=None):
    """Call the single fixed upstream endpoint. Returns parsed JSON (a dict)."""
    req = urllib.request.Request(
        UPSTREAM_URL,
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "User-Agent": "signBridge-local-proxy",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as resp:
            raw = resp.read(MAX_UPSTREAM_BYTES + 1)
    except urllib.error.HTTPError as err:
        status = 429 if err.code == 429 else 502
        raise ProxyError(status, f"The recognition service returned HTTP {err.code}.") from err
    except (TimeoutError, socket.timeout) as err:
        raise ProxyError(504, "The recognition service timed out.") from err
    except urllib.error.URLError as err:
        if isinstance(err.reason, (TimeoutError, socket.timeout)):
            raise ProxyError(504, "The recognition service timed out.") from err
        raise ProxyError(502, "Could not reach the recognition service.") from err
    except OSError as err:
        raise ProxyError(502, "Could not reach the recognition service.") from err
    if len(raw) > MAX_UPSTREAM_BYTES:
        raise ProxyError(502, "The recognition service sent an unexpectedly large response.")
    try:
        data = json.loads(raw, parse_constant=_reject_constant)
    except ValueError as err:
        raise ProxyError(502, "The recognition service did not return valid JSON.") from err
    if not isinstance(data, dict):
        raise ProxyError(502, "The recognition service returned an unexpected response.")
    return data


def _check_schema(schema):
    columns = schema.get("columns")
    lo, hi = schema.get("min_frames"), schema.get("max_frames")
    ok = (
        isinstance(columns, list)
        and columns
        and all(isinstance(c, str) for c in columns)
        and isinstance(lo, int) and not isinstance(lo, bool) and lo >= 1
        and isinstance(hi, int) and not isinstance(hi, bool) and hi >= lo
    )
    if not ok:
        raise ProxyError(502, "The recognition service returned an invalid schema.")
    return schema


_schema_cache = {"value": None, "at": 0.0}
_schema_lock = threading.Lock()


def get_schema():
    with _schema_lock:
        fresh = time.monotonic() - _schema_cache["at"] < SCHEMA_TTL_S
        if _schema_cache["value"] is None or not fresh:
            _schema_cache["value"] = _check_schema(_upstream("GET"))
            _schema_cache["at"] = time.monotonic()
        return _schema_cache["value"]


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_frames(payload, schema):
    """Validate {"frames": [[...], ...]} against the upstream schema; return the frames."""
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ProxyError(400, 'Body must be a JSON object like {"frames": [[...], ...]}.')
    frames = payload["frames"]
    lo, hi, width = schema["min_frames"], schema["max_frames"], len(schema["columns"])
    if not lo <= len(frames) <= hi:
        raise ProxyError(400, f"Expected between {lo} and {hi} frames, got {len(frames)}.")
    for i, row in enumerate(frames):
        if not isinstance(row, list) or len(row) != width:
            raise ProxyError(400, f"Frame {i} must be a list of exactly {width} values.")
        if not all(v is None or _is_number(v) for v in row):
            raise ProxyError(400, f"Frame {i} contains a value that is not a finite number or null.")
    return frames


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Static handler that disables caching, plus the two fixed ASL proxy routes."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---- ASL proxy ----
    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _hostname(value):
        if not value:
            return None
        return (urlsplit(value).hostname if "//" in value else value.split(":")[0]) or None

    def _require_local(self):
        """Block other websites (and DNS-rebinding tricks) from using this proxy."""
        host = self._hostname(self.headers.get("Host"))
        origin = self.headers.get("Origin")
        if host not in LOCAL_HOSTNAMES or (
            origin is not None and self._hostname(origin) not in LOCAL_HOSTNAMES
        ):
            raise ProxyError(403, "Requests must come from this machine (localhost).")

    def _read_json_body(self):
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ProxyError(415, "Content-Type must be application/json.")
        length = self.headers.get("Content-Length")
        if length is None:
            raise ProxyError(411, "Content-Length is required.")
        try:
            n = int(length)
        except ValueError:
            raise ProxyError(400, "Invalid Content-Length.") from None
        if n < 0:
            raise ProxyError(400, "Invalid Content-Length.")
        if n > MAX_BODY_BYTES:
            raise ProxyError(413, f"Payload too large (limit {MAX_BODY_BYTES // (1024 * 1024)} MB).")
        raw = self.rfile.read(n)
        if len(raw) != n:
            raise ProxyError(400, "Request body was truncated.")
        try:
            return json.loads(raw, parse_constant=_reject_constant)
        except ValueError:
            raise ProxyError(400, "Body is not valid JSON.") from None

    def _handle_api(self, route):
        try:
            self._require_local()
            self._send_json(200, route())
        except ProxyError as err:
            self._send_json(err.status, {"error": err.message})
        except Exception:  # never leak internals or payloads
            self.log_error("unexpected error in ASL proxy")
            self._send_json(500, {"error": "Internal proxy error."})

    def _route_schema(self):
        return get_schema()

    def _route_recognize(self):
        payload = self._read_json_body()
        frames = validate_frames(payload, get_schema())
        body = json.dumps({"frames": frames}, separators=(",", ":")).encode("utf-8")
        return _upstream("POST", body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/asl/schema":
            return self._handle_api(self._route_schema)
        if path.startswith("/api/"):
            return self._send_json(404, {"error": "Unknown API route."})
        super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/api/asl/recognize":
            return self._handle_api(self._route_recognize)
        self._send_json(404, {"error": "Unknown API route."})


def main() -> None:
    if not (STATIC_DIR / "index.html").exists():
        raise SystemExit(f"Missing {STATIC_DIR / 'index.html'}")

    handler = partial(NoCacheHandler, directory=str(STATIC_DIR))
    with ThreadingHTTPServer((HOST, PORT), handler) as server:
        print(f"signBridge spike running at http://localhost:{PORT}  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()