"""
signBridge Phase 0 spike: static file server.

Python only serves files here. Camera capture and MediaPipe landmark extraction
run in the browser, so no video ever reaches this process.

Why stdlib instead of Flask: this server has no routes or state, so an extra
dependency buys nothing. The real backend (Express + Socket.IO) comes later.

Run:   python main.py
Open:  http://localhost:8000
Note:  getUserMedia() only works on a secure context. http://localhost counts
       as secure; http://192.168.x.x does NOT, so test on the same machine.
"""

import mimetypes
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"  # loopback only; do not expose the camera page to the LAN
PORT = 8000
STATIC_DIR = Path(__file__).parent / "static"

# Some platforms (notably Windows) map .js/.mjs/.wasm to wrong or missing types,
# and browsers refuse to run ES modules served with a non-JS MIME type.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Static handler that disables caching so edits show up on refresh."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


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