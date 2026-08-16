"""Local development server for the Python inference API.

Serves the same contract as api/step.py (which runs as a Vercel Python
function in deployment) on http://127.0.0.1:8765 so `next dev` can proxy
/api/step to it locally (see next.config.ts rewrite).

Run: .venv/Scripts/python scripts/dev_api.py
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from step import MAX_BODY_BYTES, advance  # noqa: E402

from ppi_core.serialize import canonical_json  # noqa: E402

PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/api/step":
            self._reply(404, {"error": "unknown path"})
            return
        length = int(self.headers.get("content-length", 0))
        if length <= 0 or length > MAX_BODY_BYTES:
            self._reply(413, {"error": f"body length {length} out of bounds"})
            return
        try:
            result = advance(json.loads(self.rfile.read(length)))
            self._reply(200, result)
        except (KeyError, ValueError, TypeError) as exc:
            self._reply(400, {"error": str(exc)})

    def _reply(self, status: int, obj: dict) -> None:
        body = canonical_json(obj).encode("ascii")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature; quiet
        del format, args


if __name__ == "__main__":
    print(f"ppi dev api on http://127.0.0.1:{PORT}/api/step")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
