"""Vercel Python function: advance an experiment run by one chunk.

POST /api/step
Body: {"state": <RunState>, "labels": [..] | null}  ->  {"state": <RunState'>}
POST /api/step with {"init": {"config": {...}, "data": {...}}} -> round-zero state.

Thin handler only: parse -> ppi_core.runner -> serialize. The runner is
pure and deterministic; all experiment logic lives in python/ppi_core.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# Vercel bundles the repo with the function; python/ holds ppi_core.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ppi_core import runner  # noqa: E402
from ppi_core.serialize import canonical_json  # noqa: E402

MAX_BODY_BYTES = 15 * 1024 * 1024  # pools of ~2k records serialize well under this


def advance(payload: dict) -> dict:
    if "init" in payload:
        init = payload["init"]
        state = runner.init_state(init["config"], init["data"])
        return {"state": state}
    state = payload["state"]
    labels = payload.get("labels")
    new_state = runner.step(state, labels=labels)
    return {"state": new_state}


class handler(BaseHTTPRequestHandler):  # noqa: N801 (Vercel naming contract)
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        if length <= 0 or length > MAX_BODY_BYTES:
            self._reply(413, {"error": f"body length {length} out of bounds"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            result = advance(payload)
            self._reply(200, result)
        except (KeyError, ValueError, TypeError) as exc:
            self._reply(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - last-resort surface
            self._reply(500, {"error": f"internal: {exc}"})

    def _reply(self, status: int, obj: dict) -> None:
        body = canonical_json(obj).encode("ascii")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
