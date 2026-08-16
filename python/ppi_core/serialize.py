"""Canonical, deterministic JSON serialization for run results.

Byte-identical reruns are a hard requirement (docs/01-architecture.md).
``canonical_json`` produces the same bytes for the same logical content:
sorted keys, no whitespace variation, floats via repr round-trip
(shortest exact decimal form), and a hard ban on NaN/Infinity — a result
containing them is a bug we want to surface, not serialize.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _reject_nonfinite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite float in canonical output: {value!r}")
    return value


def _normalize(obj: Any) -> Any:
    """Recursively coerce to canonical JSON-safe python types."""
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return _reject_nonfinite(obj)
    if hasattr(obj, "tolist") and callable(obj.tolist):  # numpy array or scalar
        return _normalize(obj.tolist())
    raise TypeError(f"cannot canonically serialize {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )


def canonical_digest(obj: Any) -> str:
    """SHA-256 hex digest of the canonical serialization."""
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()
