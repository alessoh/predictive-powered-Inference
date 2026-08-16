"""Canonical serialization tests."""

import math

import numpy as np
import pytest

from ppi_core.serialize import canonical_digest, canonical_json


def test_key_order_is_canonical():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_floats_round_trip_exactly():
    x = 0.1 + 0.2
    s = canonical_json({"v": x})
    import json

    assert json.loads(s)["v"] == x


def test_numpy_types_normalized():
    obj = {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, 2.0])}
    assert canonical_json(obj) == '{"a":1.5,"b":3,"c":[1.0,2.0]}'


def test_nan_rejected():
    with pytest.raises(ValueError):
        canonical_json({"v": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"v": math.inf})


def test_unknown_type_rejected():
    with pytest.raises(TypeError):
        canonical_json({"v": object()})


def test_digest_stable():
    obj = {"estimate": 1.25, "list": [1, 2, 3]}
    assert canonical_digest(obj) == canonical_digest({"list": [1, 2, 3], "estimate": 1.25})
