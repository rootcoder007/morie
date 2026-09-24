"""Tests for retnet.retnet_retention."""

import math

from morie.fn import _array_core as np
from morie.fn.retnet import retnet_retention


def test_retnet_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    dk = 4
    dv = 3
    Q = rng.normal(0, 1, (n, dk))
    K = rng.normal(0, 1, (n, dk))
    V = rng.normal(0, 1, (n, dv))
    gamma = 0.9
    result = retnet_retention(Q, K=K, V=V, gamma=gamma)

    payload = result.payload

    assert "out" in payload
    assert "out_par" in payload
    assert "max_gap" in payload
    assert "estimate" in payload
    assert "state" in payload

    # Check output shapes
    assert len(payload["out"]) == n
    assert len(payload["out_par"]) == n
    for row in payload["out"]:
        assert len(row) == dv
    for row in payload["out_par"]:
        assert len(row) == dv

    # max_gap is the max absolute difference between recurrent and parallel forms
    assert math.isfinite(payload["max_gap"])
    assert payload["max_gap"] >= 0

    # estimate is a single number taken from the first output row
    assert math.isfinite(payload["estimate"])


def test_retnet_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 1
    dk = 2
    dv = 2
    Q = rng.normal(0, 1, (n, dk))
    K = rng.normal(0, 1, (n, dk))
    V = rng.normal(0, 1, (n, dv))
    gamma = 0.5
    result = retnet_retention(Q, K=K, V=V, gamma=gamma)

    payload = result.payload

    assert "out" in payload
    assert "out_par" in payload
    assert "max_gap" in payload
    assert "estimate" in payload

    assert len(payload["out"]) == n
    assert len(payload["out_par"]) == n
    for row in payload["out"]:
        assert len(row) == dv
    for row in payload["out_par"]:
        assert len(row) == dv
    assert math.isfinite(payload["max_gap"])
    assert math.isfinite(payload["estimate"])
