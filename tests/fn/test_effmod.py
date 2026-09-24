"""Tests for effmod.effect_modification."""

import math

from morie.fn import _array_core as np

from morie.fn.effmod import effect_modification


def test_effmod_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    y = list(rng.integers(0, 2, n))
    A = list(rng.integers(0, 2, n))
    V = list(rng.integers(0, 2, n))
    H = rng.normal(0, 1, (n, 3))
    result = effect_modification(y, A, V, H)
    expected_keys = {"estimate", "p00", "p10", "p01", "p11", "rr10", "rr01",
                     "rr11", "reri", "mult", "rd_int", "ap", "n00", "n10",
                     "n01", "n11", "n"}
    for key in expected_keys:
        assert key in result
    assert math.isfinite(result["estimate"])


def test_effmod_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    y = list(rng.integers(0, 2, n))
    A = list(rng.integers(0, 2, n))
    V = list(rng.integers(0, 2, n))
    result = effect_modification(y, A, V)
    expected_keys = {"estimate", "p00", "p10", "p01", "p11", "rr10", "rr01",
                     "rr11", "reri", "mult", "rd_int", "ap", "n00", "n10",
                     "n01", "n11", "n"}
    for key in expected_keys:
        assert key in result
    assert result["n00"] + result["n10"] + result["n01"] + result["n11"] == result["n"]
