"""Tests for otentf.ot_free_energy."""

import math

from morie.fn import _array_core as np

from morie.fn.otentf import ot_free_energy


def test_otentf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 4, 5
    T = np.abs(rng.normal(0, 1, (n, m)))
    C = np.abs(rng.normal(0, 1, (n, m)))
    a = np.abs(rng.normal(0, 1, n))
    b = np.abs(rng.normal(0, 1, m))
    f = rng.normal(0, 1, n)
    g = rng.normal(0, 1, m)
    epsilon = 0.1
    result = ot_free_energy(T, C, a, b, f, g, epsilon)
    assert "estimate" in result
    assert "primal" in result
    assert "dual_pairing" in result
    assert "entropy" in result
    assert "epsilon" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["primal"])
    assert math.isfinite(result["entropy"])


def test_otentf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, m = 2, 3
    T = np.abs(rng.normal(0, 1, (n, m)))
    C = np.abs(rng.normal(0, 1, (n, m)))
    a = np.abs(rng.normal(0, 1, n))
    b = np.abs(rng.normal(0, 1, m))
    f = rng.normal(0, 1, n)
    g = rng.normal(0, 1, m)
    epsilon = 1e-6
    result = ot_free_energy(T, C, a, b, f, g, epsilon)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["primal"])
