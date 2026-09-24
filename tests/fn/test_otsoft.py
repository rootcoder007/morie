"""Tests for otsoft.ot_softassignment."""

import math

from morie.fn import _array_core as np

from morie.fn.otsoft import ot_softassignment


def test_otsoft_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 10, 12
    a = [float(rng.uniform(0.1, 1.0)) for _ in range(n)]
    b = [float(rng.uniform(0.1, 1.0)) for _ in range(m)]
    C = [[float(rng.normal(0, 1)) for _ in range(m)] for _ in range(n)]
    epsilon = 1.0
    result = ot_softassignment(a, b, C, epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "T" in result
    assert "entropy_mean" in result
    assert "hard" in result
    assert "method" in result
    estimate = result["estimate"]
    assert len(estimate) == n
    assert all(len(row) == m for row in estimate)
    for row in estimate:
        s = sum(row)
        assert abs(s - 1.0) < 1e-6 or abs(s) < 1e-12
    T = result["T"]
    assert len(T) == n
    assert all(len(row) == m for row in T)
    assert math.isfinite(result["entropy_mean"])
    assert len(result["hard"]) == n
    assert isinstance(result["method"], str)


def test_otsoft_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n, m = 5, 7
    a = [float(rng.uniform(0.1, 1.0)) for _ in range(n)]
    b = [float(rng.uniform(0.1, 1.0)) for _ in range(m)]
    C = [[float(rng.normal(0, 1)) for _ in range(m)] for _ in range(n)]
    result = ot_softassignment(a, b, C, 5.0, max_iter=50)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "T" in result
    assert "entropy_mean" in result
    assert "hard" in result
    assert "method" in result
    estimate = result["estimate"]
    assert len(estimate) == n
    assert all(len(row) == m for row in estimate)
    for row in estimate:
        s = sum(row)
        assert abs(s - 1.0) < 1e-6 or abs(s) < 1e-12
    assert math.isfinite(result["entropy_mean"])
    assert len(result["hard"]) == n
    assert isinstance(result["method"], str)
