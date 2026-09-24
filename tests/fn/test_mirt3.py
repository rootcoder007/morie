"""Tests for mirt3.mirt_3d_compensatory."""

import math

from morie.fn import _array_core as np

from morie.fn.mirt3 import mirt_3d_compensatory


def test_mirt3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 40
    y = rng.integers(0, 2, n)
    theta = rng.normal(0, 1, (n, 3))
    a = rng.normal(0, 1, 3)
    d = 0.5
    result = mirt_3d_compensatory(y, theta, a, d)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loglik" in result
    assert "pbar" in result
    assert "n" in result
    assert "m" in result
    assert math.isfinite(result["loglik"])
    assert 0.0 <= result["pbar"] <= 1.0
    assert len(result["p"]) == n
    assert result["n"] == n
    assert int(result["m"]) == result["m"] and result["m"] > 0


def test_mirt3_edge():
    """Test edge cases with custom D and a small sample."""
    rng = np.random.default_rng(44)
    n = 10
    y = rng.integers(0, 2, n)
    theta = rng.normal(0, 1, (n, 3))
    a = [0.8, 1.0, 1.2]
    d = -0.2
    result = mirt_3d_compensatory(y, theta, a, d, D=1.7)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "deviance" in result
    assert math.isfinite(result["loglik"])
    assert math.isfinite(result["deviance"])
    assert 0.0 <= result["pbar"] <= 1.0
    assert len(result["p"]) == n
