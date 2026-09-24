"""Tests for depthP.projection_depth."""

import math

from morie.fn import _array_core as np

from morie.fn.depthP import projection_depth


def test_depthP_basic():
    """Test basic functionality with d=2."""
    rng = np.random.default_rng(42)
    x = [1.5, 0.5]
    X = rng.normal(0, 1, (40, 2))
    result = projection_depth(x, X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "depth" in result
    assert "outlyingness" in result
    assert "med" in result
    assert "mad" in result
    assert "worst_dir" in result
    assert result["n"] == 40
    assert result["d"] == 2
    assert math.isfinite(result["depth"])
    assert 0 < result["depth"] <= 1


def test_depthP_edge():
    """Test edge case with d=1 closed-form path."""
    rng = np.random.default_rng(42)
    x = [1.0]
    X = rng.normal(0, 1, (40, 1))
    result = projection_depth(x, X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == 40
    assert result["d"] == 1
    assert math.isfinite(result["depth"])
    assert 0 < result["depth"] <= 1
