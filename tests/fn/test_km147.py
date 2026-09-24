"""Tests for km147.kamath_ch9_output_alignment."""

from morie.fn import _array_core as np
import math

from morie.fn.km147 import kamath_ch9_output_alignment


def test_km147_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    S_X = rng.normal(0, 1, (40, 3))
    W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = kamath_ch9_output_alignment(S_X, W)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "features" in result
    assert "shape" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 40
    assert result["shape"] == [40, 2]
    assert len(result["features"]) == 40
    assert len(result["features"][0]) == 2
    for row in result["features"]:
        for v in row:
            assert math.isfinite(v)


def test_km147_edge():
    """Test edge cases."""
    S_X = [[1.0, 2.0]]
    W = [[0.0], [2.0]]
    result = kamath_ch9_output_alignment(S_X, W)
    assert isinstance(result, dict)
    assert result["features"] == [[4.0]]
    assert result["shape"] == [1, 1]
    assert result["n"] == 1
    assert math.isfinite(result["estimate"])
