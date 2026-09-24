"""Tests for km053.kamath_ch3_prefix_tuning_obj."""

from morie.fn import _array_core as np
import math

from morie.fn.km053 import kamath_ch3_prefix_tuning_obj


def test_km053_basic():
    """Test basic functionality with all target positions scored."""
    rng = np.random.default_rng(42)

    def phi(z, hp):
        return 0.5

    x = "summarise:"
    y = ["a", "b", "c", "d"]
    h = rng.normal(0, 1, len(y))
    result = kamath_ch3_prefix_tuning_obj(phi, x, y, h)
    assert "estimate" in result
    assert "per_position" in result
    assert "positions_scored" in result
    assert "n" in result
    expected = len(y) * math.log(0.5)
    assert abs(result["estimate"] - expected) < 1e-12
    assert len(result["per_position"]) == len(y)
    assert result["n"] == len(y)
    assert math.isfinite(result["estimate"])


def test_km053_edge():
    """Test edge case with a subset of positions in Y_idx."""
    rng = np.random.default_rng(42)

    def phi(z, hp):
        return 0.25

    x = "translate:"
    y = ["a", "b", "c", "d", "e"]
    h = rng.normal(0, 1, len(y))
    Y_idx = [0, 2, 4]
    result = kamath_ch3_prefix_tuning_obj(phi, x, y, h, Y_idx=Y_idx)
    assert "estimate" in result
    assert "per_position" in result
    assert "positions_scored" in result
    expected = 3 * math.log(0.25)
    assert abs(result["estimate"] - expected) < 1e-12
    assert result["positions_scored"] == Y_idx
    assert len(result["per_position"]) == len(Y_idx)
    assert result["n"] == len(y)
    assert math.isfinite(result["estimate"])
