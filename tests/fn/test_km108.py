"""Tests for km108.kamath_ch6_differential_privacy."""

import math

from morie.fn import _array_core as np

from morie.fn.km108 import kamath_ch6_differential_privacy


def _make_M():
    dist_A = {"o1": 0.6, "o2": 0.4}
    dist_B = {"o1": 0.3, "o2": 0.7}

    def M(D):
        return dist_A if D == "A" else dist_B

    return M


def test_km108_basic():
    """Test basic functionality."""
    M = _make_M()
    A = "A"
    B = "B"
    S = ["o1"]
    epsilon = 1.0
    result = kamath_ch6_differential_privacy(M, A, B, S, epsilon)
    assert isinstance(result, dict)
    for key in ("estimate", "epsilon_required", "satisfied",
                "p_A", "p_B", "ratio", "epsilon", "bound", "n", "method"):
        assert key in result
    assert result["satisfied"] is True
    assert abs(result["epsilon_required"] - math.log(2)) < 1e-12
    assert abs(result["p_A"] - 0.6) < 1e-12
    assert abs(result["p_B"] - 0.3) < 1e-12


def test_km108_edge():
    """Test edge cases."""
    M = _make_M()
    A = "A"
    B = "B"
    S = ["o1"]
    epsilon = 0.5
    result = kamath_ch6_differential_privacy(M, A, B, S, epsilon)
    assert isinstance(result, dict)
    assert "satisfied" in result
    assert result["satisfied"] is False
    assert math.isfinite(result["epsilon_required"])
    assert result["epsilon_required"] > 0.5
