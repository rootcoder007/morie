"""Tests for km073.kamath_ch5_pref_sigmoid_form."""

import math

from morie.fn import _array_core as np

from morie.fn.km073 import kamath_ch5_pref_sigmoid_form


def test_km073_basic():
    """Test basic functionality with a winner/loser reward pair."""
    rng = np.random.default_rng(42)
    r_star = rng.normal(0, 1, 2).tolist()
    result = kamath_ch5_pref_sigmoid_form(r_star)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "margin" in result
    assert "r_w" in result
    assert "r_l" in result
    assert "n" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert result["n"] == 2


def test_km073_edge():
    """Test edge case: equal rewards must yield 0.5 exactly."""
    result = kamath_ch5_pref_sigmoid_form([0.0, 0.0])
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["estimate"] == 0.5
