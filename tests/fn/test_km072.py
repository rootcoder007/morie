"""Tests for km072.kamath_ch5_bradley_terry_pref."""

import math

from morie.fn.km072 import kamath_ch5_bradley_terry_pref


def test_km072_basic():
    """Test basic functionality with a reward mapping."""
    r_star = {"a": 1.0, "b": 0.0}
    result = kamath_ch5_bradley_terry_pref(r_star, "a", "b")
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "margin" in result
    assert "r_w" in result
    assert "r_l" in result
    assert "p_reversed" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert math.isfinite(result["margin"])
    assert result["n"] == 2
    assert result["p_reversed"] == 1.0 - result["estimate"]


def test_km072_edge():
    """Test edge case with a callable r_star and equal rewards."""
    rewards = {"a": 5.0, "b": 5.0}
    r_star = lambda label: rewards[label]
    result = kamath_ch5_bradley_terry_pref(r_star, "a", "b")
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "r_w" in result
    assert "r_l" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert math.isfinite(result["margin"])
    assert math.isclose(result["margin"], 0.0, abs_tol=1e-12)
