"""Tests for km075.kamath_ch5_dpo_pref_simplified."""

import math

from morie.fn import _array_core as np

from morie.fn.km075 import kamath_ch5_dpo_pref_simplified


def test_km075_basic():
    """Test basic functionality."""
    pi_star = [0.75, 0.25]
    pi_ref = [0.5, 0.5]
    beta = 1.0
    result = kamath_ch5_dpo_pref_simplified(pi_star, pi_ref, beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert "implicit_reward_w" in result
    assert "implicit_reward_l" in result
    assert math.isfinite(result["implicit_reward_w"])
    assert math.isfinite(result["implicit_reward_l"])


def test_km075_edge():
    """Test edge cases."""
    pi_star = [0.5, 0.5]
    pi_ref = [0.5, 0.5]
    beta = 2.0
    result = kamath_ch5_dpo_pref_simplified(pi_star, pi_ref, beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert "margin" in result
    assert math.isfinite(result["margin"])
