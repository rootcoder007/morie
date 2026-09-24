"""Tests for km074.kamath_ch5_dpo_pref_substituted."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km074 import kamath_ch5_dpo_pref_substituted


def test_km074_basic():
    """Test basic functionality with two probabilities and a positive Z."""
    pi_star = [0.75, 0.25]
    pi_ref = [0.5, 0.5]
    beta = 1.0
    Z = 1000.0
    result = kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z=Z)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert result["z_terms_cancel"] is True
    for key in ("estimate", "margin", "z_offset", "z_terms_cancel",
                "simplified", "beta", "Z", "n", "method"):
        assert key in result


def test_km074_edge():
    """Test edge cases: default Z=None and invalid Z raise."""
    pi_star = [0.6, 0.4]
    pi_ref = [0.5, 0.5]
    beta = 0.5
    # Z defaults to None -> valid input
    result = kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert result["z_terms_cancel"] is True
    # A large Z should still yield a finite probability in [0, 1]
    result2 = kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z=1e6)
    assert math.isfinite(result2["estimate"])
    assert 0.0 <= result2["estimate"] <= 1.0
    # Z must be strictly positive per the docstring
    with pytest.raises(ValueError):
        kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z=0.0)
    with pytest.raises(ValueError):
        kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z=-1.0)
