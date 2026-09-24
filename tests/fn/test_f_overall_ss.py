"""Tests for f_overall_ss.f_overall_ss."""

import math

from morie.fn.f_overall_ss import f_overall_ss


def test_ca2e16_basic():
    """Test basic functionality."""
    ss_model = 30.0
    ss_resid = 100.0
    n = 40
    k = 3
    result = f_overall_ss(ss_model, ss_resid, n, k)
    assert isinstance(result, dict)
    assert "f" in result
    assert math.isfinite(result["f"])
    assert result["f"] > 0


def test_ca2e16_edge():
    """Test edge cases."""
    ss_model = 5.0
    ss_resid = 50.0
    n = 5
    k = 2
    result = f_overall_ss(ss_model, ss_resid, n, k)
    assert isinstance(result, dict)
    assert "f" in result
    assert math.isfinite(result["f"])
    assert result["f"] > 0
