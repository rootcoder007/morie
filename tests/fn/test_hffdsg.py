"""Tests for hffdsg.hoeffding_inequality."""

import math

from morie.fn.hffdsg import hoeffding_inequality


def test_hffdsg_basic():
    """Test basic functionality."""
    a = 0.0
    b = 1.0
    n = 100
    t = 0.1
    result = hoeffding_inequality(a, b, n, t)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "bound" in result
    assert "one_sided" in result
    assert "informative" in result
    assert "t_min" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert math.isfinite(result["bound"])
    assert math.isfinite(result["one_sided"])
    assert math.isfinite(result["t_min"])
    assert result["t_min"] >= 0.0


def test_hffdsg_edge():
    """Test edge cases."""
    # t = 0: bound = 2*exp(0) = 2.0, so estimate = min(2.0, 1.0) = 1.0
    a = -1.0
    b = 1.0
    n = 10
    t = 0.0
    result = hoeffding_inequality(a, b, n, t)
    assert isinstance(result, dict)
    assert result["estimate"] == 1.0
    assert result["bound"] == 2.0
    assert result["one_sided"] == 1.0
    assert result["informative"] == 0
    assert result["n"] == 10
