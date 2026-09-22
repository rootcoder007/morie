"""Tests for gb2101.gibbons_asymp_order_normal."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gb2101 import gibbons_asymp_order_normal


def test_gb2101_basic():
    """Test basic functionality against the documented formula."""
    p = 0.25
    n = 100
    xp = 1.2
    fxp = 0.4
    result = gibbons_asymp_order_normal(p, n, xp, fxp)

    # Documented return keys
    assert "mean" in result
    assert "var" in result
    assert "se" in result
    assert "p" in result
    assert "n" in result
    assert "method" in result

    # Mean is the population quantile
    assert result["mean"] == xp

    # p and n echoed back
    assert result["p"] == p
    assert result["n"] == n

    # var = p(1-p) / (n * fxp^2) and se = sqrt(var), computed independently
    expected_var = p * (1.0 - p) / (n * fxp * fxp)
    expected_se = (expected_var) ** 0.5
    assert result["var"] == expected_var
    assert result["se"] == expected_se


def test_gb2101_edge():
    """Test edge cases with different but valid inputs."""
    p = 0.9
    n = 50
    xp = -0.5
    fxp = 2.0
    result = gibbons_asymp_order_normal(p, n, xp, fxp)

    assert isinstance(result, dict)
    assert result["mean"] == xp
    assert result["p"] == p
    assert result["n"] == n

    expected_var = p * (1.0 - p) / (n * fxp * fxp)
    expected_se = (expected_var) ** 0.5
    assert result["var"] == expected_var
    assert result["se"] == expected_se
