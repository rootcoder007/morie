"""Tests for ca2e3.ca_chapter_2_equation_3."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.ca2e3 import ca_chapter_2_equation_3


def test_ca2e3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = ca_chapter_2_equation_3(x, y)
    assert isinstance(result, dict)
    assert "b0" in result
    # Independent computation of the OLS intercept from the documented formula:
    # b0 = ybar - b1 * xbar, where b1 = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    dx = x - xbar
    dy = y - ybar
    b1 = float(np.sum(dx * dy) / np.sum(dx * dx))
    expected_b0 = ybar - b1 * xbar
    actual_b0 = float(result["b0"])
    assert abs(actual_b0 - expected_b0) < 1e-10
    assert result.get("method") == "Weisburd et al. (2022) eq. (2.3)"


def test_ca2e3_edge():
    """Test edge cases: centred (zero-mean) data."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = ca_chapter_2_equation_3(x, y)
    assert isinstance(result, dict)
    assert "b0" in result
    # Sanity: independent OLS slope/intercept from the same formula
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    dx = x - xbar
    dy = y - ybar
    b1 = float(np.sum(dx * dy) / np.sum(dx * dx))
    expected_b0 = ybar - b1 * xbar
    assert abs(float(result["b0"]) - expected_b0) < 1e-10
