"""Tests for gh_c4_3.ghosal_dp_var."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_3 import ghosal_dp_var


def test_gh_c4_3_basic():
    """Test basic functionality."""
    g0 = 0.4
    alpha = 3.0
    x = np.array([g0])
    result = ghosal_dp_var(x, alpha)
    assert "estimate" in result
    expected = g0 * (1.0 - g0) / (1.0 + abs(alpha))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(float(result["estimate"]), expected)


def test_gh_c4_3_edge():
    """Test edge cases."""
    g0 = 42.0
    alpha = 0.0
    result = ghosal_dp_var(np.array([g0]), alpha)
    expected = g0 * (1.0 - g0) / (1.0 + abs(alpha))
    assert np.isclose(float(result["estimate"]), expected)
