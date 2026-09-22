"""Tests for gh_c4_23.ghosal_pen_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_23 import ghosal_pen_dp


def test_gh_c4_23_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = x / x.sum()
    alpha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lam = 10.0
    result = ghosal_pen_dp(p, alpha, lam)
    assert "estimate" in result
    expected_pen = sum((p[j + 1] - p[j]) ** 2 for j in range(len(p) - 1))
    expected_log = sum((alpha[j] - 1.0) * np.log(p[j]) for j in range(len(p))) - lam * expected_pen
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert result["estimate"] == expected_log
    assert result["penalty"] == expected_pen


def test_gh_c4_23_edge():
    """Test edge cases."""
    result = ghosal_pen_dp(np.array([1.0]), np.array([1.0]), 1.0)
    assert result["estimate"] == 0.0
