"""Tests for gh_c4_8.ghosal_dp_ndist."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_8 import ghosal_dp_ndist


def test_gh_c4_8_basic():
    """Test basic functionality."""
    n = 5
    alpha = 3.0
    result = ghosal_dp_ndist(n, alpha)
    assert "estimate" in result
    expected_mean = sum(alpha / (alpha + i - 1.0) for i in range(1, n + 1))
    assert abs(result["estimate"] - expected_mean) < 1e-12
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c4_8_edge():
    """Test edge cases."""
    result = ghosal_dp_ndist(42, 10.0)
    n = 42
    alpha = 10.0
    expected_var = sum(alpha * (i - 1.0) / (alpha + i - 1.0) ** 2
                       for i in range(1, n + 1))
    assert abs(result["variance"] - expected_var) < 1e-10
    assert result["lower_bound"] <= result["estimate"] <= result["upper_bound"]
