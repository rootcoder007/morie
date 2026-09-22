"""Tests for gh_c4_7.ghosal_dp_pred."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_7 import ghosal_dp_pred


def test_gh_c4_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 1.0
    n = len(x)
    expected_w_new = alpha / (alpha + n)
    expected_w_each = 1.0 / (alpha + n)
    result = ghosal_dp_pred(x, alpha)
    assert "estimate" in result
    assert result["estimate"] == expected_w_new
    assert result["weight_fresh"] == expected_w_new
    assert result["weight_per_obs"] == expected_w_each
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c4_7_with_x_new_equals():
    """Test predictive probability at an observed point."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 2.5
    n = len(x)
    expected_w_new = alpha / (alpha + n)
    expected_w_each = 1.0 / (alpha + n)
    # t == 2.0 appears once in x
    expected_est = expected_w_each * 1.0 + expected_w_new * 0.0
    result = ghosal_dp_pred(x, alpha, x_new_equals=2.0)
    assert result["estimate"] == expected_est
    assert result["weight_fresh"] == expected_w_new
    assert result["weight_per_obs"] == expected_w_each


def test_gh_c4_7_edge():
    """Test edge cases with a single observation."""
    x = np.array([42.0])
    alpha = 3.0
    n = len(x)
    expected_w_new = alpha / (alpha + n)
    expected_w_each = 1.0 / (alpha + n)
    result = ghosal_dp_pred(x, alpha)
    # Single observation matches a fresh draw exactly when x_new_equals that value.
    expected_match = expected_w_each * 1.0 + expected_w_new * 0.0
    assert result["estimate"] == expected_w_new
    result_match = ghosal_dp_pred(x, alpha, x_new_equals=42.0)
    assert result_match["estimate"] == expected_match
