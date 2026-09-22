"""Tests for gh_ap_e2.ghosal_spline_space."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_e2 import ghosal_spline_space


def test_gh_ap_e2_basic():
    """Test basic functionality with documented keyword arguments."""
    K = 10
    order = 4
    smoothness = 2.0
    result = ghosal_spline_space(K=K, order=order, smoothness=smoothness)
    assert "estimate" in result
    # The documented formula is dim(S_{K,r}) = K + r (knots + order).
    expected_estimate = float(K) + float(order)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert float(result["estimate"]) == expected_estimate
    # Approximation error: ||f - s*|| ~ K^{-s}
    expected_apx = float(K) ** (-smoothness)
    assert float(result["approx_error_order"]) == expected_apx
    assert result["method"] == "spline space (GvdV 2017 App E)"


def test_gh_ap_e2_edge():
    """Test edge cases with a single-element array passed as K."""
    K = np.array([42.0])
    order = 4
    result = ghosal_spline_space(K=K, order=order)
    expected_estimate = float(K) + float(order)
    assert float(result["estimate"]) == expected_estimate
