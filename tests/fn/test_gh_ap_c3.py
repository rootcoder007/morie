"""Tests for gh_ap_c3.ghosal_bracket_num."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_c3 import ghosal_bracket_num


def test_gh_ap_c3_basic():
    """Test basic functionality."""
    smoothness = 1.0
    eps = 0.1
    result = ghosal_bracket_num(smoothness=smoothness, eps=eps)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    # Formula: log N_[](eps, T, d) ~ eps^{-1/smoothness}
    expected = eps ** (-1.0 / smoothness)
    assert np.isclose(float(estimate), expected)


def test_gh_ap_c3_edge():
    """Test edge cases."""
    smoothness = 0.5
    eps = 2.0
    result = ghosal_bracket_num(smoothness=smoothness, eps=eps)
    assert "estimate" in result
    assert "entropy_exponent" in result
    assert "method" in result
    # Formula: log N_[](eps, T, d) ~ eps^{-1/smoothness}
    expected = eps ** (-1.0 / smoothness)
    assert np.isclose(float(np.asarray(result["estimate"], dtype=float)), expected)
    assert np.isclose(float(result["entropy_exponent"]), 1.0 / smoothness)
