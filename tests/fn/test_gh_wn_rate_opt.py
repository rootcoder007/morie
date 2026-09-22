"""Tests for gh_wn_rate_opt.ghosal_white_noise_optimal_rate."""

from morie.fn import _array_core as np

from morie.fn.gh_wn_rate_opt import ghosal_white_noise_optimal_rate


def test_gh_wn_rate_opt_basic():
    """Test basic functionality."""
    s = 1.5
    n = 1000.0
    result = ghosal_white_noise_optimal_rate(s, n)
    assert "estimate" in result
    expected = float(n) ** (-2.0 * float(s) / (2.0 * float(s) + 1.0))
    assert np.allclose(np.asarray(result["estimate"], dtype=float), expected)
    assert "exponent" in result
    assert np.isclose(result["exponent"], 2.0 * float(s) / (2.0 * float(s) + 1.0))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_wn_rate_opt_edge():
    """Test edge cases."""
    s = 1.0
    n = 1.0
    result = ghosal_white_noise_optimal_rate(np.array([s]), np.array([n]))
    expected = float(n) ** (-2.0 * float(s) / (2.0 * float(s) + 1.0))
    assert np.allclose(np.asarray(result["estimate"], dtype=float), expected)
    assert result["attained_by_alpha_eq_s"] is True
    assert result["method"] == "white-noise minimax rate (GvdV 2017 sec. 8.3.4)"
