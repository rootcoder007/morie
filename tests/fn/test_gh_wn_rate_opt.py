"""Tests for gh_wn_rate_opt.ghosal_white_noise_optimal_rate."""

from morie.fn import _array_core as np

from morie.fn.gh_wn_rate_opt import ghosal_white_noise_optimal_rate


def test_gh_wn_rate_opt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_white_noise_optimal_rate(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_wn_rate_opt_edge():
    """Test edge cases."""
    result = ghosal_white_noise_optimal_rate(np.array([42.0]))
    assert result["n"] == 1
