"""Tests for gh_pt_adapt.ghosal_pt_adaptive."""

from morie.fn import _array_core as np

from morie.fn.gh_pt_adapt import ghosal_pt_adaptive


def test_gh_pt_adapt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pt_adaptive(x)
    assert result["n"] == 5
    assert np.all(np.isfinite(np.asarray(result["rate"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["minimax_rate"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["log_factor"], dtype=float)))


def test_gh_pt_adapt_edge():
    """Test edge cases."""
    import math
    x = np.array([42.0, 7.0])
    result = ghosal_pt_adaptive(x, s=1.0)
    n = result["n"]
    s = result["smoothness"]
    expected_rate = (n ** (-s / (2.0 * s + 1.0))) * math.log(n)
    assert result["rate"] == expected_rate
    assert result["minimax_rate"] == n ** (-s / (2.0 * s + 1.0))
    assert result["log_factor"] == math.log(n)
