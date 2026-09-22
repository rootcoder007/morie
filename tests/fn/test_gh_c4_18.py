"""Tests for gh_c4_18.ghosal_dp_mean_dist."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_18 import ghosal_dp_mean_dist


def test_gh_c4_18_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    s = 3.0
    result = ghosal_dp_mean_dist(x, b, s)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    H = float(np.asarray(result["estimate"], dtype=float))
    assert 0.0 <= H <= 1.0


def test_gh_c4_18_edge():
    """Test edge cases."""
    x = np.array([42.0])
    b = np.array([1.0])
    s = 42.0
    result = ghosal_dp_mean_dist(x, b, s)
    # When s coincides with the only atom, the inner log is real,
    # so the imaginary part vanishes and H = 1/2.
    assert np.isclose(float(np.asarray(result["estimate"], dtype=float)), 0.5, atol=1e-6)
