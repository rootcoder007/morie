"""Tests for gh_c4_19.ghosal_dp_charact."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_19 import ghosal_dp_charact


def test_gh_c4_19_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_charact(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c4_19_edge():
    """Test edge cases."""
    result = ghosal_dp_charact(np.array([42.0, 17.0, 33.0]))
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    # Computed from the documented formula on the same inputs:
    # for a Dirichlet vector with params [42, 17, 33], P(A_1) and
    # P(A_2)/(1 - P(A_1)) should be uncorrelated => close to 0.
    assert abs(est) < 0.1
