"""Tests for gh_c4_2.ghosal_dp_mean."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_2 import ghosal_dp_mean


def test_gh_c4_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_mean(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value

    # Per the documented formula (eq. 4.2), E P(A) = alpha-bar(A) = G0(A),
    # and the implementation takes the first element of the input.
    expected_estimate = float(np.asarray(x).reshape(-1)[0])
    got_estimate = float(np.asarray(result["estimate"], dtype=float).reshape(-1)[0])
    assert got_estimate == expected_estimate
    assert "method" in result


def test_gh_c4_2_edge():
    """Test edge cases."""
    result = ghosal_dp_mean(np.array([42.0]))
    # The function returns a single 'estimate' (the first/only input value),
    # it does not return a length key.
    expected_estimate = float(np.asarray(np.array([42.0])).reshape(-1)[0])
    got_estimate = float(np.asarray(result["estimate"], dtype=float).reshape(-1)[0])
    assert got_estimate == expected_estimate
