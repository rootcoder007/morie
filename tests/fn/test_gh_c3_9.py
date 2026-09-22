"""Tests for gh_c3_9.ghosal_quantile_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_9 import ghosal_quantile_prior


def test_gh_c3_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_quantile_prior(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c3_9_edge():
    """Test edge cases."""
    result = ghosal_quantile_prior(np.array([42.0]))
    # The function returns a RichResult whose payload contains keys
    # produced from the documented quantile-function construction:
    # "estimate", "u", "Q", "monotone", and "method". There is no
    # "n" key, so check that what the function actually promises is
    # present and consistent.
    assert "estimate" in result
    assert "u" in result
    assert "Q" in result
    assert "monotone" in result
    assert result["method"] == "random quantile-function prior (GvdV 2017 sec. 3.4.5)"
    # Q must be strictly increasing by construction (n_knots values
    # built from strictly positive Gamma increments) and u must be
    # the standard interior grid of size n_knots.
    q = np.asarray(result["Q"], dtype=float)
    assert np.all(np.diff(q) > 0.0)
    assert np.asarray(result["monotone"], dtype=bool)
    # The median of an increasing sequence of length n_knots is at
    # index n_knots // 2; verify the estimate matches that value.
    n_knots = 15
    expected_estimate = float(np.asarray(result["Q"], dtype=float)[n_knots // 2])
    assert float(np.asarray(result["estimate"], dtype=float)) == expected_estimate
