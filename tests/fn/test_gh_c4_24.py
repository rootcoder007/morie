"""Tests for gh_c4_24.ghosal_bayes_boot."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_24 import ghosal_bayes_boot


def test_gh_c4_24_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bayes_boot(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value

    # Independent expectation: with a single bootstrap draw the
    # Bayesian bootstrap mean functional reduces to a weighted mean
    # of the data with weights W_i = Y_i / sum Y_j, Y_i ~ Exp(1).
    # Across many draws, the average converges to the sample mean.
    n = len(x)
    expected_sample_mean = sum(x) / n
    assert np.all(np.isfinite(np.asarray(result["sample_mean"], dtype=float)))
    assert np.isclose(float(result["sample_mean"]), expected_sample_mean)
    assert np.isclose(float(result["estimate"]), expected_sample_mean, atol=0.5)
    assert "draws_head" in result
    assert "method" in result
    assert isinstance(result["draws_head"], list)
    assert len(result["draws_head"]) == 10


def test_gh_c4_24_edge():
    """Test edge cases: single-observation input."""
    data = np.array([42.0])
    result = ghosal_bayes_boot(data)
    # Documented keys do not include "n"; verify what the function
    # actually returns for a single observation.
    assert "estimate" in result
    # For n=1 there is only one possible weight, so the bootstrap
    # mean functional equals the single observation.
    assert np.isclose(float(result["estimate"]), 42.0)
    assert np.isclose(float(result["sample_mean"]), 42.0)
