"""Tests for gh_c12_9.ghosal_wn_full_bvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_9 import ghosal_wn_full_bvm


def test_gh_c12_9_basic():
    """Test basic functionality."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wn_full_bvm(y)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent computation of expected estimate (max gap between
    # posterior mean and observed Y for the documented conjugate formula).
    n = 400.0
    prior_var = 1e6
    expected_means = [prior_var / (prior_var + 1.0 / n) * v for v in y]
    expected_gap = max(abs(m - v) for m, v in zip(expected_means, y))
    assert np.isclose(result["estimate"], expected_gap)


def test_gh_c12_9_edge():
    """Test edge case with a single observation."""
    y = np.array([42.0])
    result = ghosal_wn_full_bvm(y)
    assert "estimate" in result

    # The documented keys returned by the function.
    assert "mean_matches_Y" in result
    assert "var_matches_In" in result
    assert "method" in result

    n = 400.0
    prior_var = 1e6
    expected_mean = prior_var / (prior_var + 1.0 / n) * 42.0
    expected_gap = abs(expected_mean - 42.0)
    assert np.isclose(result["estimate"], expected_gap)

    # With prior_var = 1e6 and n = 400 the posterior mean is
    # essentially indistinguishable from Y itself.
    assert result["mean_matches_Y"] is True
