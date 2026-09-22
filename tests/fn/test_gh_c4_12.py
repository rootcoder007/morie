"""Tests for gh_c4_12.ghosal_dp_discrete."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_12 import ghosal_dp_discrete


def test_gh_c4_12_basic():
    """Test basic functionality."""
    n_terms = 5
    alpha = 1.0
    seed = 42
    result = ghosal_dp_discrete(n_terms, alpha, seed=seed)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    assert 0.0 < float(estimate) <= 1.0


def test_gh_c4_12_edge():
    """Test edge cases."""
    n_terms = 1
    alpha = 42.0
    result = ghosal_dp_discrete(n_terms, alpha)
    assert "estimate" in result
    assert 0.0 < float(result["estimate"]) <= 1.0
