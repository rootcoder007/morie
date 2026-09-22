"""Tests for gh_c6_11.ghosal_markov_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_11 import ghosal_markov_con


def test_gh_c6_11_basic():
    """Test basic functionality."""
    a0, b0, n, seed = 0.3, 0.6, 3000, 42
    result = ghosal_markov_con(a0=a0, b0=b0, n=n, seed=seed)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "a_hat" in result
    assert "b_hat" in result
    assert 0.0 < result["a_hat"] < 1.0
    assert 0.0 < result["b_hat"] < 1.0
    assert result["estimate"] >= 0.0


def test_gh_c6_11_edge():
    """Test edge cases."""
    a0, b0, n, seed = 0.3, 0.6, 1, 42
    result = ghosal_markov_con(a0=a0, b0=b0, n=n, seed=seed)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "a_hat" in result
    assert "b_hat" in result
