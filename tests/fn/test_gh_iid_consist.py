"""Tests for gh_iid_consist.ghosal_iid_posterior_consistency."""

from morie.fn import _array_core as np

from morie.fn.gh_iid_consist import ghosal_iid_posterior_consistency


def test_gh_iid_consist_basic():
    """Test basic functionality with documented parameters."""
    result = ghosal_iid_posterior_consistency(
        theta0=0.5, eps=0.2, n=600, seed=42
    )
    assert "estimate" in result
    assert "decay_exponent" in result
    assert "exponential" in result
    assert "method" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray(estimate)))
    assert 0.0 <= estimate <= 1.0
    # Documented: posterior odds of {|theta - theta0| > eps} decay
    # exponentially, so the exponent must be negative.
    exponent = float(np.asarray(result["decay_exponent"], dtype=float))
    assert exponent < 0
    assert result["exponential"] is True


def test_gh_iid_consist_edge():
    """Test edge case with deterministic seed and minimal n."""
    # Deterministic, reproducible input. With seed=42 the first Bernoulli
    # draws against theta0=0.5 give a fixed S, so we can derive expected
    # shapes of the returned payload.
    result = ghosal_iid_posterior_consistency(theta0=0.5, eps=0.2, n=1, seed=42)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert 0.0 <= estimate <= 1.0
    assert np.all(np.isfinite(np.asarray(estimate)))
