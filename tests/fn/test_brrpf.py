"""Tests for brrpf.brr_prior_posterior."""

from morie.fn import _array_core as np

from morie.fn.brrpf import brr_prior_posterior


def test_brrpf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = brr_prior_posterior(y)
    assert isinstance(result, dict)
    assert "S" in result
def test_brrpf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = brr_prior_posterior(y)
    assert isinstance(result, dict)
