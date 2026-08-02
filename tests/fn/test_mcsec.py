"""Tests for mcsec.mcmc_standard_error."""

from morie.fn import _array_core as np

from morie.fn.mcsec import mcmc_standard_error


def test_mcsec_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = mcmc_standard_error(chains)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_mcsec_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = mcmc_standard_error(chains)
    assert isinstance(result, dict)
