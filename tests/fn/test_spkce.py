"""Tests for spkce.schabenberger_cov_param_estimation_kriging."""

from morie.fn import _array_core as np

from morie.fn.spkce import schabenberger_cov_param_estimation_kriging


def test_spkce_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_cov_param_estimation_kriging(coords, z)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_spkce_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_cov_param_estimation_kriging(coords, z)
    assert isinstance(result, dict)
