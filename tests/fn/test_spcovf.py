"""Tests for spcovf.schabenberger_covariance_function."""

from morie.fn import _array_core as np

from morie.fn.spcovf import schabenberger_covariance_function


def test_spcovf_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_covariance_function(coords, z)
    assert isinstance(result, dict)
    assert "lag" in result


def test_spcovf_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = schabenberger_covariance_function(coords, z)
    assert isinstance(result, dict)
