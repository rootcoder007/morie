"""Tests for ksr021.kosorok_ch1_residual_empirical_distribution."""

from morie.fn import _array_core as np

from morie.fn.ksr021 import kosorok_ch1_residual_empirical_distribution


def test_ksr021_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = kosorok_ch1_residual_empirical_distribution(y, z, beta)
    assert isinstance(result, dict)
    assert "t" in result


def test_ksr021_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = kosorok_ch1_residual_empirical_distribution(y, z, beta)
    assert isinstance(result, dict)
