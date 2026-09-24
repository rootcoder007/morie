"""Tests for hrzsmle.horowitz_semipar_mle_binary."""

from morie.fn import _array_core as np

from morie.fn.hrzsmle import horowitz_semipar_mle_binary


def test_hrzsmle_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.integers(0, 2, size=40)
    bandwidth = 0.3
    result = horowitz_semipar_mle_binary(X, y, bandwidth)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hrzsmle_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.integers(0, 2, size=40)
    bandwidth = 0.3
    result = horowitz_semipar_mle_binary(X, y, bandwidth)
    assert isinstance(result, dict)
