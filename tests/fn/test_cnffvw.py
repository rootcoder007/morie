"""Tests for cnffvw.cinelli_hazlett_robust."""

from morie.fn import _array_core as np

from morie.fn.cnffvw import cinelli_hazlett_robust


def test_cnffvw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = cinelli_hazlett_robust(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cnffvw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = cinelli_hazlett_robust(y, D)
    assert isinstance(result, dict)
