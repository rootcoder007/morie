"""Tests for ldiff.l_diversity_check."""

from morie.fn import _array_core as np

from morie.fn.ldiff import l_diversity_check


def test_ldiff_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    quasi_ids = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sensitive = np.random.default_rng(42).normal(0.0, 1.0, 40)
    l = 5
    result = l_diversity_check(X, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ldiff_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    quasi_ids = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sensitive = np.random.default_rng(42).normal(0.0, 1.0, 40)
    l = 5
    result = l_diversity_check(X, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
