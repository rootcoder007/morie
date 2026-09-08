"""Tests for hypergeometric_pmf.hypergeometric_pmf."""

from morie.fn import _array_core as np

from morie.fn.hypergeometric_pmf import (
    hypergeometric_pmf,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e71_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hypergeometric_pmf(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e71_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hypergeometric_pmf(x)
    assert isinstance(result, dict)
