"""Tests for hypergeometric_pmf.hypergeometric_pmf."""

from morie.fn import _array_core as np

from morie.fn.hypergeometric_pmf import (
    hypergeometric_pmf,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e71_basic():
    """Test basic functionality."""
    k = 5
    N = 5
    K = 5
    n = 5
    result = hypergeometric_pmf(k, N, K, n)
    assert isinstance(result, dict)
    assert "probability" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e71_edge():
    """Test edge cases."""
    k = 5
    N = 5
    K = 5
    n = 5
    result = hypergeometric_pmf(k, N, K, n)
    assert isinstance(result, dict)
