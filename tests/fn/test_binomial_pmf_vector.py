"""Tests for binomial_pmf_vector.binomial_pmf_vector."""

import pytest

from morie.fn import _array_core as np

from morie.fn.binomial_pmf_vector import (
    binomial_pmf_vector,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e10_basic():
    """Test basic functionality."""
    n = 10
    p = 0.5
    result = binomial_pmf_vector(n, p)
    assert isinstance(result, dict)
    assert "pmf" in result
    assert "total" in result
    pmf = result["pmf"]
    assert len(pmf) == n + 1
    assert all(0.0 <= float(x) <= 1.0 for x in pmf)
    assert abs(float(result["total"]) - 1.0) < 1e-9


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e10_edge():
    """Test edge cases."""
    n = 1
    p = 0.3
    result = binomial_pmf_vector(n, p)
    assert isinstance(result, dict)
    assert "pmf" in result
    assert "total" in result
    pmf = result["pmf"]
    assert len(pmf) == n + 1
    assert abs(float(result["total"]) - 1.0) < 1e-9
