"""Tests for otunbal.ot_unbalanced."""

import math

from morie.fn import _array_core as np
from morie.fn.otunbal import ot_unbalanced


def test_otunbal_basic():
    """Test basic functionality."""
    rng_a = np.random.default_rng(44)
    rng_b = np.random.default_rng(42)
    rng_c = np.random.default_rng(43)
    n, m = 5, 6
    a = rng_a.uniform(0, 1, n)
    b = rng_b.uniform(0, 1, m)
    C = rng_c.uniform(0, 1, (n, m))
    epsilon = 0.1
    lam = 0.1
    result = ot_unbalanced(a, b, C, epsilon, lam)
    assert isinstance(result, dict)
    assert "T" in result
    assert "cost" in result
    assert math.isfinite(result["cost"])
    assert "n" in result
    assert "m" in result


def test_otunbal_edge():
    """Test edge cases."""
    rng_a = np.random.default_rng(44)
    rng_b = np.random.default_rng(42)
    rng_c = np.random.default_rng(43)
    n, m = 3, 4
    a = rng_a.uniform(0, 1, n)
    b = rng_b.uniform(0, 1, m)
    C = rng_c.uniform(0, 1, (n, m))
    epsilon = 0.1
    lam = 0.5
    result = ot_unbalanced(a, b, C, epsilon, lam)
    assert isinstance(result, dict)
    assert "T" in result
    assert "mass" in result
    assert "mass_a" in result
    assert "mass_b" in result
    assert result["n"] == n
    assert result["m"] == m
