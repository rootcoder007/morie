"""Tests for otemd.ot_emd_solver."""

import math

from morie.fn import _array_core as np

from morie.fn.otemd import ot_emd_solver


def test_otemd_basic():
    """Test basic functionality."""
    n, m = 5, 4
    rng_a = np.random.default_rng(44)
    rng_b = np.random.default_rng(45)
    rng_C = np.random.default_rng(42)
    # Generate non-negative histograms with equal total mass.
    a = rng_a.uniform(0.001, 1, n)
    sum_a = np.sum(a)
    b = rng_b.uniform(0.001, 1, m)
    sum_b = np.sum(b)
    # Scale b to match the total mass of a.
    b = [x * (sum_a / sum_b) for x in b]
    C = rng_C.uniform(0, 1, (n, m))
    result = ot_emd_solver(a, b, C)
    assert isinstance(result, dict)
    assert "T" in result
    assert "cost" in result
    assert "n" in result
    assert "m" in result
    assert "n_basic" in result
    assert result["n"] == n
    assert result["m"] == m
    assert math.isfinite(result["cost"])
    T = result["T"]
    assert len(T) == n
    assert all(len(row) == m for row in T)


def test_otemd_edge():
    """Test edge cases."""
    n, m = 3, 3
    rng_a = np.random.default_rng(46)
    rng_b = np.random.default_rng(47)
    rng_C = np.random.default_rng(48)
    a = rng_a.uniform(0.001, 1, n)
    sum_a = np.sum(a)
    b = rng_b.uniform(0.001, 1, m)
    sum_b = np.sum(b)
    b = [x * (sum_a / sum_b) for x in b]
    C = rng_C.uniform(0, 1, (n, m))
    result = ot_emd_solver(a, b, C)
    assert isinstance(result, dict)
    assert "T" in result
    assert "cost" in result
    assert "n" in result
    assert "m" in result
    assert math.isfinite(result["cost"])
    T = result["T"]
    assert len(T) == n
    assert all(len(row) == m for row in T)
