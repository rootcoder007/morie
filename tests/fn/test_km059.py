"""Tests for km059.kamath_ch4_kronecker_product."""

from morie.fn import _array_core as np
from morie.fn.km059 import kamath_ch4_kronecker_product


def test_km059_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch4_kronecker_product(A, B)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km059_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch4_kronecker_product(A, B)
    assert isinstance(result, dict)


def test_kronecker_blocks_and_rank():
    A = [[1.0, 2.0], [0.0, 3.0]]
    B = [[1.0, -1.0], [2.0, 0.5]]
    W = [[A[i // 2][j // 2] * B[i % 2][j % 2] for j in range(4)] for i in range(4)]
    r = kamath_ch4_kronecker_product(A, B)
    assert r["W"] == W
    assert r["rank"] == 4 and r["shape"] == (4, 4) and r["n_params"] == 8
