"""Tests for otbar.ot_barycenter_fixed."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.otbar import ot_barycenter_fixed


def test_otbar_basic():
    """Test basic functionality."""
    n = 5
    K = 3
    rng = np.random.default_rng(0)
    # Build input histograms (n x K), each column sums to 1
    A = []
    for i in range(n):
        row = []
        for k in range(K):
            row.append(rng.uniform(0, 1))
        A.append(row)
    for k in range(K):
        col_sum = sum(A[i][k] for i in range(n))
        for i in range(n):
            A[i][k] /= col_sum

    # Build cost matrices (n x n) for each input
    x = np.linspace(0, 1, n)
    C = [[(x[i] - x[j]) ** 2 for j in range(n)] for i in range(n)]
    C_list = [C for _ in range(K)]

    # Weights
    weights = [1.0 / K] * K

    epsilon = 0.1

    result = ot_barycenter_fixed(A, C_list, weights, epsilon)

    assert isinstance(result, dict)
    assert "bary" in result
    assert "mass" in result
    assert "n" in result
    assert "K" in result
    assert "iters" in result
    assert "method" in result

    bary = result["bary"]
    assert isinstance(bary, list)
    assert len(bary) == n
    for v in bary:
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert v >= 0.0

    mass = result["mass"]
    assert math.isfinite(mass)
    assert result["n"] == n
    assert result["K"] == K


def test_otbar_edge():
    """Test that invalid epsilon raises ValueError."""
    n = 5
    K = 3
    rng = np.random.default_rng(0)
    A = []
    for i in range(n):
        row = []
        for k in range(K):
            row.append(rng.uniform(0, 1))
        A.append(row)
    for k in range(K):
        col_sum = sum(A[i][k] for i in range(n))
        for i in range(n):
            A[i][k] /= col_sum

    x = np.linspace(0, 1, n)
    C = [[(x[i] - x[j]) ** 2 for j in range(n)] for i in range(n)]
    C_list = [C for _ in range(K)]

    weights = [1.0 / K] * K

    epsilon = -0.1

    with pytest.raises(ValueError):
        ot_barycenter_fixed(A, C_list, weights, epsilon)
