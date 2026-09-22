"""Tests for depthS.simplicial_depth."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.depthS import simplicial_depth


def test_depthS_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 1))
    theta = [0.0]
    result = simplicial_depth(X, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    # Independent closed-form computation: 2 * F(theta) * (1 - F(theta))
    xs = [float(X[i, 0]) for i in range(100)]
    n = len(xs)
    F_hat = sum(1 for v in xs if v <= 0.0) / float(n)
    expected_cf = 2.0 * F_hat * (1.0 - F_hat)
    assert abs(result["closed_form_1d"] - expected_cf) < 1e-12
    assert abs(result["depth"] - result["estimate"]) < 1e-12
    assert result["n"] == 100
    assert result["d"] == 1
    # n_simplices = C(100, 2) = 4950
    assert result["n_simplices"] == 4950
    # Depth = n_containing / n_simplices
    assert abs(result["depth"] - result["n_containing"] / result["n_simplices"]) < 1e-12


def test_depthS_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 1))
    theta = [0.0]
    result = simplicial_depth(X, theta)
    assert isinstance(result, dict)
    assert result["n"] == 50
    assert result["d"] == 1
    assert result["n_simplices"] == 1225  # C(50, 2)
    # 0 <= depth <= 1
    assert 0.0 <= result["depth"] <= 1.0
    # 0 <= ecdf <= 1
    assert 0.0 <= result["ecdf"] <= 1.0
