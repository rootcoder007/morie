"""Tests for cfafm2.cfa_multifactor."""

import math

from morie.fn import _array_core as np

from morie.fn.cfafm2 import cfa_multifactor


def test_cfafm2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p, k = 100, 5, 2
    X = rng.normal(0, 1, (n, p))
    factor_pattern = [[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]]
    result = cfa_multifactor(X, factor_pattern)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert "loadings" in result
    assert "uniquenesses" in result
    assert "fml" in result
    assert "max_resid" in result
    assert "communality" in result
    assert "n_iter" in result
    assert result["p"] == p
    assert result["k"] == k
    assert len(result["loadings"]) == p
    assert len(result["loadings"][0]) == k
    assert len(result["uniquenesses"]) == p
    assert len(result["communality"]) == p


def test_cfafm2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p, k = 40, 3, 2
    X = rng.normal(0, 1, (n, p))
    factor_pattern = [[1, 1], [1, 1], [1, 1]]
    result = cfa_multifactor(X, factor_pattern)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert "loadings" in result
    assert "uniquenesses" in result
    assert "communality" in result
    assert result["p"] == p
    assert result["k"] == k
    assert len(result["loadings"]) == p
    assert len(result["loadings"][0]) == k
    assert len(result["uniquenesses"]) == p
    assert len(result["communality"]) == p
