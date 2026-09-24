"""Tests for fpca.functional_pca."""

from morie.fn import _array_core as np
from morie.fn.fpca import functional_pca


def test_fpca_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 40, 50
    data_functions = rng.normal(0, 1, (n, m))
    result = functional_pca(data_functions, 3)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scores" in result
    assert "eigenfuncs" in result
    assert "eigenvalues" in result
    assert "mean" in result
    assert 0.0 <= result["estimate"] <= 1.0
    assert len(result["scores"]) == n
    assert len(result["scores"][0]) == 3
    assert len(result["eigenfuncs"]) == 3
    assert len(result["eigenfuncs"][0]) == m
    assert len(result["eigenvalues"]) == 3
    assert len(result["mean"]) == m


def test_fpca_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    n, m = 5, 4
    data_functions = rng.normal(0, 1, (n, m))
    result = functional_pca(data_functions, 1, a=0.0, b=1.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert 0.0 <= result["estimate"] <= 1.0
    assert len(result["scores"]) == n
    assert len(result["scores"][0]) == 1
    assert len(result["eigenfuncs"]) == 1
    assert len(result["eigenfuncs"][0]) == m
