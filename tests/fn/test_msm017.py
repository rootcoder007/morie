"""Tests for msm017.mvsml_linear_mixed_models_eq_5_3."""

from morie.fn import _array_core as np

from morie.fn.msm017 import mvsml_linear_mixed_models_eq_5_3


def test_msm017_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    q = 3
    sigma2_g = 0.5
    sigma2_e = 1.0

    # Genomic relationship matrix (identity is positive definite)
    G = np.eye(q)

    # Incidence matrix n x q: each observation assigned to one line
    Z_L = np.zeros((n, q))
    for i in range(n):
        Z_L[i, rng.integers(0, q)] = 1

    # Phenotype vector of length n
    y = rng.normal(0, 1, n)

    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g, sigma2_e)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert "mu" in result
    assert "method" in result
    # estimate and mu are both the intercept estimate
    assert result["estimate"] == result["mu"]
    # GEBV length matches number of columns of Z_L
    assert len(result["gebv"]) == q


def test_msm017_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    n = 40
    q = 4
    sigma2_g = 0.7
    # default sigma2_e = 1.0

    G = np.eye(q)

    Z_L = np.zeros((n, q))
    for i in range(n):
        Z_L[i, rng.integers(0, q)] = 1

    y = rng.normal(0, 1, n)

    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert "mu" in result
    assert "method" in result
    assert result["estimate"] == result["mu"]
    assert len(result["gebv"]) == q
