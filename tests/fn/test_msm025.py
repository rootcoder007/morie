"""Tests for msm025.mvsml_linear_mixed_models_eq_5_3."""

from morie.fn import _array_core as np

from morie.fn.msm025 import mvsml_linear_mixed_models_eq_5_3


def test_msm025_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    sigma2_g = 0.5
    sigma2_e = 1.0
    mu_true = 1.0

    # Random line assignments (incidence matrix)
    line = rng.integers(0, p, n)
    Z_L = np.zeros((n, p))
    for i in range(n):
        Z_L[i, line[i]] = 1

    # Genomic relationship matrix (identity for simplicity)
    G = np.eye(p)

    # Breeding values: b ~ N(0, sigma2_g * I)
    b = [rng.normal(0, 1) * (sigma2_g ** 0.5) for _ in range(p)]

    # Residuals: eps ~ N(0, sigma2_e)
    eps = [rng.normal(0, 1) * (sigma2_e ** 0.5) for _ in range(n)]

    # Phenotype
    y = [mu_true + b[line[i]] + eps[i] for i in range(n)]

    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g, sigma2_e)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert "mu" in result
    assert result["estimate"] == result["mu"]
    assert len(result["gebv"]) == p


def test_msm025_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    n = 20
    p = 2
    sigma2_g = 0.7
    mu_true = 2.0

    # Random line assignments
    line = rng.integers(0, p, n)
    Z_L = np.zeros((n, p))
    for i in range(n):
        Z_L[i, line[i]] = 1

    G = np.eye(p)

    # Breeding values
    b = [rng.normal(0, 1) * (sigma2_g ** 0.5) for _ in range(p)]

    # Residuals (sigma2_e defaults to 1.0)
    eps = [rng.normal(0, 1) for _ in range(n)]

    # Phenotype
    y = [mu_true + b[line[i]] + eps[i] for i in range(n)]

    # Use default sigma2_e
    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert len(result["gebv"]) == p
    assert result["estimate"] == result["mu"]
