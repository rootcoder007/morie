"""Tests for msm024.mvsml_linear_mixed_models_eq_5_3."""

from morie.fn import _array_core as np

from morie.fn.msm024 import mvsml_linear_mixed_models_eq_5_3


def test_msm024_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    sigma2_g = 0.5
    sigma2_e = 1.0

    # Build a positive-definite G (genomic relationship matrix)
    A = rng.normal(0, 1, (10, 10))
    G = A @ A.T + 10  # 10x10, PD

    # Generate b ~ N(0, sigma2_g * G)
    L = np.linalg.cholesky(sigma2_g * G)
    b = L @ rng.normal(0, 1, 10)

    # Z_L is n x 10 incidence matrix
    Z_L = np.zeros((n, 10))
    for i in range(n):
        Z_L[i, rng.integers(0, 10)] = 1

    # Generate phenotype
    eps = rng.normal(0, np.sqrt(sigma2_e), n)
    mu_true = 1.0
    y = mu_true + Z_L @ b + eps

    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g, sigma2_e)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert "mu" in result
    assert "method" in result
    assert result["estimate"] == result["mu"]
    assert len(result["gebv"]) == 10
    assert len(result["gebv"]) == Z_L.shape[1]


def test_msm024_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 50
    sigma2_g = 0.7

    A = rng.normal(0, 1, (5, 5))
    G = A @ A.T + 10  # 5x5, PD

    L = np.linalg.cholesky(sigma2_g * G)
    b = L @ rng.normal(0, 1, 5)

    Z_L = np.zeros((n, 5))
    for i in range(n):
        Z_L[i, rng.integers(0, 5)] = 1

    eps = rng.normal(0, 1, n)
    y = 2.0 + Z_L @ b + eps

    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g)  # default sigma2_e=1.0
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert len(result["gebv"]) == 5
    assert result["estimate"] == result["mu"]
