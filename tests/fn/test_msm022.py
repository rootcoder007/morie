"""Tests for msm022.mvsml_linear_mixed_models_eq_5_3."""

from morie.fn import _array_core as np
import math

from morie.fn.msm022 import mvsml_linear_mixed_models_eq_5_3


def test_msm022_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    J = 3

    # Genomic relationship matrix G (positive-definite)
    A = rng.normal(0, 1, (J, J))
    G = A @ A.T + 10 * np.eye(J)

    sigma2_g = 0.5
    sigma2_e = 1.0

    # Simulate line effects
    b = rng.normal(0, 1, J) * math.sqrt(sigma2_g)

    # Incidence matrix Z_L
    Z_L = np.zeros((n, J))
    for i in range(n):
        col = rng.integers(0, J)
        Z_L[i, col] = 1

    # Residuals
    eps = rng.normal(0, 1, n) * math.sqrt(sigma2_e)

    mu_true = 1.0
    y = mu_true + Z_L @ b + eps

    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g, sigma2_e)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mu" in result
    assert "gebv" in result
    assert "method" in result
    assert result["estimate"] == result["mu"]
    assert len(result["gebv"]) == J


def test_msm022_edge():
    """Test edge cases: default sigma2_e."""
    rng = np.random.default_rng(42)
    n = 40
    J = 3

    A = rng.normal(0, 1, (J, J))
    G = A @ A.T + 10 * np.eye(J)

    sigma2_g = 0.7

    b = rng.normal(0, 1, J) * math.sqrt(sigma2_g)

    Z_L = np.zeros((n, J))
    for i in range(n):
        col = rng.integers(0, J)
        Z_L[i, col] = 1

    eps = rng.normal(0, 1, n)

    y = 2.0 + Z_L @ b + eps

    result = mvsml_linear_mixed_models_eq_5_3(y, Z_L, G, sigma2_g)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gebv" in result
    assert "mu" in result
    assert len(result["gebv"]) == J
    assert result["estimate"] == result["mu"]
