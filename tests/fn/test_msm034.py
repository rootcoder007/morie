"""Tests for msm034.mvsml_linear_mixed_models_eq_5_6."""

from morie.fn import _array_core as np

from morie.fn.msm034 import mvsml_linear_mixed_models_eq_5_6


def test_msm034_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    I, J, T = 4, 2, 2
    n = I * J
    Y = rng.normal(0, 1, (n, T))
    Z_L = rng.normal(0, 1, (n, I))
    Z_EL = rng.normal(0, 1, (n, I * J))
    G = rng.normal(0, 1, (I, I))
    Sigma_T = rng.normal(0, 1, (T, T))
    Sigma_E = rng.normal(0, 1, (J, J))
    Sigma_2T = rng.normal(0, 1, (T, T))
    R_T = rng.normal(0, 1, (T, T))
    result = mvsml_linear_mixed_models_eq_5_6(Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T)
    assert isinstance(result, dict)
    assert "estimate" in result


def test_msm034_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    I, J, T = 3, 2, 2
    n = I * J
    Y = rng.normal(0, 1, (n, T))
    Z_L = rng.normal(0, 1, (n, I))
    Z_EL = rng.normal(0, 1, (n, I * J))
    G = rng.normal(0, 1, (I, I))
    Sigma_T = rng.normal(0, 1, (T, T))
    Sigma_E = rng.normal(0, 1, (J, J))
    Sigma_2T = rng.normal(0, 1, (T, T))
    R_T = rng.normal(0, 1, (T, T))
    result = mvsml_linear_mixed_models_eq_5_6(Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T)
    assert isinstance(result, dict)
    assert "estimate" in result
