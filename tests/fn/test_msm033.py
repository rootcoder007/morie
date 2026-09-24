"""Tests for msm033.mvsml_linear_mixed_models_eq_5_6."""

from morie.fn import _array_core as np

from morie.fn.msm033 import mvsml_linear_mixed_models_eq_5_6


def test_msm033_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, T, J = 20, 2, 2
    Y = rng.normal(0, 1, (n * J * T, 1))
    Z_L = rng.normal(0, 1, (n * J * T, n * T))
    Z_EL = rng.normal(0, 1, (n * J * T, n * J * T))
    G = rng.normal(0, 1, (n, n))
    Sigma_T = np.eye(T)
    Sigma_E = np.eye(J)
    Sigma_2T = np.eye(T)
    R_T = np.eye(T)
    result = mvsml_linear_mixed_models_eq_5_6(
        Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T
    )
    assert isinstance(result, dict)
    assert "estimate" in result


def test_msm033_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n, T, J = 10, 2, 2
    Y = rng.normal(0, 1, (n * J * T, 1))
    Z_L = rng.normal(0, 1, (n * J * T, n * T))
    Z_EL = rng.normal(0, 1, (n * J * T, n * J * T))
    G = rng.normal(0, 1, (n, n))
    Sigma_T = np.eye(T)
    Sigma_E = np.eye(J)
    Sigma_2T = np.eye(T)
    R_T = np.eye(T)
    result = mvsml_linear_mixed_models_eq_5_6(
        Y, Z_L, Z_EL, G, Sigma_T, Sigma_E, Sigma_2T, R_T
    )
    assert isinstance(result, dict)
    assert "estimate" in result
