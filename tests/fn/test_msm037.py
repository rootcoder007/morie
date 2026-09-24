"""Tests for msm037.mvsml_linear_mixed_models_eq_5_4."""

from morie.fn import _array_core as np

from morie.fn.msm037 import mvsml_linear_mixed_models_eq_5_4


def test_msm037_basic():
    """Test basic functionality."""
    y = [5.0, 6.0, 5.4, 6.8]
    X_E = [[0.0]] * 4
    Z_L = [[1, 0], [0, 1], [1, 0], [0, 1]]
    Z_EL = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    G = [[1.0, 0.0], [0.0, 1.0]]
    sigma2_g = 0.5
    Sigma_E = [[0.3, 0.0], [0.0, 0.3]]
    result = mvsml_linear_mixed_models_eq_5_4(y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm037_edge():
    """Test edge cases."""
    y = [5.0, 6.0, 5.4, 6.8]
    X_E = [[0.0]] * 4
    Z_L = [[1, 0], [0, 1], [1, 0], [0, 1]]
    Z_EL = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    G = [[1.0, 0.0], [0.0, 1.0]]
    sigma2_g = 0.5
    Sigma_E = [[0.3, 0.0], [0.0, 0.3]]
    result = mvsml_linear_mixed_models_eq_5_4(y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E)
    assert isinstance(result, dict)
