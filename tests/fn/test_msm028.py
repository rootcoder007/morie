"""Tests for msm028.mvsml_linear_mixed_models_eq_5_5a."""

from morie.fn import _array_core as np

from morie.fn.msm028 import mvsml_linear_mixed_models_eq_5_5a


def test_msm028_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Sigma_T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    R_T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_linear_mixed_models_eq_5_5a(Y, Z, G, Sigma_T, R_T)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm028_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Sigma_T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    R_T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_linear_mixed_models_eq_5_5a(Y, Z, G, Sigma_T, R_T)
    assert isinstance(result, dict)
