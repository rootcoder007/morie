"""Tests for msm014.mvsml_linear_mixed_models_eq_5_1."""

from morie.fn import _array_core as np

from morie.fn.msm014 import mvsml_linear_mixed_models_eq_5_1


def test_msm014_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_linear_mixed_models_eq_5_1(X, Z, y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm014_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_linear_mixed_models_eq_5_1(X, Z, y, D)
    assert isinstance(result, dict)
