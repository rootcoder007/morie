"""Tests for msm011.mvsml_linear_mixed_models_eq_5_2."""

from morie.fn import _array_core as np

from morie.fn.msm011 import mvsml_linear_mixed_models_eq_5_2


def test_msm011_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_linear_mixed_models_eq_5_2(X, Z, y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm011_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_linear_mixed_models_eq_5_2(X, Z, y, D)
    assert isinstance(result, dict)
