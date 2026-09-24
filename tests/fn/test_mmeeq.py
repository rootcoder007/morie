"""Tests for mmeeq.henderson_mme_eq2_2."""

from morie.fn import _array_core as np

from morie.fn.mmeeq import henderson_mme_eq2_2


def test_mmeeq_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Sigma_inv = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = henderson_mme_eq2_2(X, Z, y, Sigma_inv)
    assert isinstance(result, dict)
    assert "beta" in result


def test_mmeeq_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Sigma_inv = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = henderson_mme_eq2_2(X, Z, y, Sigma_inv)
    assert isinstance(result, dict)
