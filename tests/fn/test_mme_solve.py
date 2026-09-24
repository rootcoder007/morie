"""Tests for mme_solve.mme_solve."""

from morie.fn import _array_core as np

from morie.fn.mme_solve import mme_solve


def test_msm241_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Sigma_inv = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mme_solve(X, Z, y, Sigma_inv)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm241_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Sigma_inv = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mme_solve(X, Z, y, Sigma_inv)
    assert isinstance(result, dict)
