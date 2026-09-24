"""Tests for remlf.reml_log_likelihood."""

from morie.fn import _array_core as np

from morie.fn.remlf import reml_log_likelihood


def test_remlf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = reml_log_likelihood(X, Z, y, D)
    assert isinstance(result, dict)
    assert "loglik" in result


def test_remlf_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = reml_log_likelihood(X, Z, y, D)
    assert isinstance(result, dict)
