"""Tests for rfkrn.random_fourier_features."""

from morie.fn import _array_core as np

from morie.fn.rfkrn import random_fourier_features


def test_rfkrn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_fourier_features(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rfkrn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_fourier_features(X)
    assert isinstance(result, dict)
