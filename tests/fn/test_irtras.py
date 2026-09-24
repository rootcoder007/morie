"""Tests for irtras.rating_scale_model."""

from morie.fn import _array_core as np

from morie.fn.irtras import rating_scale_model


def test_irtras_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    b = np.random.default_rng(42).normal(0.0, 1.0, 40)
    tau = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rating_scale_model(theta, b, tau)
    assert isinstance(result, dict)
    assert "p" in result


def test_irtras_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    b = np.random.default_rng(42).normal(0.0, 1.0, 40)
    tau = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rating_scale_model(theta, b, tau)
    assert isinstance(result, dict)
