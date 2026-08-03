"""Tests for grand_mean_model.grand_mean_model."""

from morie.fn import _array_core as np

from morie.fn.grand_mean_model import grand_mean_model


def test_ca7e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = grand_mean_model(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca7e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = grand_mean_model(x)
    assert isinstance(result, dict)
