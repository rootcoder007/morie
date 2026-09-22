"""Tests for confgg.configuration_model."""

from morie.fn import _array_core as np

from morie.fn.confgg import configuration_model


def test_confgg_basic():
    """Test basic functionality."""
    degrees = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = configuration_model(degrees)
    assert isinstance(result, dict)
    assert "edges" in result
def test_confgg_edge():
    """Test edge cases."""
    degrees = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = configuration_model(degrees)
    assert isinstance(result, dict)
