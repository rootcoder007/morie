"""Tests for shfflm.shuffle_model."""

from morie.fn import _array_core as np

from morie.fn.shfflm import shuffle_model


def test_shfflm_basic():
    """Test basic functionality."""
    epsilon0 = 0.1
    n = 5
    delta = 0.1
    result = shuffle_model(epsilon0, n, delta)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_shfflm_edge():
    """Test edge cases."""
    epsilon0 = 0.1
    n = 5
    delta = 0.1
    result = shuffle_model(epsilon0, n, delta)
    assert isinstance(result, dict)
