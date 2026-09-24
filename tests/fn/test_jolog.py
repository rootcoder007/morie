"""Tests for jolog.joseph_log_transform."""

from morie.fn import _array_core as np

from morie.fn.jolog import joseph_log_transform


def test_jolog_basic():
    """Test basic functionality."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = joseph_log_transform(x)
    assert isinstance(result, dict)
    assert "w" in result


def test_jolog_edge():
    """Test edge cases."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = joseph_log_transform(x)
    assert isinstance(result, dict)
