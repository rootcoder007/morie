"""Tests for grtts.geron_train_test_split."""

from morie.fn import _array_core as np

from morie.fn.grtts import geron_train_test_split


def test_grtts_basic():
    """Test basic functionality."""
    ids = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = geron_train_test_split(ids)
    assert isinstance(result, dict)
    assert "test" in result or "test" in result


def test_grtts_edge():
    """Test edge cases."""
    ids = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = geron_train_test_split(ids)
    assert isinstance(result, dict)
