"""Tests for hmtvt.geron_train_val_test_split."""

from morie.fn import _array_core as np

from morie.fn.hmtvt import geron_train_val_test_split


def test_hmtvt_basic():
    """Test basic functionality."""
    ids = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = geron_train_val_test_split(ids)
    assert isinstance(result, dict)
    assert "train" in result or "train" in result


def test_hmtvt_edge():
    """Test edge cases."""
    ids = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = geron_train_val_test_split(ids)
    assert isinstance(result, dict)
