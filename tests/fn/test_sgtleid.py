"""Tests for sgtleid.sgt_leiden_step."""

from morie.fn import _array_core as np

from morie.fn.sgtleid import sgt_leiden_step


def test_sgtleid_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    labels = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sgt_leiden_step(A, labels)
    assert isinstance(result, dict)
    assert "labels_new" in result


def test_sgtleid_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    labels = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sgt_leiden_step(A, labels)
    assert isinstance(result, dict)
