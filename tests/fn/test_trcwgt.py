"""Tests for trcwgt.truncated_combined_weights."""

from morie.fn import _array_core as np

from morie.fn.trcwgt import truncated_combined_weights


def test_trcwgt_basic():
    """Test basic functionality."""
    sw_A = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    sw_C = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = truncated_combined_weights(sw_A, sw_C)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_trcwgt_edge():
    """Test edge cases."""
    sw_A = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    sw_C = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = truncated_combined_weights(sw_A, sw_C)
    assert isinstance(result, dict)
