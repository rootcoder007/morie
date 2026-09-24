"""Tests for pcmpr1.prediction_compression."""

from morie.fn import _array_core as np

from morie.fn.pcmpr1 import prediction_compression


def test_pcmpr1_basic():
    """Test basic functionality."""
    model = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    data = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = prediction_compression(model, data)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_pcmpr1_edge():
    """Test edge cases."""
    model = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    data = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = prediction_compression(model, data)
    assert isinstance(result, dict)
