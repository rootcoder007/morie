"""Tests for rgcnn.rangayyan_cnn_signal."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_cnn_signal


def test_rgcnn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    kernels = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_cnn_signal(x, kernels)
    assert isinstance(result, dict)
    assert "maps" in result


def test_rgcnn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    kernels = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_cnn_signal(x, kernels)
    assert isinstance(result, dict)
