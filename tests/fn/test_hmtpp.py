"""Tests for hmtpp.geron_tensor_parallelism."""

from morie.fn import _array_core as np

from morie.fn.hmtpp import geron_tensor_parallelism


def test_hmtpp_basic():
    """Test basic functionality."""
    model = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = geron_tensor_parallelism(model)
    assert isinstance(result, dict)
    assert "estimate" in result or "output" in result


def test_hmtpp_edge():
    """Test edge cases."""
    model = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = geron_tensor_parallelism(model)
    assert isinstance(result, dict)
