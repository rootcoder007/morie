"""Tests for hmmpp.geron_model_parallelism."""

from morie.fn import _array_core as np

from morie.fn.hmmpp import geron_model_parallelism


def test_hmmpp_basic():
    """Test basic functionality."""
    model = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    n_devices = 5
    result = geron_model_parallelism(model, n_devices)
    assert isinstance(result, dict)
    assert "estimate" in result or "assignment" in result


def test_hmmpp_edge():
    """Test edge cases."""
    model = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    n_devices = 5
    result = geron_model_parallelism(model, n_devices)
    assert isinstance(result, dict)
