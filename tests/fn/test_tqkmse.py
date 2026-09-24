"""Tests for tqkmse.turboquant_kv_mse."""

from morie.fn import _array_core as np

from morie.fn.tqkmse import turboquant_kv_mse


def test_tqkmse_basic():
    """Test basic functionality."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = turboquant_kv_mse(K)
    assert isinstance(result, dict)
    assert "mse" in result


def test_tqkmse_edge():
    """Test edge cases."""
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = turboquant_kv_mse(K)
    assert isinstance(result, dict)
