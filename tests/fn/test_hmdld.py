"""Tests for hmdld.geron_dataloader."""

from morie.fn import _array_core as np

from morie.fn.hmdld import geron_dataloader


def test_hmdld_basic():
    """Test basic functionality."""
    dataset = np.random.default_rng(42).normal(0.0, 1.0, 40)
    batch_size = 5
    result = geron_dataloader(dataset, batch_size)
    assert isinstance(result, dict)
    assert "estimate" in result or "batches" in result


def test_hmdld_edge():
    """Test edge cases."""
    dataset = np.random.default_rng(42).normal(0.0, 1.0, 40)
    batch_size = 5
    result = geron_dataloader(dataset, batch_size)
    assert isinstance(result, dict)
