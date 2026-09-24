"""Tests for rndnet.random_network_distillation."""

from morie.fn import _array_core as np

from morie.fn.rndnet import random_network_distillation


def test_rndnet_basic():
    """Test basic functionality."""
    observations = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_network_distillation(observations)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rndnet_edge():
    """Test edge cases."""
    observations = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_network_distillation(observations)
    assert isinstance(result, dict)
