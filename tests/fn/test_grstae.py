"""Tests for grstae.geron_stacked_autoencoder."""

from morie.fn import _array_core as np

from morie.fn.grstae import geron_stacked_autoencoder


def test_grstae_basic():
    """Test basic functionality."""
    x = [[1.0, 2.0]]
    layer_weights = [W]
    result = geron_stacked_autoencoder(x, layer_weights)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grstae_edge():
    """Test edge cases."""
    x = [[1.0, 2.0]]
    layer_weights = [W]
    result = geron_stacked_autoencoder(x, layer_weights)
    assert isinstance(result, dict)
