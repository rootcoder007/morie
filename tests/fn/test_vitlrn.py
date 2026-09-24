"""Tests for vitlrn.vit_layer_norm."""

from morie.fn import _array_core as np

from morie.fn.vitlrn import vit_layer_norm


def test_vitlrn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = vit_layer_norm(x)
    assert isinstance(result, dict)
    assert "y" in result


def test_vitlrn_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = vit_layer_norm(x)
    assert isinstance(result, dict)
