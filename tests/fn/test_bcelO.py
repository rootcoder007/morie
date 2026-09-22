"""Tests for bcelO.binary_crossentropy_loss."""

from morie.fn import _array_core as np

from morie.fn.bcelO import binary_crossentropy_loss


def test_bcelO_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    P = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = binary_crossentropy_loss(Y, P)
    assert isinstance(result, dict)
    assert "loss" in result
def test_bcelO_edge():
    """Test edge cases."""
    Y = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    P = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = binary_crossentropy_loss(Y, P)
    assert isinstance(result, dict)
