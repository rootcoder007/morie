"""Tests for ccelO.categorical_crossentropy_loss."""

from morie.fn import _array_core as np

from morie.fn.ccelO import categorical_crossentropy_loss


def test_ccelO_basic():
    """Test basic functionality."""
    Y = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    P = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = categorical_crossentropy_loss(Y, P)
    assert isinstance(result, dict)
    assert "loss" in result
def test_ccelO_edge():
    """Test edge cases."""
    Y = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    P = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = categorical_crossentropy_loss(Y, P)
    assert isinstance(result, dict)
