"""Tests for t5enc.t5."""

from morie.fn import _array_core as np

from morie.fn.t5enc import t5


def test_t5enc_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = t5(tokens)
    assert isinstance(result, dict)
    assert "input" in result


def test_t5enc_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = t5(tokens)
    assert isinstance(result, dict)
