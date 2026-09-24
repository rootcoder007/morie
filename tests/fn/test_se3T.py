"""Tests for se3T.se3_transformer."""

from morie.fn import _array_core as np

from morie.fn.se3T import se3_transformer


def test_se3T_basic():
    """Test basic functionality."""
    positions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    type0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    type1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = se3_transformer(positions, type0, type1)
    assert isinstance(result, dict)
    assert "type1" in result


def test_se3T_edge():
    """Test edge cases."""
    positions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    type0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    type1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = se3_transformer(positions, type0, type1)
    assert isinstance(result, dict)
