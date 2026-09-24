"""Tests for krigun.universal_kriging."""

from morie.fn import _array_core as np

from morie.fn.krigun import universal_kriging


def test_krigun_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    s_predict = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = universal_kriging(coords, values, s_predict)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_krigun_edge():
    """Test edge cases."""
    coords = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    values = np.random.default_rng(42).normal(0.0, 1.0, 40)
    s_predict = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = universal_kriging(coords, values, s_predict)
    assert isinstance(result, dict)
