"""Tests for multM.multiple_mediators."""

from morie.fn import _array_core as np

from morie.fn.multM import multiple_mediators


def test_multM_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M_list = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = multiple_mediators(Y, X, M_list)
    assert isinstance(result, dict)
    assert "indirect" in result


def test_multM_edge():
    """Test edge cases."""
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M_list = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = multiple_mediators(Y, X, M_list)
    assert isinstance(result, dict)
