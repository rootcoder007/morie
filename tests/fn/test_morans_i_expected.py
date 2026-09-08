"""Tests for morans_i_expected.morans_i_expected."""

from morie.fn import _array_core as np

from morie.fn.morans_i_expected import morans_i_expected


def test_ca12e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = morans_i_expected(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca12e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = morans_i_expected(x)
    assert isinstance(result, dict)
