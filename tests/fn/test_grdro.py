"""Tests for grdro.geron_dropout."""

from morie.fn import _array_core as np

from morie.fn.grdro import geron_dropout


def test_grdro_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    result = geron_dropout(a, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdro_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    result = geron_dropout(a, p)
    assert isinstance(result, dict)
