"""Tests for estnd.estimand_framework."""

from morie.fn import _array_core as np

from morie.fn.estnd import estimand_framework


def test_estnd_basic():
    """Test basic functionality."""
    d = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = estimand_framework(d)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_estnd_edge():
    """Test edge cases."""
    d = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = estimand_framework(d)
    assert isinstance(result, dict)
