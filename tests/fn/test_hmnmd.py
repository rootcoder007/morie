"""Tests for hmnmd.geron_numerical_diff."""

from morie.fn import _array_core as np

from morie.fn.hmnmd import geron_numerical_diff


def test_hmnmd_basic():
    """Test basic functionality."""
    f = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_numerical_diff(f, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "derivative" in result


def test_hmnmd_edge():
    """Test edge cases."""
    f = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_numerical_diff(f, x)
    assert isinstance(result, dict)
