"""Tests for morrisM.morris_screening."""

from morie.fn import _array_core as np

from morie.fn.morrisM import morris_screening


def test_morrisM_basic():
    """Test basic functionality."""
    fun = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    k = 5
    result = morris_screening(fun, k)
    assert isinstance(result, dict)
    assert "mu" in result


def test_morrisM_edge():
    """Test edge cases."""
    fun = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    k = 5
    result = morris_screening(fun, k)
    assert isinstance(result, dict)
