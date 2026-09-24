"""Tests for difpst.dif_p_diff."""

from morie.fn import _array_core as np
import pytest

from morie.fn.difpst import dif_p_diff


def test_difpst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, k = 100, 5
    X = rng.integers(0, 2, (n, k))
    group = rng.integers(0, 2, n)
    result = dif_p_diff(X, group)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "p_diff" in result


def test_difpst_edge():
    """Test edge case: missing reference group."""
    rng = np.random.default_rng(42)
    n, k = 100, 5
    X = rng.integers(0, 2, (n, k))
    group = rng.integers(1, 2, n)
    with pytest.raises(ValueError):
        dif_p_diff(X, group)
