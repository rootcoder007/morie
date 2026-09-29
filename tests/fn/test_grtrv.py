"""Tests for grtrv.geron_tree_regression_leaf."""

from morie.fn import _array_core as np
from morie.fn.grtrv import geron_tree_regression_leaf


def test_grtrv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    leaf_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_regression_leaf(y, leaf_mask)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grtrv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    leaf_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tree_regression_leaf(y, leaf_mask)
    assert isinstance(result, dict)


def test_leaf_mean_and_mse_recomputed():
    import pytest

    y = [2.0, 7.5, 3.0, 9.0, 4.0]
    mask = [True, False, True, False, True]
    sel = [2.0, 3.0, 4.0]
    m = sum(sel) / 3
    r = geron_tree_regression_leaf(y, mask)
    assert r["prediction"] == pytest.approx(m, rel=1e-15)
    assert r["mse"] == pytest.approx(sum((v - m) ** 2 for v in sel) / 3, rel=1e-14)
    assert r["n_leaf"] == 3 and r["n"] == 5
