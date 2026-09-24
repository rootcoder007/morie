"""Tests for hmcart.geron_cart_algorithm."""

from morie.fn import _array_core as np

from morie.fn.hmcart import geron_cart_algorithm


def test_hmcart_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = geron_cart_algorithm(X, y, criterion="mse")
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "n_leaves" in result
    assert "tree" in result
    assert "depth" in result
    assert "train_mse" in result
    assert len(result["predictions"]) == 40
    assert isinstance(result["n_leaves"], int)
    assert result["depth"] >= 0


def test_hmcart_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = geron_cart_algorithm(X, y, criterion="mse", max_depth=3)
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "n_leaves" in result
    assert "train_mse" in result
    assert len(result["predictions"]) == 40
    assert isinstance(result["n_leaves"], int)
