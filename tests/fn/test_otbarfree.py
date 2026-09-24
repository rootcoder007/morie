"""Tests for otbarfree.ot_barycenter_free."""

from morie.fn import _array_core as np
import math

from morie.fn.otbarfree import ot_barycenter_free


def test_otbarfree_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    K = 3
    n_k = 20
    d = 2
    X_list = [rng.normal(0, 1, (n_k, d)) for _ in range(K)]
    weights = rng.uniform(1.0, 10.0, K)
    n_supp = 5
    result = ot_barycenter_free(X_list, weights, n_supp)
    assert isinstance(result, dict)
    assert "Y" in result
    assert "weights_y" in result
    assert "cost" in result
    assert "n_supp" in result
    assert "d" in result
    assert "K" in result
    assert "iters" in result
    assert math.isfinite(result["cost"])
    assert result["n_supp"] == 5
    assert result["d"] == 2
    assert result["K"] == K
    assert len(result["Y"]) == 5
    assert len(result["weights_y"]) == 5
    assert math.isfinite(result["iters"])


def test_otbarfree_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    K = 2
    n_k = 10
    d = 3
    X_list = [rng.normal(0, 1, (n_k, d)) for _ in range(K)]
    weights = rng.uniform(1.0, 10.0, K)
    n_supp = 1
    result = ot_barycenter_free(X_list, weights, n_supp, max_iter=5)
    assert isinstance(result, dict)
    assert "Y" in result
    assert "weights_y" in result
    assert "cost" in result
    assert result["n_supp"] == 1
    assert result["iters"] == 5
    assert math.isfinite(result["cost"])
    assert len(result["Y"]) == 1
    assert len(result["weights_y"]) == 1
