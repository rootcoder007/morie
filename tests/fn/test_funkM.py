"""Tests for funkM.funk_svd."""

from morie.fn import _array_core as np

from morie.fn.funkM import funk_svd


def test_funkM_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    nu, ni = 5, 5
    n_ratings = 10
    ratings = [
        (int(rng.integers(0, nu)), int(rng.integers(0, ni)),
         float(rng.uniform(1.0, 5.0)))
        for _ in range(n_ratings)
    ]
    result = funk_svd(ratings, nu, ni, factors=2, epochs=2)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "rmse" in result
    assert "rmse_history" in result
    assert "mu" in result
    assert "b_user" in result
    assert "b_item" in result
    assert "P" in result
    assert "Q" in result
    assert "factors" in result
    assert "incremental" in result
    assert "observed" in result
    assert "density" in result
    assert "method" in result
    assert "note" in result
    assert result["factors"] == 2
    assert result["observed"] == n_ratings
    assert result["density"] == n_ratings / float(nu * ni)


def test_funkM_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    nu, ni = 3, 4
    n_ratings = 8
    ratings = [
        (int(rng.integers(0, nu)), int(rng.integers(0, ni)),
         float(rng.uniform(1.0, 5.0)))
        for _ in range(n_ratings)
    ]
    result = funk_svd(ratings, nu, ni, factors=2, epochs=1)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mu" in result
