"""Tests for hotcld.hot_cold_spots."""

from morie.fn import _array_core as np

from morie.fn.hotcld import hot_cold_spots


def _build_weights(n, threshold, seed):
    """Build a symmetric 0/1 weight matrix of shape (n, n)."""
    rng = np.random.default_rng(seed)
    rand_mat = rng.uniform(0, 1, (n, n))
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rand_mat[i][j] > threshold:
                W[i][j] = 1.0
                W[j][i] = 1.0
    return W


def test_hotcld_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    W = _build_weights(n, threshold=0.7, seed=42)
    alpha = 0.05
    result = hot_cold_spots(x, W, alpha)
    assert isinstance(result, dict)
    has_stat = (
        ("estimate" in result)
        or ("statistic" in result)
        or ("z" in result)
        or ("p_value" in result)
    )
    assert has_stat


def test_hotcld_edge():
    """Test edge cases with a small but valid input."""
    rng = np.random.default_rng(7)
    n = 40
    x = rng.normal(0, 1, n)
    W = _build_weights(n, threshold=0.5, seed=7)
    alpha = 0.10
    result = hot_cold_spots(x, W, alpha)
    assert isinstance(result, dict)
    has_stat = (
        ("estimate" in result)
        or ("statistic" in result)
        or ("z" in result)
        or ("p_value" in result)
    )
    assert has_stat
