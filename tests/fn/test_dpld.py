"""Tests for dpld.l_diversity."""

from morie.fn import _array_core as np

from morie.fn.dpld import l_diversity


def test_dpld_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(0, 1, n)
    quasi_ids = rng.integers(0, 5, (n, 2))
    sensitive = ["A" if i % 3 == 0 else "B" if i % 3 == 1 else "C" for i in range(n)]
    l = 2
    result = l_diversity(X, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
    expected_keys = [
        "estimate", "distinct_l", "entropy_l", "min_entropy", "c_min",
        "satisfies_distinct", "satisfies_entropy", "satisfies_recursive",
        "n_blocks", "min_block_size", "l", "c", "n",
    ]
    for key in expected_keys:
        assert key in result
    assert result["l"] == 2
    assert result["c"] == 1.0
    assert result["n"] == n


def test_dpld_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 20
    X = rng.normal(0, 1, n)
    quasi_ids = rng.integers(0, 2, (n, 2))
    sensitive = ["X" if i % 2 == 0 else "Y" for i in range(n)]
    l = 1
    result = l_diversity(X, quasi_ids, sensitive, l)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "distinct_l" in result
    assert result["l"] == 1
    assert result["n"] == n
