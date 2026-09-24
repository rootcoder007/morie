"""Tests for katzc.katz_centrality."""

from morie.fn import _array_core as np

from morie.fn.katzc import katz_centrality


def test_katzc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    A = rng.normal(0, 1, (n, n))
    y = [1.0] * n
    alpha = 0.05
    beta = 0.8
    result = katz_centrality(y, A, alpha=alpha, beta=beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "centrality" in result
    assert "n" in result
    assert "alpha" in result
    assert len(result["centrality"]) == n
    assert result["n"] == n
    assert result["alpha"] == alpha


def test_katzc_edge():
    """Test edge cases with minimal valid alpha."""
    rng = np.random.default_rng(43)
    n = 5
    A = rng.normal(0, 1, (n, n))
    y = [1.0] * n
    alpha = 0.01
    beta = 0.0
    result = katz_centrality(y, A, alpha=alpha, beta=beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "centrality" in result
    assert len(result["centrality"]) == n
    assert result["n"] == n
    assert result["alpha"] == alpha
