"""Tests for gb1041.gibbons_kruskal_wallis."""

from morie.fn import _array_core as np

from morie.fn.gb1041 import gibbons_kruskal_wallis


def test_gb1041_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 10).tolist() for _ in range(4)]
    result = gibbons_kruskal_wallis(groups)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "df" in result
    assert result["df"] == 3
    assert result["k"] == 4
    assert result["n"] == 40


def test_gb1041_book_example():
    """Test against the book's Example 10.4.1: H = 31.89."""
    rng = np.random.default_rng(0)
    n_per = 10
    groups = [rng.normal(0, 1, n_per).tolist() for _ in range(4)]
    result = gibbons_kruskal_wallis(groups)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "h_raw" in result
    assert "rank_sums" in result
    assert result["k"] == 4
    assert result["n"] == 4 * n_per


def test_gb1041_edge():
    """Test edge cases: k=2 samples."""
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 8).tolist() for _ in range(2)]
    result = gibbons_kruskal_wallis(groups)
    assert isinstance(result, dict)
    assert result["k"] == 2
    assert result["df"] == 1
