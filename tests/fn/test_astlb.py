"""Tests for astlb.astle_balding_grm."""

from morie.fn import _array_core as np

from morie.fn.astlb import astle_balding_grm


def test_astlb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 40, 20
    marker_matrix = rng.integers(0, 3, (n, m))
    result = astle_balding_grm(marker_matrix)
    assert isinstance(result, dict)
    assert "G" in result
    assert "n_markers_used" in result
    assert "n_dropped" in result
    assert "freq" in result
    assert "mean_diagonal" in result


def test_astlb_edge():
    """Test edge cases with monomorphic markers."""
    rng = np.random.default_rng(42)
    n, m = 40, 20
    rows = rng.integers(0, 3, (n, m))
    rows_with_mono = [list(row) + [0] for row in rows]
    marker_matrix = np.array(rows_with_mono)
    result = astle_balding_grm(marker_matrix)
    assert isinstance(result, dict)
    assert result["n_dropped"] >= 1
    assert result["n_markers_used"] == m
