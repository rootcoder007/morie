"""Tests for gb1051.gibbons_k_rank_alt."""

from morie.fn import _array_core as np

from morie.fn.gb1051 import gibbons_k_rank_alt


def test_gb1051_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 40), rng.normal(0, 1, 40), rng.normal(0, 1, 40)]
    result = gibbons_k_rank_alt(groups)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb1051_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 20), rng.normal(0, 1, 20)]
    result = gibbons_k_rank_alt(groups)
    assert isinstance(result, dict)
