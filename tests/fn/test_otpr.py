"""Tests for otpr.ot_partial_ot."""

from morie.fn import _array_core as np

from morie.fn.otpr import ot_partial_ot


def test_otpr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, k = 5, 4
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, k)
    C = rng.normal(0, 1, (n, k))
    m = 1.0
    result = ot_partial_ot(a, b, C, m)
    assert isinstance(result, dict)
    assert "T" in result
    assert "cost" in result
    assert "mass" in result
    assert "a_left" in result
    assert "b_left" in result
    assert "n" in result
    assert "m_bins" in result
    assert "method" in result


def test_otpr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, k = 3, 3
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, k)
    C = rng.normal(0, 1, (n, k))
    m = 0.5
    result = ot_partial_ot(a, b, C, m)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["m_bins"] == k
