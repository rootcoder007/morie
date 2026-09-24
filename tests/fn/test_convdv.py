"""Tests for convdv.convex_divergence."""

from morie.fn import _array_core as np

from morie.fn.convdv import convex_divergence


def test_convdv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 50)
    q = rng.uniform(0, 1, 50)
    result = convex_divergence(p, q, "kl")
    assert isinstance(result, dict)
    for key in ("divergence", "estimate", "terms", "support", "n", "generator"):
        assert key in result
    assert result["n"] == 50
    assert len(result["terms"]) == 50
    assert 0 <= result["support"] <= 50


def test_convdv_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 10)
    q = list(p)
    result = convex_divergence(p, q, "kl")
    assert isinstance(result, dict)
    for key in ("divergence", "estimate", "terms", "support", "n", "f_inf"):
        assert key in result
    assert result["n"] == 10
    assert len(result["terms"]) == 10
