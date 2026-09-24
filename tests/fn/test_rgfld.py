"""Tests for rgfld.rangayyan_fisher_lda."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_fisher_lda


def test_rgfld_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.integers(0, 2, 40)
    result = rangayyan_fisher_lda(X, y)
    assert isinstance(result, dict)
    assert "w" in result
    assert len(result["w"]) == 3


def test_rgfld_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (20, 2))
    y = rng.integers(0, 2, 20)
    result = rangayyan_fisher_lda(X, y)
    assert isinstance(result, dict)
    assert "w" in result
    assert len(result["w"]) == 2
