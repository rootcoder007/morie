"""Tests for grace.grace."""

from morie.fn import _array_core as np

from morie.fn.grace import grace


def test_grace_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    U = rng.normal(0, 1, (n, 5))
    V = rng.normal(0, 1, (n, 5))
    result = grace(U, V)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loss" in result
    assert "tau" in result
    assert result["tau"] == 0.5
    assert result["intra_view_negatives"] is True
    assert result["n_nodes"] == n
    assert result["method"].startswith("node-level contrastive")
    # estimate and loss must coincide and be finite
    assert result["estimate"] == result["loss"]
    assert np.isfinite(result["estimate"])


def test_grace_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 4
    U = rng.normal(0, 1, (n, 3))
    V = rng.normal(0, 1, (n, 3))
    result = grace(U, V, tau=0.25, intra=False)
    assert isinstance(result, dict)
    assert result["tau"] == 0.25
    assert result["intra_view_negatives"] is False
    assert result["n_nodes"] == n
    assert result["estimate"] == result["loss"]
