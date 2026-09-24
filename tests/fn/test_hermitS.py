"""Tests for hermitS.hermite_basis."""

from morie.fn import _array_core as np

from morie.fn.hermitS import hermite_basis


def test_hermitS_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 50)
    result = hermite_basis(x, K=3, kind="physicist")
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "basis" in result
    assert "top" in result
    assert "K" in result
    assert "kind" in result
    assert "n" in result
    assert "method" in result
    assert result["K"] == 3
    assert result["kind"] == "physicist"
    assert result["n"] == 50
    # basis has shape (n, K+1)
    assert result["basis"].shape[0] == 50
    assert result["basis"].shape[1] == 4


def test_hermitS_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10)
    result = hermite_basis(x, K=2, kind="probabilist")
    assert isinstance(result, dict)
    assert result["K"] == 2
    assert result["kind"] == "probabilist"
    assert result["n"] == 10
    # basis has K+1 columns
    assert result["basis"].shape[1] == 3
