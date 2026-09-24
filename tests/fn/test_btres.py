"""Tests for btres.boot_residual_regression."""

from morie.fn import _array_core as np

from morie.fn.btres import boot_residual_regression


def test_btres_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = boot_residual_regression(X, y, B=20)
    assert isinstance(result, dict)
    assert "beta_hat" in result
    assert "beta_b" in result
    assert "se" in result
    assert len(result["beta_hat"]) == 3
    assert len(result["se"]) == 3
    assert result["n"] == 40
    assert result["p"] == 3
    assert result["B"] == 20


def test_btres_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = boot_residual_regression(X, y, B=10, seed=7, alpha=0.1, rescale=True)
    assert isinstance(result, dict)
    assert "beta_b" in result
    assert len(result["beta_b"]) == 10
    assert "resid" in result
    assert len(result["resid"]) == 40
