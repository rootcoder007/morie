"""Tests for cvxlnf.boyd_l1_fitting."""

from morie.fn import _array_core as np

from morie.fn.cvxlnf import boyd_l1_fitting


def test_cvxlnf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, (10, 3))
    b = rng.normal(0, 1, 10)
    result = boyd_l1_fitting(A, b)
    assert isinstance(result, dict)
    assert "x" in result
    assert "residual" in result
    assert "l1_norm" in result
    assert "n_exact" in result
    assert "status" in result
    assert result["x"].shape == (3,)
    assert result["residual"].shape == (10,)
    x = result["x"]
    resid = A @ x - b
    expected_l1 = float(np.abs(resid).sum())
    assert abs(result["l1_norm"] - expected_l1) < 1e-6


def test_cvxlnf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, (10, 3))
    b = rng.normal(0, 1, 10)
    result = boyd_l1_fitting(A, b)
    assert isinstance(result, dict)
