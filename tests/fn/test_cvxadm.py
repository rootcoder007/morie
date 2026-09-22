"""Tests for cvxadm.boyd_admm."""

from morie.fn import _array_core as np

from morie.fn.cvxadm import boyd_admm


def test_cvxadm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    f = lambda v, r: v
    g = lambda v, r: v
    A = rng.normal(0, 1, (10, 10))
    B = rng.normal(0, 1, (10, 10))
    c = rng.normal(0, 1, 10)
    rho = 0.5
    result = boyd_admm(f, g, A, B, c, rho, n=10)
    assert isinstance(result, dict)
    assert "x" in result
    assert "z" in result
    assert "u" in result
    assert "n_iter" in result
    assert "converged" in result
    assert "primal_residual" in result
    assert "dual_residual" in result
    assert "residual_path" in result


def test_cvxadm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    f = lambda v, r: v
    g = lambda v, r: v
    A = rng.normal(0, 1, (10, 10))
    B = rng.normal(0, 1, (10, 10))
    c = rng.normal(0, 1, 10)
    rho = 0.5
    result = boyd_admm(f, g, A, B, c, rho, n=10)
    assert isinstance(result, dict)
