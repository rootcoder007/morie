"""Tests for ghbvm.ghosal_bernstein_von_mises."""

from morie.fn import _array_core as np
from morie.fn.ghbvm import ghosal_bernstein_von_mises


def test_ghbvm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bernstein_von_mises(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghbvm_edge():
    """Test edge cases."""
    result = ghosal_bernstein_von_mises(np.array([42.0]))
    assert result["n"] == 1


def test_bayesian_bootstrap_draws_replayed():
    """theta_b = sum_i u_bi x_i with u_b ~ Dirichlet(1, ..., 1)."""
    import math

    import pytest

    from morie.fn import _array_core as np

    x = np.array([1.2, 0.4, 2.2, 1.8, 0.9, 1.5])
    rng = np.random.default_rng(6)
    th = [float(rng.dirichlet(np.ones(6)) @ x) for _ in range(40)]
    m = sum(th) / 40
    r = ghosal_bernstein_von_mises(x, B=40, seed=6)
    assert r["estimate"] == pytest.approx(m, rel=1e-13)
    assert r["se"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in th) / 39), rel=1e-11)
    assert r["theta_hat"] == pytest.approx(float(np.mean(x)), rel=1e-14)
