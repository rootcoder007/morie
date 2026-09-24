"""Tests for hyper2.hyperparam_optim_gp."""

from morie.fn import _array_core as np

from morie.fn.hyper2 import hyperparam_optim_gp


def test_hyper2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    prior = [0.0, 0.0, -1.0]
    result = hyperparam_optim_gp(X, y, prior=prior, n_iter=10, seed=1)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_hyper2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (5, 2))
    y = rng.normal(0, 1, 5)
    result = hyperparam_optim_gp(X, y, n_iter=5, seed=1)
    assert isinstance(result, dict)
    assert len(result) > 0
