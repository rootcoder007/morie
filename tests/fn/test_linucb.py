"""Tests for linucb.linucb."""

from morie.fn import _array_core as np

from morie.fn.linucb import linucb


def test_linucb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = 3
    n_arms = 5

    x = rng.normal(0, 1, p)
    theta = rng.normal(0, 1, (n_arms, p))
    Ainv = [np.eye(p) for _ in range(n_arms)]
    alpha = 0.5

    result = linucb(x, theta, Ainv, alpha)
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_linucb_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    p = 3
    n_arms = 2

    x = rng.normal(0, 1, p)
    theta = rng.normal(0, 1, (n_arms, p))
    Ainv = [np.eye(p) for _ in range(n_arms)]

    result = linucb(x, theta, Ainv)
    assert isinstance(result, dict)
    assert len(result) >= 1
