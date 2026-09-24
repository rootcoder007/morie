"""Tests for auglag.augmented_lagrangian."""

from morie.fn import _array_core as np

from morie.fn.auglag import augmented_lagrangian


def test_auglag_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    x0 = rng.normal(0, 1, n)

    def f(x):
        xa = np.array(x, dtype=float)
        total = 0.0
        for xi in xa:
            total = total + xi * xi
        return total

    def g(x):
        xa = np.array(x, dtype=float)
        s = 0.0
        for xi in xa:
            s = s + xi
        return np.array([s - 2.0])

    mu = 1.0
    lambda0 = np.zeros(1)

    result = augmented_lagrangian(f, g, x0, lambda0, mu)
    assert isinstance(result, dict)


def test_auglag_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 3
    x0 = rng.normal(0, 1, n)

    def f(x):
        xa = np.array(x, dtype=float)
        total = 0.0
        for xi in xa:
            total = total + xi * xi
        return total

    def g(x):
        xa = np.array(x, dtype=float)
        s = 0.0
        for xi in xa:
            s = s + xi
        return np.array([s - 1.0])

    mu = 0.5
    lambda0 = np.zeros(1)

    result = augmented_lagrangian(f, g, x0, lambda0, mu)
    assert isinstance(result, dict)
