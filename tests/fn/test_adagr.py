"""Tests for morie.fn.adagr -- Adagrad optimizer."""

from morie.fn import _array_core as np
from morie.fn.adagr import adagr


def test_quadratic_minimization():
    """Adagrad should reduce f(x) = sum(x^2) toward 0."""

    def f(x):
        return np.sum(x**2)

    def g(x):
        return 2.0 * x

    x0 = np.array([5.0, 5.0])
    result = adagr(f, g, x0, learning_rate=0.5, max_iter=2000)
    assert isinstance(result, np.ndarray)
    assert np.sum(result**2) < np.sum(x0**2)


def test_returns_ndarray():
    def f(x):
        return x[0] ** 2

    def g(x):
        return np.array([2.0 * x[0]])

    result = adagr(f, g, np.array([3.0]))
    assert isinstance(result, np.ndarray)


def test_full_output():
    def f(x):
        return x[0] ** 2

    def g(x):
        return np.array([2.0 * x[0]])

    x, info = adagr(f, g, np.array([3.0]), full_output=True)
    assert isinstance(x, np.ndarray)
    assert "iterations" in info
