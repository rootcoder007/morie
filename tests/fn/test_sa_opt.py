"""Tests for sa_opt.simulated_annealing."""

from morie.fn import _array_core as np
from morie.fn.sa_opt import simulated_annealing


def test_sa_opt_basic():
    """Test basic functionality."""

    def fun(*a, **k):
        return float(np.sum(np.asarray(a[0]) ** 2))

    x0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = simulated_annealing(fun, x0)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sa_opt_edge():
    """Test edge cases."""

    def fun(*a, **k):
        return float(np.sum(np.asarray(a[0]) ** 2))

    x0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = simulated_annealing(fun, x0)
    assert isinstance(result, dict)


def test_annealing_trajectory_replayed():
    """Philox proposals (stream 0) and Metropolis uniforms (stream 1), geometric cooling."""
    import math

    import pytest

    from morie.fn._rng import random_normal, random_uniform

    def f(v):
        return (v[0] - 1.0) ** 2 + 0.5 * v[1] ** 2

    n, K, T0, a, st = 2, 25, 1.0, 0.9, 0.5
    Z = [float(v) for v in random_normal(K * n, seed=3, stream=0)]
    U = [float(v) for v in random_uniform(K, seed=3, stream=1)]
    x, fx = [0.0, 0.0], f([0.0, 0.0])
    best = (list(x), fx)
    for k in range(1, K + 1):
        T = T0 * a**k
        p = [x[j] + st * Z[(k - 1) * n + j] for j in range(n)]
        fp = f(p)
        if fp - fx <= 0 or U[k - 1] < math.exp(-(fp - fx) / T):
            x, fx = p, fp
            if fx < best[1]:
                best = (list(x), fx)
    r = simulated_annealing(f, [0.0, 0.0], step=st, T0=T0, n_iter=K, alpha=a, seed=3)
    assert r["x"] == pytest.approx(best[0], rel=1e-14)
    assert r["final_x"] == pytest.approx(x, rel=1e-14)
