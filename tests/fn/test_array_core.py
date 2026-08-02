"""Equivalence tests: _array_core vs numpy on the primitives it covers.

The contract of the de-numpy shim is drop-in numerical agreement with
numpy on the call patterns morie.fn actually uses; every check here
compares the two implementations directly.
"""

import math

real_np = __import__("pytest").importorskip("numpy")
import pytest

from morie.fn import _array_core as mnp


def close(a, b, tol=1e-12):
    a = a.tolist() if hasattr(a, "tolist") else a
    b = b.tolist() if hasattr(b, "tolist") else b
    return real_np.allclose(real_np.asarray(a), real_np.asarray(b),
                            rtol=tol, atol=tol)


X = [2.0, 5.0, 3.0, 8.0, 4.0]
M = [[2.0, 1.0], [1.0, 3.0]]


class TestElementwise:
    def test_arithmetic(self):
        a = mnp.asarray(X)
        na = real_np.asarray(X)
        assert close(a + 2.0, na + 2.0)
        assert close(2.0 - a, 2.0 - na)
        assert close(a * a, na * na)
        assert close(a / 4.0, na / 4.0)
        assert close(a ** 2, na ** 2)
        assert close(-a, -na)

    def test_ufuncs(self):
        a = mnp.asarray(X)
        na = real_np.asarray(X)
        assert close(mnp.sqrt(a), real_np.sqrt(na))
        assert close(mnp.exp(a), real_np.exp(na))
        assert close(mnp.log(a), real_np.log(na))
        assert close(mnp.log1p(a), real_np.log1p(na))
        assert close(mnp.abs(-a), real_np.abs(-na))
        assert close(mnp.clip(a, 3.0, 6.0), real_np.clip(na, 3.0, 6.0))
        assert close(mnp.maximum(a, 4.0), real_np.maximum(na, 4.0))
        assert close(mnp.minimum(a, 4.0), real_np.minimum(na, 4.0))
        assert mnp.sqrt(4.0) == pytest.approx(2.0)   # scalar passthrough

    def test_where_and_masks(self):
        a = mnp.asarray(X)
        na = real_np.asarray(X)
        assert close(mnp.where(a > 3.0, a, 0.0),
                     real_np.where(na > 3.0, na, 0.0))
        assert (a > 3.0).sum() == float((na > 3.0).sum())


class TestReductions:
    def test_moments(self):
        a = mnp.asarray(X)
        na = real_np.asarray(X)
        assert a.sum() == pytest.approx(float(na.sum()), rel=1e-15)
        assert a.mean() == pytest.approx(float(na.mean()), rel=1e-15)
        assert a.std() == pytest.approx(float(na.std()), rel=1e-14)
        assert a.std(ddof=1) == pytest.approx(float(na.std(ddof=1)),
                                              rel=1e-14)
        assert a.var(ddof=1) == pytest.approx(float(na.var(ddof=1)),
                                              rel=1e-14)
        assert a.max() == float(na.max())
        assert a.min() == float(na.min())

    def test_sort_unique(self):
        assert close(mnp.sort([3.0, 1.0, 2.0]), real_np.sort([3.0, 1, 2]))
        assert close(mnp.unique([3.0, 1.0, 3.0]),
                     real_np.unique([3.0, 1.0, 3.0]))


class TestConstruction:
    def test_builders(self):
        assert close(mnp.arange(5), real_np.arange(5))
        assert close(mnp.arange(1, 7, 2), real_np.arange(1, 7, 2))
        assert close(mnp.zeros(3), real_np.zeros(3))
        assert close(mnp.ones(3), real_np.ones(3))
        assert close(mnp.full(3, 2.5), real_np.full(3, 2.5))
        assert close(mnp.linspace(0, 1, 5), real_np.linspace(0, 1, 5))
        assert close(mnp.eye(3), real_np.eye(3))
        assert close(mnp.diag([1.0, 2.0]), real_np.diag([1.0, 2.0]))
        assert close(mnp.column_stack([[1.0, 2], [3.0, 4]]),
                     real_np.column_stack([[1.0, 2], [3.0, 4]]))
        assert close(mnp.concatenate([[1.0], [2.0, 3.0]]),
                     real_np.concatenate([[1.0], [2.0, 3.0]]))


class TestLinalg:
    def test_matmul_dot(self):
        a = mnp.asarray(M)
        na = real_np.asarray(M)
        v = mnp.asarray([1.0, 2.0])
        nv = real_np.asarray([1.0, 2.0])
        assert close(a @ v, na @ nv)
        assert close(a @ a, na @ na)
        assert mnp.dot(v, v) == pytest.approx(float(nv @ nv))
        assert close(a.T, na.T)

    def test_solve_inv_norm(self):
        assert close(mnp.linalg.solve(M, [1.0, 2.0]),
                     real_np.linalg.solve(M, [1.0, 2.0]), 1e-12)
        assert close(mnp.linalg.inv(M), real_np.linalg.inv(M), 1e-12)
        assert mnp.linalg.norm([3.0, 4.0]) == pytest.approx(5.0)
        with pytest.raises(ValueError):
            mnp.linalg.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 1.0])

    def test_lstsq(self):
        x = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        y = [1.0, 3.0, 5.0, 7.0]
        b_m, *_ = mnp.linalg.lstsq(x, y)
        b_n, *_ = real_np.linalg.lstsq(real_np.asarray(x),
                                       real_np.asarray(y), rcond=None)
        assert close(b_m, b_n, 1e-10)


class TestRandom:
    def test_deterministic_and_distributed(self):
        rng = mnp.random.default_rng(42)
        rng2 = mnp.random.default_rng(42)
        a = rng.normal(0, 1, 4000)
        b = rng2.normal(0, 1, 4000)
        assert close(a, b)                        # reproducible
        assert a.mean() == pytest.approx(0.0, abs=0.06)
        assert a.std() == pytest.approx(1.0, abs=0.05)
        u = mnp.random.default_rng(7).uniform(0, 1, 4000)
        assert 0.0 <= u.min() and u.max() <= 1.0
        assert u.mean() == pytest.approx(0.5, abs=0.03)
        ints = mnp.random.default_rng(3).integers(0, 10, 100)
        assert ints.min() >= 0 and ints.max() < 10


class TestDropIn:
    def test_backend_function_runs_on_shim(self):
        # a real shelf computation executed on the shim, compared to numpy
        def ols(np_mod, x, y):
            xd = np_mod.asarray(x) - np_mod.mean(x)
            yd = np_mod.asarray(y) - np_mod.mean(y)
            b1 = np_mod.dot(xd, yd) / np_mod.dot(xd, xd)
            return b1

        x = [1.0, 2, 2, 3, 4, 5]
        y = [1.0, 0, 1, 3, 4, 7]
        assert ols(mnp, x, y) == pytest.approx(
            float(ols(real_np, real_np.asarray(x), real_np.asarray(y))),
            rel=1e-12)
