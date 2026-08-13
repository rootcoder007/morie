"""_array_core checked against closed-form expectations on the
primitives it covers.

Every input is a small fixed vector or matrix, so the expected outputs
are exact by hand (or by a one-line stdlib formula computed
independently of the implementation under test).
"""

import math
import statistics

import pytest

from morie.fn import _array_core as mnp


def close(a, b, tol=1e-12):
    av = a.tolist() if hasattr(a, "tolist") else a
    bv = b.tolist() if hasattr(b, "tolist") else b
    if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
        return av == pytest.approx(bv, rel=tol, abs=tol)

    def flat(v):
        out = []
        for e in v:
            out.extend(flat(e)) if isinstance(e, list) else out.append(e)
        return out
    fa, fb = flat(av), flat(bv)
    return len(fa) == len(fb) and fa == pytest.approx(fb, rel=tol,
                                                      abs=tol)


X = [2.0, 5.0, 3.0, 8.0, 4.0]
M = [[2.0, 1.0], [1.0, 3.0]]


class TestElementwise:
    def test_arithmetic(self):
        a = mnp.asarray(X)
        assert close(a + 2.0, [v + 2.0 for v in X])
        assert close(2.0 - a, [2.0 - v for v in X])
        assert close(a * a, [v * v for v in X])
        assert close(a / 4.0, [v / 4.0 for v in X])
        assert close(a ** 2, [v ** 2 for v in X])
        assert close(-a, [-v for v in X])

    def test_ufuncs(self):
        a = mnp.asarray(X)
        assert close(mnp.sqrt(a), [math.sqrt(v) for v in X])
        assert close(mnp.exp(a), [math.exp(v) for v in X])
        assert close(mnp.log(a), [math.log(v) for v in X])
        assert close(mnp.log1p(a), [math.log1p(v) for v in X])
        assert close(mnp.abs(-a), X)
        assert close(mnp.clip(a, 3.0, 6.0),
                     [min(max(v, 3.0), 6.0) for v in X])
        assert close(mnp.maximum(a, 4.0), [max(v, 4.0) for v in X])
        assert close(mnp.minimum(a, 4.0), [min(v, 4.0) for v in X])
        assert mnp.sqrt(4.0) == pytest.approx(2.0)   # scalar passthrough

    def test_where_and_masks(self):
        a = mnp.asarray(X)
        assert close(mnp.where(a > 3.0, a, 0.0),
                     [v if v > 3.0 else 0.0 for v in X])
        assert (a > 3.0).sum() == float(sum(1 for v in X if v > 3.0))


class TestReductions:
    def test_moments(self):
        a = mnp.asarray(X)
        n = len(X)
        mean = sum(X) / n
        var0 = sum((v - mean) ** 2 for v in X) / n
        assert a.sum() == pytest.approx(sum(X), rel=1e-15)
        assert a.mean() == pytest.approx(mean, rel=1e-15)
        assert a.std() == pytest.approx(math.sqrt(var0), rel=1e-14)
        assert a.std(ddof=1) == pytest.approx(statistics.stdev(X),
                                              rel=1e-14)
        assert a.var(ddof=1) == pytest.approx(statistics.variance(X),
                                              rel=1e-14)
        assert a.max() == max(X)
        assert a.min() == min(X)

    def test_sort_unique(self):
        assert close(mnp.sort([3.0, 1.0, 2.0]), [1.0, 2.0, 3.0])
        assert close(mnp.unique([3.0, 1.0, 3.0]), [1.0, 3.0])


class TestConstruction:
    def test_builders(self):
        assert close(mnp.arange(5), [0.0, 1.0, 2.0, 3.0, 4.0])
        assert close(mnp.arange(1, 7, 2), [1.0, 3.0, 5.0])
        assert close(mnp.zeros(3), [0.0, 0.0, 0.0])
        assert close(mnp.ones(3), [1.0, 1.0, 1.0])
        assert close(mnp.full(3, 2.5), [2.5, 2.5, 2.5])
        assert close(mnp.linspace(0, 1, 5),
                     [0.0, 0.25, 0.5, 0.75, 1.0])
        assert close(mnp.eye(3), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])
        assert close(mnp.diag([1.0, 2.0]), [[1.0, 0.0], [0.0, 2.0]])
        assert close(mnp.column_stack([[1.0, 2], [3.0, 4]]),
                     [[1.0, 3.0], [2.0, 4.0]])
        assert close(mnp.concatenate([[1.0], [2.0, 3.0]]),
                     [1.0, 2.0, 3.0])


class TestLinalg:
    def test_matmul_dot(self):
        a = mnp.asarray(M)
        v = mnp.asarray([1.0, 2.0])
        assert close(a @ v, [4.0, 7.0])
        assert close(a @ a, [[5.0, 5.0], [5.0, 10.0]])
        assert mnp.dot(v, v) == pytest.approx(5.0)
        assert close(a.T, [[2.0, 1.0], [1.0, 3.0]])

    def test_solve_inv_norm(self):
        # M x = [1, 2]: det = 5, x = (1/5, 3/5)
        assert close(mnp.linalg.solve(M, [1.0, 2.0]), [0.2, 0.6], 1e-12)
        assert close(mnp.linalg.inv(M),
                     [[0.6, -0.2], [-0.2, 0.4]], 1e-12)
        assert mnp.linalg.norm([3.0, 4.0]) == pytest.approx(5.0)
        with pytest.raises(ValueError):
            mnp.linalg.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 1.0])

    def test_lstsq(self):
        # perfect line y = 1 + 2x
        x = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        y = [1.0, 3.0, 5.0, 7.0]
        b_m, *_ = mnp.linalg.lstsq(x, y)
        assert close(b_m, [1.0, 2.0], 1e-10)

    def test_lstsq_matrix_rhs_solves_each_column(self):
        # A two-dimensional b holds one right-hand side per column.
        # This used to flatten b and read the first n entries of the
        # row-major order, mixing the columns together and returning a
        # single vector -- a wrong answer with no error raised.
        a = [[2.0, 1.0], [1.0, 3.0], [0.0, 1.0], [4.0, -1.0]]
        xt = [[1.0, -2.0], [0.5, 3.0]]
        b = [[sum(a[r][k] * xt[k][c] for k in range(2)) for c in range(2)]
             for r in range(4)]
        sol, *_ = mnp.linalg.lstsq(a, b, rcond=None)
        assert sol.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                assert sol[i][j] == pytest.approx(xt[i][j], abs=1e-10)
        # each column must equal the one-dimensional solve of that column
        for c in range(2):
            one, *_ = mnp.linalg.lstsq(a, [b[r][c] for r in range(4)],
                                       rcond=None)
            for i in range(2):
                assert sol[i][c] == pytest.approx(one[i], abs=1e-12)
        with pytest.raises(ValueError):
            mnp.linalg.lstsq(a, [[1.0, 2.0], [3.0, 4.0]], rcond=None)


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
        # a real shelf computation executed on the shim, checked
        # against the closed-form slope of the same data
        def ols(np_mod, x, y):
            xd = np_mod.asarray(x) - np_mod.mean(x)
            yd = np_mod.asarray(y) - np_mod.mean(y)
            return np_mod.dot(xd, yd) / np_mod.dot(xd, xd)

        x = [1.0, 2, 2, 3, 4, 5]
        y = [1.0, 0, 1, 3, 4, 7]
        xb, yb2 = sum(x) / len(x), sum(y) / len(y)
        num = sum((a - xb) * (b - yb2) for a, b in zip(x, y))
        den = sum((a - xb) ** 2 for a in x)
        assert ols(mnp, x, y) == pytest.approx(num / den, rel=1e-12)
