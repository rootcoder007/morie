"""Tests for tmlpse.tmle_path_specific (path-specific effect, linear SEM)."""

import math
import statistics

import pytest

from morie.fn.tmlpse import tmle_path_specific


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _inv(A):
    n = len(A)
    cols = [_solve(A, [1.0 if r == c else 0.0 for r in range(n)]) for c in range(n)]
    return [[cols[c][r] for c in range(n)] for r in range(n)]


def _ols(Z, t):
    """Normal equations, and the influence functions (Z'Z/n)^{-1} z_i e_i."""
    n, q = len(Z), len(Z[0])
    ZtZ = [[sum(z[r] * z[c] for z in Z) for c in range(q)] for r in range(q)]
    b = _solve(ZtZ, [sum(z[r] * y for z, y in zip(Z, t)) for r in range(q)])
    Ai = _inv([[v / n for v in row] for row in ZtZ])
    e = [y - sum(x * c for x, c in zip(z, b)) for z, y in zip(Z, t)]
    return b, [[sum(Ai[r][c] * Z[i][c] for c in range(q)) * e[i] for r in range(q)] for i in range(n)]


def _data(n=60):
    X = [[math.sin(k)] for k in range(n)]
    D = [1.0 if math.cos(3 * k) > 0 else 0.0 for k in range(n)]
    M1 = [0.5 * d + 0.3 * x[0] + 0.1 * math.sin(5 * k) for k, (d, x) in enumerate(zip(D, X))]
    M2 = [0.4 * d + 0.6 * m + 0.1 * math.cos(7 * k) for k, (m, d) in enumerate(zip(M1, D))]
    y = [1 + 0.2 * d + 0.7 * a + 0.5 * b + 0.3 * x[0] + 0.2 * math.sin(11 * k)
         for k, (d, a, b, x) in enumerate(zip(D, M1, M2, X))]
    return y, D, [[a, b] for a, b in zip(M1, M2)], X


def _reference(y, D, M, X, path):
    """Fit the three linear models; psi is the mean of Q(1, W, M*) -
    Q(0, W, M0) with M* and M0 simulated forward through the mediator
    models at the path-assigned treatments.  psi is affine in every
    single coefficient, so a central difference gives its gradient
    exactly (up to rounding); the IC is gradient x OLS influence."""
    n = len(y)
    W = [[1.0] + x for x in X]
    Z1 = [[d] + w for d, w in zip(D, W)]
    Z2 = [[d] + w + [m[0]] for d, w, m in zip(D, W, M)]
    Zy = [[d] + w + m for d, w, m in zip(D, W, M)]
    b1, i1 = _ols(Z1, [m[0] for m in M])
    b2, i2 = _ols(Z2, [m[1] for m in M])
    by, iy = _ols(Zy, y)
    sizes = [len(b1), len(b2), len(by)]
    theta = b1 + b2 + by

    def psi(th):
        c1, c2, cy = th[:sizes[0]], th[sizes[0]:sizes[0] + sizes[1]], th[sizes[0] + sizes[1]:]
        tot = 0.0
        for w in W:
            def chain(a1, a2):
                m1 = sum(u * v for u, v in zip([a1] + w, c1))
                m2 = sum(u * v for u, v in zip([a2] + w + [m1], c2))
                return m1, m2
            s1, s2 = chain(float(path[0]), float(path[1]))
            z1, z2 = chain(0.0, 0.0)
            tot += sum(u * v for u, v in zip([1.0] + w + [s1, s2], cy))
            tot -= sum(u * v for u, v in zip([0.0] + w + [z1, z2], cy))
        return tot / n

    h = 1e-4
    grad = []
    for j in range(len(theta)):
        tp, tm = theta[:], theta[:]
        tp[j] += h
        tm[j] -= h
        grad.append((psi(tp) - psi(tm)) / (2 * h))
    inf = [a + b + c for a, b, c in zip(i1, i2, iy)]
    ic = [sum(g * f for g, f in zip(grad, row)) for row in inf]
    return psi(theta), math.sqrt(statistics.variance(ic) / n)


def test_tmlpse_basic():
    """Both path choices match the simulated plug-in and its delta-method
    standard error.  Tolerance 1e-8 on se: the central difference is
    exact for an affine function, leaving rounding of order
    1e-16 / 1e-4 = 1e-12 relative in each gradient entry."""
    y, D, M, X = _data()
    for path in ([1, 1], [0, 0], [1, 0], [0, 1]):
        psi, se = _reference(y, D, M, X, path)
        r = tmle_path_specific(y, D, M, X, path)
        assert r["estimate"] == pytest.approx(psi, abs=1e-9)
        assert r["se"] == pytest.approx(se, rel=1e-8)
        assert r["n_path"] == float(sum(path))


def test_tmlpse_edge():
    """The all-paths contrast recovers the generating total effect
    0.2 + 0.7*0.5 + 0.5*(0.4 + 0.6*0.5) = 0.9 closely on this nearly
    noise-free design; mismatched lengths raise."""
    y, D, M, X = _data()
    assert tmle_path_specific(y, D, M, X, [1, 1])["estimate"] == pytest.approx(0.9, abs=0.05)
    with pytest.raises(ValueError):
        tmle_path_specific(y, D, M, X, [1])
    with pytest.raises(ValueError):
        tmle_path_specific(y[:-1], D, M, X, [1, 1])
