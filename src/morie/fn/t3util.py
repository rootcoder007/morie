# morie.fn -- tail3 batch shared helpers (rootcoder007/morie)
"""Deterministic numerical helpers shared by the tail3 (long-tail) batch.

Nothing here is a published method.  These are plain utilities kept out of
the shared cores so the batch owns its own helpers, and written so the R
mirror in ``R/t3util.R`` performs the identical arithmetic.
"""

from __future__ import annotations

from . import _array_core as np

__all__ = ["golden", "gaussnodes", "bfsdist", "relu", "expit", "ols", "chi2sf", "normcdf"]

_INVPHI = 0.6180339887498949


def golden(f, lo, hi, iters=80):
    """Golden-section minimiser with a fixed iteration count."""
    a = float(lo)
    b = float(hi)
    c = b - _INVPHI * (b - a)
    d = a + _INVPHI * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(int(iters)):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - _INVPHI * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + _INVPHI * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def gaussnodes(m=401, lim=8.0):
    """Trapezoid nodes/weights for integrating against the N(0,1) density."""
    m = int(m)
    u = np.linspace(-float(lim), float(lim), m)
    w = np.exp(-0.5 * u * u)
    w = w / float(np.sum(w))
    return u, w


def bfsdist(A, s):
    """Unweighted breadth-first distances from node ``s`` (inf if unreachable)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    d = [float("inf")] * n
    d[int(s)] = 0.0
    frontier = [int(s)]
    while frontier:
        nxt = []
        for i in frontier:
            for j in range(n):
                if A[i, j] != 0.0 and d[j] == float("inf"):
                    d[j] = d[i] + 1.0
                    nxt.append(j)
        frontier = nxt
    return np.asarray(d, dtype=float)


def relu(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0.0, x, 0.0)


def expit(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def ols(X, y):
    """Least squares; ``X`` must already carry any intercept column."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    xtx = X.T @ X
    xty = X.T @ y
    try:
        beta = np.linalg.solve(xtx, xty)
    except Exception:
        beta = np.linalg.pinv(xtx) @ xty
    return np.asarray(beta, dtype=float).ravel()


def normcdf(z):
    """Standard normal CDF via the complementary error function."""
    from . import _stats_core as stats

    return stats.norm.cdf(z)


def chi2sf(x, df):
    """Upper tail of the chi-square distribution."""
    from . import _stats_core as stats

    return float(stats.chi2.sf(float(x), int(df)))
