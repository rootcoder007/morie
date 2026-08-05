# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Genomic heritability from a relationship matrix.

VanRaden (2008), Journal of Dairy Science 91(11):4414-4423,
doi:10.3168/jds.2007-0980, for the genomic relationship matrix, with
the variance-component definition

    h^2 = sigma_g^2 / (sigma_g^2 + sigma_e^2).

Under y ~ N(mu 1, K sigma_g^2 + I sigma_e^2) the restricted
log-likelihood profiled over the total variance depends only on h^2,

    -0.5 [ log|V| + (n - 1) log(y' P y) ],   V = h^2 K + (1 - h^2) I,

so the estimate is found by evaluating that profile on a deterministic
grid and refining it by bisection on the grid maximum.  sigma_e^2 = 0
forces h^2 = 1 exactly, which is the degenerate case the tests use.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["heritability"]


def _reml(y, K, h2, n):
    V = [[h2 * K[i][j] + ((1.0 - h2) if i == j else 0.0) for j in range(n)] for i in range(n)]
    L = core.chol(V)
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    Vi_y = core.cholsolve(V, y)
    one = [1.0] * n
    Vi_1 = core.cholsolve(V, one)
    s11 = sum(Vi_1)
    s1y = sum(Vi_y)
    yPy = sum(y[i] * Vi_y[i] for i in range(n)) - s1y * s1y / s11
    return -0.5 * (logdet + math.log(s11) + (n - 1.0) * math.log(yPy)), yPy


def heritability(y, K, grid=41, refine=40):
    """REML estimate of h^2 on a deterministic grid, then bisection refinement."""
    yv = core.vec(y)
    Km = core.mat(K)
    n = len(yv)
    if n < 3:
        raise ValueError("heritability: need at least three observations")
    if len(Km) != n or len(Km[0]) != n:
        raise ValueError("heritability: K must be n x n")
    g = int(grid)
    if g < 3:
        raise ValueError("heritability: grid must have at least three points")
    lo, hi = 1e-6, 1.0 - 1e-6
    best = None
    bh = lo
    hs = []
    lls = []
    for i in range(g):
        h2 = lo + (hi - lo) * i / (g - 1.0)
        ll, _ = _reml(yv, Km, h2, n)
        hs.append(h2)
        lls.append(ll)
        if best is None or ll > best:
            best = ll
            bh = h2
    step = (hi - lo) / (g - 1.0)
    a = max(bh - step, lo)
    b = min(bh + step, hi)
    for _ in range(int(refine)):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        f1, _ = _reml(yv, Km, m1, n)
        f2, _ = _reml(yv, Km, m2, n)
        if f1 < f2:
            a = m1
        else:
            b = m2
    h2 = (a + b) / 2.0
    ll, yPy = _reml(yv, Km, h2, n)
    total = yPy / (n - 1.0)
    return RichResult(
        title="Genomic heritability",
        summary_lines=[("n", n), ("h2", h2)],
        payload={
            "estimate": h2,
            "h2": h2,
            "var_g": h2 * total,
            "var_e": (1.0 - h2) * total,
            "total_var": total,
            "loglik": ll,
            "grid_h2": hs,
            "grid_loglik": lls,
            "n": n,
            "method": "REML profile in h^2 over V = h^2 K + (1 - h^2) I, golden-section refined; VanRaden (2008) G",
        },
    )


def cheatsheet():
    return "hertbg: genomic heritability"
