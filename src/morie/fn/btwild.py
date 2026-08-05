# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Wild bootstrap for OLS under heteroskedasticity (Mammen two-point).

Mammen, E. (1993), "Bootstrap and Wild Bootstrap for High Dimensional
Linear Models", *The Annals of Statistics* 21(1), 255-285,
doi:10.1214/aos/1176349025 (verified against Crossref).

The design and the residuals are both held fixed; only a scalar
multiplier is redrawn per observation:

    y*_i = x_i' beta_hat + r_i v_i,

with v_i iid, mean 0, variance 1, third moment 1.  Mammen's two-point
law attains all three:

    v = (1 - sqrt 5)/2  with probability (sqrt 5 + 1)/(2 sqrt 5),
    v = (1 + sqrt 5)/2  otherwise,

which is the law the package's shared ``_s03core.mammen`` /
``.s03mammen`` helper encodes; here the point is drawn from the shared
Lehmer stream rather than a low-discrepancy sequence, because a single
van der Corput stream shared across the n positions of a replicate makes
the multipliers within a replicate deterministically dependent.  The
Rademacher alternative (+/-1 with probability 1/2) is available via
``weights="rademacher"``; it has third moment 0 and so loses the skewness
correction, but is symmetric.

Anchor, and it is exact rather than asymptotic: because Var*(v) = 1 and
the multipliers are independent across i,

    Var*(beta*) = (X'X)^{-1} ( sum_i r_i^2 x_i x_i' ) (X'X)^{-1},

which is precisely the HC0 heteroskedasticity-robust sandwich.  The wild
bootstrap standard error is therefore not merely "robust"; it targets the
HC0 number, and ``var_hc0`` reports that target computed directly from
the sandwich, never through the resampling loop.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult
from .btres import _xtxinv

__all__ = ["boot_wild_regression"]

_R5 = math.sqrt(5.0)
_MAMMEN_P = (_R5 + 1.0) / (2.0 * _R5)


def _mult(u, kind):
    if kind == "rademacher":
        return 1.0 if u < 0.5 else -1.0
    return (1.0 - _R5) / 2.0 if u < _MAMMEN_P else (1.0 + _R5) / 2.0


def boot_wild_regression(X, y, B=200, seed=1, alpha=0.05, weights="mammen"):
    """Wild bootstrap replicates of the OLS coefficient vector.

    Parameters
    ----------
    X : array-like
        The n x p design.
    y : array-like
        The n responses.
    B : int
        Replicates.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate.
    weights : {"mammen", "rademacher"}
        Multiplier law.

    Returns
    -------
    RichResult
        ``beta_b``, ``beta_hat``, ``se``, ``lo``/``hi``, ``var_hc0``
        (the HC0 sandwich diagonal the wild bootstrap targets),
        ``v_mean``/``v_var``/``v_m3`` (realised multiplier moments),
        ``n``, ``p``, ``B``.
    """
    from . import _tail1core as C

    if weights not in ("mammen", "rademacher"):
        raise ValueError("boot_wild_regression: weights must be 'mammen' or 'rademacher'")
    Xm = core.mat(X)
    yy = core.vec(y)
    n = core.nrow(Xm)
    p = core.ncol(Xm)
    if n != len(yy):
        raise ValueError("boot_wild_regression: X and y have different lengths")
    if n <= p:
        raise ValueError("boot_wild_regression: need more rows than columns")
    if int(B) < 2:
        raise ValueError("boot_wild_regression: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_wild_regression: alpha must lie strictly between 0 and 1")
    bh = core.lstsq(Xm, yy)
    fit = [sum(Xm[i][j] * bh[j] for j in range(p)) for i in range(n)]
    res = [yy[i] - fit[i] for i in range(n)]
    XtXinv = _xtxinv(Xm, n, p)
    # HC0 meat: sum_i r_i^2 x_i x_i'
    meat = [[0.0] * p for _ in range(p)]
    for i in range(n):
        w = res[i] * res[i]
        for j in range(p):
            for k in range(p):
                meat[j][k] += w * Xm[i][j] * Xm[i][k]
    mid = [[sum(XtXinv[j][t] * meat[t][k] for t in range(p)) for k in range(p)] for j in range(p)]
    hc0 = [sum(mid[j][t] * XtXinv[t][j] for t in range(p)) for j in range(p)]
    g = C.Lcg(seed)
    reps = []
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    N = 0
    for _ in range(int(B)):
        ys = []
        for i in range(n):
            v = _mult(g.unif(), weights)
            s1 += v
            s2 += v * v
            s3 += v * v * v
            N += 1
            ys.append(fit[i] + res[i] * v)
        reps.append(core.lstsq(Xm, ys))
    se = []
    lo = []
    hi = []
    for j in range(p):
        col = [r[j] for r in reps]
        se.append(core.sd(col, 1))
        lo.append(core.quantile7(col, a / 2.0))
        hi.append(core.quantile7(col, 1.0 - a / 2.0))
    vm = s1 / N
    return RichResult(
        title="Wild bootstrap (Mammen 1993)",
        summary_lines=[("n", n), ("p", p), ("B", int(B)), ("weights", weights)],
        payload={
            "beta_b": reps,
            "beta_hat": bh,
            "se": se,
            "lo": lo,
            "hi": hi,
            "var_hc0": hc0,
            "v_mean": vm,
            "v_var": s2 / N - vm * vm,
            "v_m3": s3 / N,
            "n": n,
            "p": p,
            "B": int(B),
            "estimate": bh[0],
            "method": "Mammen (1993) Ann. Statist. 21(1):255-285, two-point multiplier",
        },
    )


def cheatsheet():
    return "btwild: y* = xb + r_i v_i, v two-point mean 0 var 1 skew 1; Var*(beta*) IS the HC0 sandwich"
