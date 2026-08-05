# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Pairs (case) bootstrap for OLS: resample (x_i, y_i) jointly.

Freedman, D. A. (1981), "Bootstrapping Regression Models", *The Annals
of Statistics* 9(6), 1218-1228, doi:10.1214/aos/1176345638 (verified
against Crossref).

Freedman's paper contains both regression bootstraps and is explicit
about when each is legitimate.  Resampling residuals conditions on X and
therefore assumes the errors are iid; resampling whole cases treats the
rows as an iid draw from a joint distribution and so survives
heteroskedasticity and a stochastic design, at the price of not
conditioning on X.  This module is the case version:

    draw i*_1, ..., i*_n uniformly with replacement from 1..n,
    refit OLS on (X[i*], y[i*]), collect beta*.

Because the design is redrawn, a resample can be rank deficient; the fit
is the package's ridge-stabilised normal-equation solve, so such a
resample yields a shrunken rather than an undefined coefficient, and the
count of near-singular resamples is reported in ``n_illcond``.

Anchor: on a noiseless fixture y = X beta the OLS fit of every resample
with full column rank recovers beta exactly, so the replicate spread is
zero -- a property no amount of resampling machinery can fake.
"""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_pairs_regression"]


def boot_pairs_regression(X, y, B=200, seed=1, alpha=0.05):
    """Case-resampling bootstrap for the OLS coefficient vector.

    Parameters
    ----------
    X : array-like
        The n x p design, intercept column included if wanted.
    y : array-like
        The n responses.
    B : int
        Replicates.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate for the percentile interval.

    Returns
    -------
    RichResult
        ``beta_b`` (B x p), ``beta_hat``, ``se`` (per coefficient),
        ``lo``/``hi`` (percentile interval per coefficient),
        ``n_illcond``, ``n``, ``p``, ``B``.
    """
    from . import _tail1core as C

    Xm = core.mat(X)
    yy = core.vec(y)
    n = core.nrow(Xm)
    p = core.ncol(Xm)
    if n != len(yy):
        raise ValueError("boot_pairs_regression: X and y have different lengths")
    if n <= p:
        raise ValueError("boot_pairs_regression: need more rows than columns")
    if int(B) < 2:
        raise ValueError("boot_pairs_regression: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_pairs_regression: alpha must lie strictly between 0 and 1")
    bh = core.lstsq(Xm, yy)
    g = C.Lcg(seed)
    reps = []
    ill = 0
    for _ in range(int(B)):
        idx = []
        for _i in range(n):
            j = int(g.unif() * n)
            if j >= n:
                j = n - 1
            idx.append(j)
        Xs = [Xm[j] for j in idx]
        ys = [yy[j] for j in idx]
        seen = len(set(idx))
        if seen < p:
            ill += 1
        reps.append(core.lstsq(Xs, ys))
    se = []
    lo = []
    hi = []
    for j in range(p):
        col = [r[j] for r in reps]
        se.append(core.sd(col, 1))
        lo.append(core.quantile7(col, a / 2.0))
        hi.append(core.quantile7(col, 1.0 - a / 2.0))
    return RichResult(
        title="Pairs (case) bootstrap for OLS",
        summary_lines=[("n", n), ("p", p), ("B", int(B))],
        payload={
            "beta_b": reps,
            "beta_hat": bh,
            "se": se,
            "lo": lo,
            "hi": hi,
            "n_illcond": ill,
            "n": n,
            "p": p,
            "B": int(B),
            "estimate": bh[0],
            "method": "Freedman (1981) Ann. Statist. 9(6):1218-1228, case resampling",
        },
    )


def cheatsheet():
    return "btpair: resample whole rows; survives heteroskedasticity but does not condition on X"
