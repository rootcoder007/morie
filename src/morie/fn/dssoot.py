# morie.fn -- function file (rootcoder007/morie)
"""Percentile bootstrap confidence interval for the indirect effect ab.

Source opened: Preacher, K. J. and Hayes, A. F. (2008).  Asymptotic and
resampling strategies for assessing and comparing indirect effects in
multiple mediator models.  *Behavior Research Methods* 40(3), 879-891,
doi:10.3758/BRM.40.3.879.  The paper's argument is that the sampling
distribution of the product ab is skewed, so the Sobel test's normal
approximation is wrong in finite samples; the remedy is to resample
cases with replacement, recompute ab in each resample, and read the
interval off the empirical percentiles.

The two fitted equations are

    M = i_1 + a X + e_1
    Y = i_2 + c' X + b M + e_2

so ab is the indirect effect, c' the direct effect, and the total effect
c = c' + ab is an algebraic identity of least squares in the
single-mediator case -- the identity this module asserts as its anchor,
since it holds exactly and does not depend on the resampling.

The resampling stream is a deterministic linear congruential generator
(Numerical Recipes constants, modulus 2^32), NOT a system RNG: both
language arms must draw the same case indices or the interval endpoints
would only agree in distribution.
"""

from __future__ import annotations

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["bootstrap_indirect"]

_LCG_A = 1664525.0
_LCG_C = 1013904223.0
_LCG_M = 4294967296.0


def _ols2(y, x1, x2=None):
    """Least squares of y on an intercept, x1 and optionally x2."""
    n = len(y)
    Z = [[1.0, x1[i]] + ([x2[i]] if x2 is not None else []) for i in range(n)]
    return core.lstsq(Z, y)


def bootstrap_indirect(Y, X, M, n_boot=1000, alpha=0.05, seed=42):
    """Indirect effect ab with a percentile bootstrap interval.

    Parameters
    ----------
    Y : array-like
        Outcome.
    X : array-like
        Independent variable.
    M : array-like
        Mediator.
    n_boot : int
        Number of bootstrap resamples, at least 1.
    alpha : float
        Two-sided level of the percentile interval, in (0, 1).
    seed : int
        Seed of the deterministic congruential stream.

    Returns
    -------
    result : dict
        Keys: estimate (ab), a, b, c_prime, c_total, ci_lo, ci_hi,
        se_boot, bias, n_boot, alpha, n.

    References
    ----------
    Preacher & Hayes (2008), Behavior Research Methods 40(3):879-891,
    doi:10.3758/BRM.40.3.879.
    """
    yv = core.vec(Y)
    xv = core.vec(X)
    mv = core.vec(M)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: Y has no observations")
    if len(xv) != n or len(mv) != n:
        raise ValueError("Y, X and M must have the same length")
    B = int(n_boot)
    if B < 1:
        raise ValueError("n_boot must be at least 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    if n < 3:
        raise ValueError("need at least three observations to fit both paths")
    pa = _ols2(mv, xv)
    pb = _ols2(yv, xv, mv)
    pc = _ols2(yv, xv)
    a, cp, b = pa[1], pb[1], pb[2]
    ab = a * b
    state = float(int(seed) % int(_LCG_M))
    draws = []
    for _ in range(B):
        idx = []
        for _ in range(n):
            state = (_LCG_A * state + _LCG_C) % _LCG_M
            idx.append(int((state / _LCG_M) * n) % n)
        ys = [yv[i] for i in idx]
        xs = [xv[i] for i in idx]
        ms = [mv[i] for i in idx]
        qa = _ols2(ms, xs)
        qb = _ols2(ys, xs, ms)
        draws.append(qa[1] * qb[2])
    draws.sort()
    lo = core.quantile7(draws, alpha / 2.0)
    hi = core.quantile7(draws, 1.0 - alpha / 2.0)
    return RichResult(
        title="Bootstrap CI for the indirect effect",
        summary_lines=[("ab", ab), ("resamples", B)],
        payload={
            "estimate": ab,
            "a": a,
            "b": b,
            "c_prime": cp,
            "c_total": pc[1],
            "ci_lo": lo,
            "ci_hi": hi,
            "se_boot": core.sd(draws) if B > 1 else float("nan"),
            "bias": core.mean(draws) - ab,
            "n_boot": float(B),
            "alpha": alpha,
            "n": n,
            "method": "Bootstrap CI for indirect effect ab",
        },
    )


def cheatsheet():
    return "dssoot: Bootstrap CI for indirect effect ab"
