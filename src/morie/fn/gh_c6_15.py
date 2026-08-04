# morie.fn -- function file (rootcoder007/morie)
"""Martingale route to posterior consistency."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["martcons", "ghosal_martg_consist"]


def martcons(dh2, variances=None):
    """Cesaro-average Hellinger discrepancy of the predictive densities.

    The martingale approach earns its place by needing NO tests at all:
    the predictive-to-truth discrepancies form a martingale, and the
    strong law for martingale differences turns their Cesaro average
    into an almost-sure statement.  What is checked here is the
    conclusion, n^-1 sum d_H^2(phat_i, p_0) -> 0, plus the summability
    condition sum n^-2 var_0 < infinity that Lemma 6.52 requires -- and
    which is automatic for the square-root discrepancy.

    Formula: n^-1 sum_{i=1}^{n} d_H^2(phat_i, p_0) -> 0 a.s.;
             Lemma 6.52 needs sum_{n>=1} n^-2 var_0(Psi(phat_n/phat_{0,n})(X_n)) < inf

    Parameters
    ----------
    dh2 : array-like
        Squared Hellinger distances d_H^2(phat_i, p_0), i = 1..n,
        each in [0, 1].
    variances : array-like, optional
        var_0 of the martingale differences, same length; when given,
        the Lemma 6.52 partial sum is returned.

    Returns
    -------
    RichResult
        ``cesaro`` (the running average at n), ``final``, ``tail_mean``
        (average over the last half, which is what actually has to go
        to zero), ``lemma652_sum`` (nan if variances is None),
        ``summable``, ``n``.

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Section 6.8.4 (Martingale Approach), equations
    (6.17) and (6.18) for the compensators -K(phat_{0,i}; phat_i) and
    -d_H^2(phat_{0,i}; phat_i), and Lemma 6.52: "If sum_{n=1}^{inf}
    n^{-2} var_0(Psi(phat_n/phat_{0,n})(X_n)) < inf, then n^{-1} M_n ->
    0 almost surely ... this implies that n^{-1} sum_{i=1}^{n}
    d_H^2(phat_i, p_0) -> 0, almost surely".  The approach is due to
    Walker (2003, 2004) as the book's historical notes record.  Read
    from the copy of the book held in the corpus.
    """
    d = C.vec(dh2)
    n = len(d)
    if n < 1:
        raise ValueError("at least one discrepancy is required")
    if any(v < 0.0 or v > 1.0 for v in d):
        raise ValueError("squared Hellinger distances must lie in [0, 1]")
    run = C.cumsum(d)
    ces = [run[i] / (i + 1) for i in range(n)]
    half = n // 2
    tail = sum(d[half:]) / (n - half)
    if variances is None:
        ls = float("nan")
        sm = float("nan")
    else:
        v = C.vec(variances)
        if len(v) != n:
            raise ValueError("variances must have the same length as dh2")
        if any(x < 0.0 for x in v):
            raise ValueError("variances must be non-negative")
        ls = sum(v[i] / ((i + 1) ** 2) for i in range(n))
        sm = 1.0 if ls < float("inf") else 0.0
    return RichResult(payload={
        "cesaro": ces, "final": ces[-1], "tail_mean": tail,
        "lemma652_sum": ls, "summable": sm, "n": float(n),
        "method": "Martingale consistency check, Ghosal Section 6.8.4"})


ghosal_martg_consist = martcons


def cheatsheet():
    return "gh_c6_15: n^-1 sum d_H^2(phat_i, p0) -> 0; Lemma 6.52 sum n^-2 var"
