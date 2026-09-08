# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap confidence interval for a targeted estimate."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["tmleboot", "tmle_bootstrap_ci"]


def tmleboot(Y, A, QAW, Q1W, Q0W, g1W, B=200, seed=1, gbound=0.025,
             level=0.95):
    """Nonparametric bootstrap interval for the targeted risk difference.

    Note what is and is not resampled: the INITIAL FITS are carried
    along as fixed columns, so this bootstraps the targeting step and
    the empirical means, not the machine learning that produced Q and
    g.  A full bootstrap would refit those inside every replicate; this
    one therefore UNDERSTATES uncertainty when the initial fits are
    themselves adaptive, and that is stated rather than glossed.

    The influence-curve interval is returned beside it, because the
    two agreeing is the reassurance and the two disagreeing is the
    finding.

    Formula: resample indices with replacement, re-target each
             replicate, take the empirical 2.5th and 97.5th
             percentiles of psi*

    Parameters
    ----------
    Y, A : array-like
        Outcome in [0, 1] and binary treatment.
    QAW, Q1W, Q0W : array-like
        Initial outcome predictions.
    g1W : array-like
        Initial propensity.
    B : int
        Fixed number of bootstrap replicates.
    seed : int
        Seed for the pinned generator.
    gbound : float
        Propensity truncation level.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``boot_se``, ``ci_lower``, ``ci_upper``,
        ``ic_se``, ``ic_lower``, ``ic_upper``, ``boot_mean``,
        ``B``, ``n``.

    References
    ----------
    Verified against the reference implementation in the CRAN package
    ``tmle`` 2.1.1 (Gruber & van der Laan) for the targeting step and
    the influence-curve interval it is compared with.  The bootstrap
    is the ordinary nonparametric one of Efron (1979), Bootstrap
    methods: another look at the jackknife, Annals of Statistics 7(1),
    1-26; van der Laan & Rose (2011), Targeted Learning, recommend the
    influence-curve interval as the default and the bootstrap as a
    check.
    """
    Y = C.vec(Y)
    A = C.vec(A)
    n = len(Y)
    QAW = C.vec(QAW)
    Q1W = C.vec(Q1W)
    Q0W = C.vec(Q0W)
    g1W = C.vec(g1W)
    if any(len(v) != n for v in (A, QAW, Q1W, Q0W, g1W)):
        raise ValueError("every argument must have one entry per observation")
    B = int(B)
    if B < 2:
        raise ValueError("B must be at least 2")
    if n < 2:
        raise ValueError("at least two observations are required")
    fit = T.target(Y, A, QAW, Q1W, Q0W, g1W, gbound)
    mu1, mu0, ic1, ic0 = T.curves(Y, A, fit)
    psi = mu1 - mu0
    ic = [ic1[i] - ic0[i] for i in range(n)]
    icse = math.sqrt(C.var(ic, 1) / n)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    g = C.Lcg(seed)
    reps = []
    for _ in range(B):
        idx = []
        for _ in range(n):
            j = int(g.unif() * n)
            if j >= n:
                j = n - 1
            idx.append(j)
        if len(set(A[j] for j in idx)) < 2:
            # A replicate with only one treatment arm cannot be
            # targeted; it is skipped rather than silently contributing
            # a degenerate estimate.
            continue
        f = T.target([Y[j] for j in idx], [A[j] for j in idx],
                     [QAW[j] for j in idx], [Q1W[j] for j in idx],
                     [Q0W[j] for j in idx], [g1W[j] for j in idx], gbound)
        reps.append(sum(f["Q1star"]) / n - sum(f["Q0star"]) / n)
    if len(reps) < 2:
        raise ValueError("too few usable bootstrap replicates")
    q = sorted(reps)
    m = len(q)
    a = (1.0 - float(level)) / 2.0
    lo = q[max(0, int(math.floor(a * (m - 1))))]
    hi = q[min(m - 1, int(math.ceil((1.0 - a) * (m - 1))))]
    return RichResult(payload={
        "estimate": psi, "boot_se": C.sd(reps, 1), "ci_lower": lo,
        "ci_upper": hi, "ic_se": icse, "ic_lower": psi - z * icse,
        "ic_upper": psi + z * icse, "boot_mean": sum(reps) / m,
        "B": float(m), "n": float(n),
        "method": "Bootstrap and influence-curve intervals for a TMLE"})


tmle_bootstrap_ci = tmleboot


def cheatsheet():
    return "tmlboo: resample + re-target; compare with the influence-curve CI"
