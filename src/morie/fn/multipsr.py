# morie.fn -- function file (rootcoder007/morie)
"""Two-stage (multi-stage) sampling estimate and variance."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["twostage", "multi_stage_sampling"]


def twostage(Y, Nl, M, N, level=0.95):
    """Two-stage sample: PSUs drawn, then elements within each PSU.

    Two-stage sampling is what makes national surveys affordable, and
    its variance has the two terms you would expect: one for having
    sampled PSUs rather than taken them all, one for having sampled
    within them.  The second term carries the factor m/M, so it
    vanishes when every PSU is taken -- which is the check that the
    bookkeeping is right.

    Formula: Yhat = sum_l N_l ybar_l / sum_l N_l;
             V = (M/N)^2 [ ((M-m)/M) / (m(m-1)) * sum (N_l ybar_l - N_l Yhat)^2
                           + (1/(mM)) * sum N_l^2 ((N_l-n_l)/N_l) s_l^2 / n_l ]

    Parameters
    ----------
    Y : sequence of sequences
        Observations, one inner sequence per sampled PSU.
    Nl : array-like
        Population size of each sampled PSU, same length as Y.
    M : float
        Number of PSUs in the population.
    N : float
        Number of elements in the population.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``, ``psu_mean``,
        ``psu_var``, ``between_term``, ``within_term``, ``m``, ``M``,
        ``N``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 10
    (subsampling / two-stage sampling).  Chapter 10 was NOT in the
    scanned excerpt available to this batch, so the estimator and
    variance are taken from the reference implementation in the CRAN
    package ``samplingbook`` 1.2.4, function ``submean`` with
    ``method = "ratio"``, which computes exactly the expression above.
    """
    m = len(Y)
    if m < 2:
        raise ValueError("at least two PSUs are needed for a variance")
    Nl = C.vec(Nl)
    if len(Nl) != m:
        raise ValueError("Nl must have one entry per sampled PSU")
    M = float(M)
    N = float(N)
    if M < m:
        raise ValueError("M must be at least the number of sampled PSUs")
    yb = []
    s2 = []
    nl = []
    for i in range(m):
        yi = C.vec(Y[i])
        k = len(yi)
        if k < 2:
            raise ValueError("every sampled PSU needs at least two elements")
        if k > Nl[i]:
            raise ValueError("a PSU sample cannot exceed its population")
        nl.append(k)
        yb.append(sum(yi) / k)
        s2.append(C.var(yi, 1))
    tot = sum(Nl[i] * yb[i] for i in range(m))
    est = tot / sum(Nl)
    f1 = (M - m) / M
    VB = sum((Nl[i] * yb[i] - Nl[i] * est) ** 2 for i in range(m))
    between = f1 / (m * (m - 1)) * VB
    within = sum(Nl[i] ** 2 * ((Nl[i] - nl[i]) / Nl[i]) * s2[i] / nl[i]
                 for i in range(m)) / (m * M)
    var = (M / N) ** 2 * (between + within)
    se = math.sqrt(var)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    return RichResult(payload={
        "estimate": est, "se": se, "ci_lower": est - z * se,
        "ci_upper": est + z * se, "psu_mean": yb, "psu_var": s2,
        "between_term": (M / N) ** 2 * between,
        "within_term": (M / N) ** 2 * within, "m": m, "M": M, "N": N,
        "method": "Two-stage sampling, ratio form"})


multi_stage_sampling = twostage


def cheatsheet():
    return "multipsr: two-stage ratio estimate; V = between + within terms"
