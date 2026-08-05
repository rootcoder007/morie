# morie.fn -- function file (rootcoder007/morie)
"""Mantel-Haenszel pooled odds ratio."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_mantel_haenszel"]


def ma_mantel_haenszel(a, b, c, d, level=0.95):
    """Pool odds ratios across strata without modelling the strata.

    Fitting a stratum effect per table costs a parameter per table, and
    with sparse tables the maximum-likelihood estimate is badly biased --
    the classic Neyman-Scott problem.  The Mantel-Haenszel weights sidestep
    it entirely: they are the weights that make the pooled estimate
    consistent both when the strata are few and large and when they are
    many and small, which no likelihood-based weighting achieves at once.

    Formula: ``OR_MH = sum(a_i d_i/N_i) / sum(b_i c_i/N_i)``; the standard
    error of its logarithm is the Robins-Breslow-Greenland expression
    ``Var = sum(P R)/(2 (sum R)^2) + sum(P S + Q R)/(2 sum R sum S) +
    sum(Q S)/(2 (sum S)^2)`` with ``P = (a+d)/N``, ``Q = (b+c)/N``,
    ``R = ad/N``, ``S = bc/N`` -- Mantel & Haenszel (1959); Robins,
    Breslow & Greenland (1986).

    Parameters
    ----------
    a, b, c, d : array-like
        Per-stratum cells: exposed cases, exposed non-cases, unexposed
        cases, unexposed non-cases.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    RichResult
        ``OR_MH``, ``log_OR``, ``se_log``, ``ci`` (lower, upper), ``R``,
        ``S``, ``k``.

    References
    ----------
    Mantel, N. and Haenszel, W. (1959).  Statistical aspects of the
    analysis of data from retrospective studies of disease.  Journal of
    the National Cancer Institute 22(4):719-748.
    doi:10.1093/jnci/22.4.719.  Variance: Robins, J., Breslow, N. and
    Greenland, S. (1986).  Biometrics 42(2):311-323.
    doi:10.2307/2531052.
    """
    A = [float(t) for t in core.vec(a)]
    B = [float(t) for t in core.vec(b)]
    C = [float(t) for t in core.vec(c)]
    D = [float(t) for t in core.vec(d)]
    k = len(A)
    if k == 0:
        raise ValueError("no strata")
    if not (len(B) == len(C) == len(D) == k):
        raise ValueError("the four cell vectors must have equal length")
    if any(t < 0.0 for t in A + B + C + D):
        raise ValueError("cell counts must be non-negative")
    sR = sS = 0.0
    sPR = sPSQR = sQS = 0.0
    for i in range(k):
        n = A[i] + B[i] + C[i] + D[i]
        if n <= 0.0:
            raise ValueError("each stratum needs at least one observation")
        P = (A[i] + D[i]) / n
        Q = (B[i] + C[i]) / n
        R = A[i] * D[i] / n
        S = B[i] * C[i] / n
        sR += R
        sS += S
        sPR += P * R
        sPSQR += P * S + Q * R
        sQS += Q * S
    if sS <= 0.0 or sR <= 0.0:
        raise ValueError("the pooled odds ratio is not finite for these tables")
    orr = sR / sS
    v = (sPR / (2.0 * sR * sR) + sPSQR / (2.0 * sR * sS)
         + sQS / (2.0 * sS * sS))
    se = math.sqrt(v)
    lor = math.log(orr)
    z = core.qnorm(1.0 - (1.0 - float(level)) / 2.0)
    return RichResult(payload={
        "OR_MH": orr, "log_OR": lor, "se_log": se,
        "ci": [math.exp(lor - z * se), math.exp(lor + z * se)],
        "R": sR, "S": sS, "k": k,
        "method": "Mantel-Haenszel pooled odds ratio"})


def cheatsheet():
    return "mamh: Mantel-Haenszel pooled odds ratio with the RBG variance"
