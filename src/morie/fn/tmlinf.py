# morie.fn -- function file (rootcoder007/morie)
"""Influence-curve inference for a targeted estimate."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmleinf", "tmle_inference"]


def tmleinf(psi, ic, level=0.95, null_value=0.0):
    """Standard error, interval and score check from an influence curve.

    The check worth running is ``score_solved``: an efficient influence
    curve must have empirical mean ZERO at the targeted estimate,
    because that is the equation targeting solves.  A mean that is not
    numerically zero means the fluctuation did not converge and the
    standard error below it is not trustworthy -- so it is returned
    rather than assumed.

    Formula: se = sqrt(var(IC)/n);  CI = psi -+ z_{alpha/2} se;
             the estimating equation is (1/n) sum IC_i = 0

    Parameters
    ----------
    psi : float
        The targeted point estimate.
    ic : array-like
        Influence-curve values, one per observation.
    level : float
        Confidence level.
    null_value : float
        Value tested against.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``statistic``, ``p_value``, ``ic_mean``, ``score_solved``
        (1 when |mean IC| < 1e-8 sd), ``n``.

    References
    ----------
    Verified against the reference implementation in the CRAN package
    ``tmle`` 2.1.1 (Gruber & van der Laan), whose ``calcParameters``
    computes ``var.psi <- var(IC)/n``,
    ``CI <- c(psi - mult*sqrt(var.psi), psi + mult*sqrt(var.psi))``
    with ``mult <- abs(qnorm(alpha.sig/2))`` and
    ``pvalue <- 2*pnorm(-abs(psi/sqrt(var.psi)))``.  Van der Laan's
    group's own software for van der Laan & Rose (2011), Targeted
    Learning.
    """
    ic = C.vec(ic)
    n = len(ic)
    if n < 2:
        raise ValueError("at least two influence-curve values are required")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    psi = float(psi)
    m = sum(ic) / n
    sd = C.sd(ic, 1)
    se = math.sqrt(C.var(ic, 1) / n)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    st = (psi - float(null_value)) / se if se > 0 else float("inf")
    return RichResult(payload={
        "estimate": psi, "se": se, "ci_lower": psi - z * se,
        "ci_upper": psi + z * se, "statistic": st,
        "p_value": 2.0 * (1.0 - C.pnorm(abs(st))) if se > 0 else 0.0,
        "ic_mean": m,
        "score_solved": 1.0 if abs(m) < 1e-8 * (sd if sd > 0 else 1.0) else 0.0,
        "n": float(n),
        "method": "Influence-curve inference for a targeted estimate"})


tmle_inference = tmleinf


def cheatsheet():
    return "tmlinf: se = sqrt(var(IC)/n); mean(IC) must be 0 or the fit did not target"
