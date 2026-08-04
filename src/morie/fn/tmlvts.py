# morie.fn -- function file (rootcoder007/morie)
"""Targeted estimate of the asymptotic variance itself."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmlevar", "tmle_var_targeting"]


def tmlevar(ic, level=0.95):
    """Target the variance parameter sigma^2 = E[D*^2] and bound it.

    The point is that the variance is a PARAMETER too.  The usual
    plug-in var(IC) is consistent but has its own sampling error, and
    a confidence interval built on it treats that error as zero.
    Targeting sigma^2 gives it its own influence curve, D*^2 -
    sigma^2, and therefore its own interval -- which is what lets a
    caller see when the standard error is itself unstable.

    The interval on sigma^2 is built on the LOG scale, because a
    variance is positive and a symmetric interval can otherwise reach
    below zero at the sample sizes where this matters.

    Formula: sigma^2 = E[D*(O)^2];  IC_{sigma^2}(O) = D*(O)^2 - sigma^2;
             se(sigma^2) = sqrt(var(D*^2)/n);
             se(psi) = sqrt(sigma^2/n)

    Parameters
    ----------
    ic : array-like
        Efficient influence-curve values D*(O_i), mean zero.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``sigma2``, ``se_sigma2``, ``ci_lower``, ``ci_upper``,
        ``se_psi``, ``se_psi_lower``, ``se_psi_upper``, ``kurtosis``,
        ``ic_mean``, ``n``.

    References
    ----------
    van der Laan, Hubbard and co-authors argue for treating the
    asymptotic variance as a target parameter in its own right; the
    row cites vdL-Hubbard-Pajouh (2018).  That paper was NOT
    obtainable, so what is implemented is the standard construction
    that follows directly from the definition: the influence curve of
    sigma^2 = E[D*^2] is D*^2 - sigma^2, and the delta method gives
    se(psi) = sqrt(sigma^2/n).  The plug-in variance it refines is the
    ``var.psi <- var(IC)/n`` of the CRAN package ``tmle`` 2.1.1
    (Gruber & van der Laan), which was fetched and read.
    """
    d = C.vec(ic)
    n = len(d)
    if n < 3:
        raise ValueError("at least three influence-curve values are required")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    m = sum(d) / n
    s2 = sum(v * v for v in d) / n
    if s2 <= 0.0:
        raise ValueError("the influence curve is identically zero")
    sq = [v * v for v in d]
    ses = math.sqrt(C.var(sq, 1) / n)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    # Log-scale interval: a variance is positive and the symmetric
    # interval can otherwise reach below zero.
    ls = ses / s2
    lo = s2 * math.exp(-z * ls)
    hi = s2 * math.exp(z * ls)
    m4 = sum(v ** 4 for v in d) / n
    return RichResult(payload={
        "sigma2": s2, "se_sigma2": ses, "ci_lower": lo, "ci_upper": hi,
        "se_psi": math.sqrt(s2 / n),
        "se_psi_lower": math.sqrt(lo / n),
        "se_psi_upper": math.sqrt(hi / n),
        "kurtosis": m4 / (s2 * s2), "ic_mean": m, "n": float(n),
        "method": "Variance targeting: sigma^2 = E[D*^2] with its own IC"})


tmle_var_targeting = tmlevar


def cheatsheet():
    return "tmlvts: IC of sigma^2 is D*^2 - sigma^2; log-scale interval"
