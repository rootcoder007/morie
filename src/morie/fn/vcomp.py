"""Variance-components estimation, ANOVA or REML (Searle et al. 1992)."""

import math

from ._richresult import RichResult
from .ranova import ranova
from .remlfn import remlfn

__all__ = ["vcomp", "variance_components"]


def _f_cdf(x, d1, d2):
    # F CDF via the regularized incomplete beta:
    # P(F <= x) = I_{d1 x / (d1 x + d2)}(d1/2, d2/2)
    from . import _stats_core as sc
    if x <= 0:
        return 0.0
    return sc._betainc(d1 / 2.0, d2 / 2.0, d1 * x / (d1 * x + d2))


def _f_ppf(p, d1, d2, iters=300):
    # monotone bisection on the CDF (same convention as the R arm)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    lo, hi = 0.0, 1.0
    while _f_cdf(hi, d1, d2) < p and hi < 1e12:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _f_cdf(mid, d1, d2) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vcomp(y, group, method="reml", conf_level=0.95):
    r"""
    Variance components for the one-way random model, with an ICC interval.

    Dispatches to the two estimators Searle, Casella & McCulloch
    (1992) treat as the primary methods for this model:

    * ``method="anova"`` -- the method-of-moments/ANOVA estimators,
      sigma_e^2 = MSE and sigma_a^2 = (MSA - MSE)/n_0 (their Eq. 21
      and Ch. 5), unbiased but able to go negative;
    * ``method="reml"`` (default) -- restricted maximum likelihood
      (their Sec. 6.6), which stays non-negative and accounts for
      the estimation of the fixed effect.  REML is the default
      because it is the estimator they carry forward for unbalanced
      data; on BALANCED data the two coincide exactly (their
      Sec. 4.8), so the choice only matters when the design is
      unbalanced.

    The exact confidence interval for the intraclass correlation
    rho = sigma_a^2 / (sigma_a^2 + sigma_e^2) under balanced data
    inverts the F pivot MSA/MSE ~ (1 + n rho/(1-rho)) F_{a-1, N-a}
    (Searle et al. Sec. 3.5, interval estimation for balanced data):
    with F = MSA/MSE and F_L = F / F_{1-alpha/2}, F_U = F /
    F_{alpha/2}, the limits are (F_L - 1)/(F_L - 1 + n) and
    (F_U - 1)/(F_U - 1 + n).  It is reported only when the data are
    balanced, since that is the case the pivot is exact for.

    Sources
    -------
    Searle, S. R., Casella, G. & McCulloch, C. E. (1992). *Variance
    Components*. Wiley: ANOVA estimators Eq. (21) and Chs. 4-5, REML
    Sec. 6.6, balanced-data equivalence Sec. 4.8, interval estimation
    Sec. 3.5 (local copy fetched-wave3/Variance_components_FULL.pdf).

    Parameters
    ----------
    y : sequence of float
        Observations.
    group : sequence
        Class label per observation.
    method : {"reml", "anova"}
        Estimator.
    conf_level : float
        Confidence level for the ICC interval (balanced data only).

    Returns
    -------
    RichResult
        Keys: sigma2_a, sigma2_e, icc, icc_lower, icc_upper (None if
        unbalanced), method_used, balanced, a, N, plus the underlying
        fit under "fit".
    """
    if method not in ("reml", "anova"):
        raise ValueError("method must be 'reml' or 'anova'")
    # Without this the interval silently divides by an F quantile of zero:
    # conf_level = 1.5 makes alpha negative, _f_ppf returns 0, and the
    # upper limit raises ZeroDivisionError instead of saying what is wrong.
    if not 0.0 < float(conf_level) < 1.0:
        raise ValueError("vcomp: conf_level must lie in (0, 1), got %r"
                         % (conf_level,))
    av = ranova(y, group)
    fit = remlfn(y, group) if method == "reml" else av
    s2a = float(fit["sigma2_a"])
    s2e = float(fit["sigma2_e"])
    denom = s2a + s2e
    icc = (s2a / denom) if denom > 0 else 0.0
    lo = hi = None
    if bool(av["balanced"]):
        n = av["n_i"][0]
        a = int(av["a"])
        N = int(av["N"])
        alpha = 1.0 - float(conf_level)
        F = av["msa"] / av["mse"] if av["mse"] > 0 else float("inf")
        f_hi = _f_ppf(1.0 - alpha / 2.0, a - 1, N - a)
        f_lo = _f_ppf(alpha / 2.0, a - 1, N - a)
        FL = F / f_hi
        FU = F / f_lo
        lo = (FL - 1.0) / (FL - 1.0 + n)
        hi = (FU - 1.0) / (FU - 1.0 + n)
        if lo < 0.0:
            lo = 0.0
        if hi > 1.0:
            hi = 1.0
    return RichResult(payload={
        "sigma2_a": s2a,
        "sigma2_e": s2e,
        "icc": icc,
        "icc_lower": lo,
        "icc_upper": hi,
        "method_used": method,
        "balanced": bool(av["balanced"]),
        "a": int(av["a"]),
        "N": int(av["N"]),
        "fit": dict(fit),
        "method": "variance components, %s (Searle et al. 1992)" % method,
    })


# long descriptive alias (stub-era name)
variance_components = vcomp


def cheatsheet():
    return ("vcomp: ANOVA or REML variance components + exact ICC F "
            "interval on balanced data (Searle Sec. 3.5)")

# public names resolved by fn/_lazy_map.json
variance_components_henderson3 = vcomp
