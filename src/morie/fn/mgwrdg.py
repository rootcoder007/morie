# morie.fn -- function file (rootcoder007/morie)
"""MGWR diagnostic summary."""

import math

from ._richresult import RichResult


def mgwrdg(ll, tr_S, n, k=None, alpha=0.05):
    r"""Information criteria and the multiple-testing level of an (M)GWR fit.

    From the Gaussian log-likelihood ``ll`` and ``tr S``: ``AIC = -2 ll + 2
    (tr S + 1)``, the (M)GWR ``AICc = -2 ll - n + n (n + tr S) / (n - 2 - tr
    S)``, ``BIC = -2 ll + (tr S + 1) log n`` (Fotheringham, Brunsdon and
    Charlton 2002, sec. 2.9), ``ENP = tr S``, residual degrees of freedom
    ``n - tr S`` and, with ``k`` covariates, the da Silva and Fotheringham
    (2016) corrected level ``alpha / (ENP / k)`` for the local t tests.

    References
    ----------
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley.
    da Silva, A. R. and Fotheringham, A. S. (2016). The multiple testing
    issue in geographically weighted regression. *Geographical Analysis* 48,
    233-247.

    Examples
    --------
    >>> r = mgwrdg(-20.0, 5.5, 40, k=3)
    >>> round(r["AICc"], 10), round(r["alpha_adjusted"], 12)
    (56.0, 0.027272727273)
    """
    n = float(n)
    t = float(tr_S)
    if not t < n - 2.0:
        raise ValueError("tr_S must be below n - 2")
    ll = float(ll)
    out = {
        "AIC": -2.0 * ll + 2.0 * (t + 1.0),
        "AICc": -2.0 * ll - n + n * (n + t) / (n - 2.0 - t),
        "BIC": -2.0 * ll + (t + 1.0) * math.log(n),
        "ENP": t,
        "df_residual": n - t,
    }
    if k is not None:
        out["alpha_adjusted"] = alpha / (t / float(k))
    return RichResult(payload=out)


mgwrdg_fn = mgwrdg


def cheatsheet() -> str:
    return "mgwrdg(ll, tr_S, n, k) -> AIC, AICc, BIC, ENP, residual df and da Silva-Fotheringham alpha."
