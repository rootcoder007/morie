# morie.fn -- function file (rootcoder007/morie)
"""MGWR AICc for model selection."""

from ._containers import SpatialResult


def mgwraic(ll, tr_S, n):
    r"""Corrected Akaike criterion of a (M)GWR fit from its Gaussian log-likelihood and ``tr S``.

    With ``ll = -n/2 (log(2 pi RSS/n) + 1)`` the (M)GWR AICc of Hurvich,
    Simonoff and Tsai (1998) as used by Fotheringham, Brunsdon and Charlton
    (2002, eq. 2.33) and MGWR, ``n log(RSS/n) + n log(2 pi) + n (n + tr S) /
    (n - 2 - tr S)``, equals ``-2 ll - n + n (n + tr S) / (n - 2 - tr S)``.

    References
    ----------
    Hurvich, C. M., Simonoff, J. S. and Tsai, C.-L. (1998). Smoothing
    parameter selection in nonparametric regression using an improved Akaike
    information criterion. *JRSS B* 60, 271-293.
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley.

    Examples
    --------
    >>> round(mgwraic(-20.0, 5.5, 40).statistic, 12)
    56.0
    """
    n = float(n)
    t = float(tr_S)
    if not t < n - 2.0:
        raise ValueError("tr_S must be below n - 2")
    val = -2.0 * float(ll) - n + n * (n + t) / (n - 2.0 - t)
    return SpatialResult(name="mgwraic", statistic=val, extra={"trS": t, "n": n})


mgwraic_fn = mgwraic


def cheatsheet() -> str:
    return "mgwraic(ll, tr_S, n) -> AICc = -2 ll - n + n (n + tr S)/(n - 2 - tr S)."
