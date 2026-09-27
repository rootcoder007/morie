"""Exact (Garwood) interval for a Poisson mean from n observations.

Bilder & Loughin (2025), Analysis of Categorical Data with R, Sec 4.1.
"""

from ._richresult import RichResult
from ._rrng_core import qchisq

__all__ = ["poisexactci"]


def poisexactci(total, n=1, alpha=0.05):
    """chi2_{2 n mu_hat, alpha/2}/(2n) < mu < chi2_{2(n mu_hat + 1), 1 - alpha/2}/(2n).

    n mu_hat is the total count; the lower limit is 0 when it is 0.

    Parameters
    ----------
    total : int
        Total count over the n observations.
    n : float
        Number of observations (or exposure).
    alpha : float

    Returns
    -------
    RichResult
        Keys: estimate (total/n), ci.

    References
    ----------
    Garwood, F. (1936). Biometrika 28, 437-442.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Sec 4.1.

    Examples
    --------
    >>> poisexactci(0, 5)["ci"][0]
    0.0
    """
    if total < 0 or int(total) != total or n <= 0:
        raise ValueError("total must be a non-negative integer and n > 0")
    lo = 0.0 if total == 0 else qchisq(alpha / 2, 2 * total) / (2 * n)
    hi = qchisq(1 - alpha / 2, 2 * (total + 1)) / (2 * n)
    return RichResult(
        title="Exact Poisson interval",
        summary_lines=[("ci", (lo, hi))],
        payload={"estimate": total / n, "ci": (lo, hi)},
    )


def cheatsheet():
    return "poisexactci: Garwood exact Poisson CI via chi-square quantiles. Bilder & Loughin Sec 4.1."
