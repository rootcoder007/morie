"""Odds ratio for a c-unit change when the variable is in an interaction, with its Wald interval.

Bilder & Loughin (2025), Analysis of Categorical Data with R, eqs (2.18) and (3.9).
"""

import math

from ._richresult import RichResult
from ._rrng_core import qnorm

__all__ = ["orint"]


def orint(b2, b3, x1, var2, var3, cov23, c=1.0, alpha=0.05):
    """OR = exp(c (b2 + b3 x1)) (2.18), with exp(c (b2 + b3 x1) +- c z sqrt(Var(b2 + b3 x1))) (3.9).

    Var(b2 + b3 x1) = Var(b2) + x1^2 Var(b3) + 2 x1 Cov(b2, b3).

    Parameters
    ----------
    b2, b3 : float
        Main-effect and interaction coefficients.
    x1 : float
        Level of the interacting variable.
    var2, var3, cov23 : float
        Estimated variances and covariance.
    c : float
        Size of the change in x2.
    alpha : float

    Returns
    -------
    RichResult
        Keys: odds_ratio, ci, se_log.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eqs (2.18), (3.9).

    Examples
    --------
    >>> round(orint(0.5, 0.1, 2.0, 0.01, 0.001, -0.002)["odds_ratio"], 12)
    2.01375270747
    """
    v = var2 + x1 * x1 * var3 + 2 * x1 * cov23
    if v < 0:
        raise ValueError("variance of b2 + b3 x1 is negative")
    est = c * (b2 + b3 * x1)
    z = qnorm(1 - alpha / 2)
    h = abs(c) * z * math.sqrt(v)
    return RichResult(
        title="Odds ratio with an interaction",
        summary_lines=[("OR", math.exp(est))],
        payload={
            "odds_ratio": math.exp(est),
            "ci": (math.exp(est - h), math.exp(est + h)),
            "se_log": abs(c) * math.sqrt(v),
        },
    )


def cheatsheet():
    return "orint: OR = exp(c(b2 + b3 x1)) with Wald CI. Bilder & Loughin eqs (2.18), (3.9)."
