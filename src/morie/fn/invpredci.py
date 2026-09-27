"""Confidence interval for the explanatory value giving a stated success probability (Fieller form).

Bilder & Loughin (2025), Analysis of Categorical Data with R, eq (2.23).
"""

import math

from ._richresult import RichResult
from ._rrng_core import qnorm

__all__ = ["invpredci"]


def invpredci(b0, b1, var0, var1, cov01, pi=0.5, alpha=0.05):
    """All x with |b0 + b1 x - logit(pi)| / sqrt(Var(b0) + x^2 Var(b1) + 2x Cov) < z_{1-alpha/2} (2.23).

    Squaring gives a x^2 + b x + c < 0 with a = b1^2 - z^2 Var(b1),
    b = 2 b1 (b0 - L) - 2 z^2 Cov, c = (b0 - L)^2 - z^2 Var(b0), L = logit(pi);
    when a > 0 and the discriminant is positive the set is the finite interval
    between the roots. The point estimate is x_pi = (L - b0)/b1.

    Parameters
    ----------
    b0, b1 : float
    var0, var1, cov01 : float
    pi : float
        Target probability (0.5 gives the median effective dose).
    alpha : float

    Returns
    -------
    RichResult
        Keys: estimate, ci (None if the set is not a bounded interval), bounded.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eq (2.23).

    Examples
    --------
    >>> invpredci(-4.0, 2.0, 0.25, 0.04, -0.09)["estimate"]
    2.0
    """
    if not 0 < pi < 1 or b1 == 0:
        raise ValueError("need 0 < pi < 1 and b1 != 0")
    L = math.log(pi / (1 - pi))
    z2 = qnorm(1 - alpha / 2) ** 2
    a = b1 * b1 - z2 * var1
    b = 2 * b1 * (b0 - L) - 2 * z2 * cov01
    c = (b0 - L) ** 2 - z2 * var0
    disc = b * b - 4 * a * c
    ci = None
    if a > 0 and disc > 0:
        r = math.sqrt(disc)
        ci = tuple(sorted(((-b - r) / (2 * a), (-b + r) / (2 * a))))
    return RichResult(
        title="Inverse-prediction interval",
        summary_lines=[("x_pi", (L - b0) / b1), ("ci", ci)],
        payload={"estimate": (L - b0) / b1, "ci": ci, "bounded": ci is not None},
    )


def cheatsheet():
    return "invpredci: Fieller-type CI for the x giving probability pi in a logistic model. Bilder & Loughin eq (2.23)."
