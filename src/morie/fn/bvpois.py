"""Bivariate Poisson distribution (trivariate reduction).

Holgate, P. (1964). Estimation for the bivariate Poisson distribution. Biometrika 51, 241-245.
"""

import math

from ._richresult import RichResult

__all__ = ["bvpois"]


def bvpois(x, y, a=1.0, b=1.0, c=0.0):
    r"""X = U + W, Y = V + W with independent Poisson U (a), V (b), W (c):

    P(X = x, Y = y) = exp(-(a + b + c)) a^x b^y / (x! y!) sum_{k=0}^{min(x, y)} C(x, k) C(y, k) k! (c/(a b))^k,
    evaluated term by term in log space; Cov(X, Y) = c.

    Parameters
    ----------
    x, y : int
    a, b : float
        Rates, > 0.
    c : float
        Common rate, >= 0.

    Returns
    -------
    RichResult
        Keys: pmf, cov.

    References
    ----------
    Holgate, P. (1964). Biometrika 51, 241-245.
    Matches ``extraDistr::dbvpois(x, y, a, b, c)``.

    Examples
    --------
    >>> round(bvpois(0, 0, 1.0, 2.0, 0.5)["pmf"], 12)
    0.030197383422
    """
    if not (a > 0 and b > 0 and c >= 0) or x < 0 or y < 0 or int(x) != x or int(y) != y:
        raise ValueError("need a, b > 0, c >= 0 and non-negative integer x, y")
    x, y = int(x), int(y)
    base = -(a + b + c) + x * math.log(a) + y * math.log(b) - math.lgamma(x + 1) - math.lgamma(y + 1)
    s = 0.0
    for k in range(min(x, y) + 1):
        if c == 0 and k > 0:
            break
        lt = (
            math.lgamma(x + 1)
            - math.lgamma(k + 1)
            - math.lgamma(x - k + 1)
            + math.lgamma(y + 1)
            - math.lgamma(k + 1)
            - math.lgamma(y - k + 1)
            + math.lgamma(k + 1)
            + (k * math.log(c / (a * b)) if k else 0.0)
        )
        s += math.exp(lt)
    return RichResult(
        title="Bivariate Poisson",
        summary_lines=[("pmf", math.exp(base) * s)],
        payload={"pmf": math.exp(base) * s, "cov": c},
    )


def cheatsheet():
    return "bvpois: bivariate Poisson pmf by trivariate reduction."
