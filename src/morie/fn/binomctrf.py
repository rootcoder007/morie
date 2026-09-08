"""Factorial form of the centered binomial: (2n)!/((n+x)!(n-x)! 2^(2n)).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (5.5).
"""

import math

from . import _morin

from ._richresult import RichResult

__all__ = ["binomctrf"]


def binomctrf(x, n):
    """Factorial form of the centered binomial: (2n)!/((n+x)!(n-x)! 2^(2n)).

    Cross-checked against the C(2n, n+x)/2^(2n) form.

    Parameters
    ----------
    x : int
        Deviation from n Heads; zero outside |x| <= n.
    n : int
        Half the number of flips, >= 0.

    Returns
    -------
    RichResult
        Keys: probability.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (5.5).
    """
    x_i, n_i = int(x), int(n)
    if abs(x_i) > n_i:
        value = 0.0
    else:
        value = (math.factorial(2 * n_i)
                 / (math.factorial(n_i + x_i) * math.factorial(n_i - x_i))
                 / 4.0 ** n_i)
    check = _morin.binomial_centered_pmf(x_i, n_i)
    if abs(value - check) > 1e-12 * max(1.0, check):
        raise AssertionError("factorial form disagrees with C(2n, n+x)/2^2n")
    payload = {"probability": value}
    lines = [("PB(x)", value)]
    return RichResult(
        title="Factorial form of the centered binomial: (2n)!/((n+x)!(n-x)! 2^(2n)).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binomctrf: Factorial form of the centred binomial. Morin (2016) eq (5.5)."
