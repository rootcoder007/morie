"""Population variance of a data set, with the computational identity.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.37), (3.60), (3.66).
"""

import math

from ._richresult import RichResult

__all__ = ["popvar"]


def popvar(x):
    """Population variance of a data set, with the computational identity.

    ``s~^2 = (1/n) sum (x_i - xbar)^2`` (eqs 3.37, 3.60; the ``1/n`` divisor,
    not Bessel's ``1/(n - 1)``), computed by the two-pass definition, and
    beside it the computational identity ``mean(x^2) - xbar^2`` (eq 3.66)
    evaluated by its own route; ``identity_error`` is their difference,
    which grows with ``xbar^2 / s~^2`` through cancellation (the reason the
    two-pass form is the one returned).

    Parameters
    ----------
    x : array-like
        Numeric data, non-empty.

    Returns
    -------
    RichResult
        Keys: variance, n, lhs, rhs, identity_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.37), (3.60), (3.66).

    Examples
    --------
    >>> popvar([1.0, 2.0, 4.0, 7.0])["variance"]
    5.25
    """
    v = [float(t) for t in (x.tolist() if hasattr(x, "tolist") else (x if hasattr(x, "__len__") else [x]))]
    n = len(v)
    if n == 0:
        raise ValueError("x must be non-empty")
    xbar = math.fsum(v) / n
    lhs = math.fsum((t - xbar) ** 2 for t in v) / n
    rhs = math.fsum(t * t for t in v) / n - xbar * xbar
    return RichResult(
        title="Population variance of a data set, with the computational identity.",
        summary_lines=[("s-tilde^2", lhs)],
        payload={"variance": lhs, "n": n, "lhs": lhs, "rhs": rhs, "identity_error": abs(lhs - rhs)},
    )


def cheatsheet():
    return "popvar: Population variance (1/n) sum (xi - xbar)^2. Morin (2016) eqs (3.37), (3.60), (3.66)."
