"""Population variance of a data set, with the computational identity.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.37), (3.60), (3.66).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["popvar"]


def popvar(x):
    """Population variance of a data set, with the computational identity.

    s-tilde^2 = (1/n) sum (xi - xbar)^2 (eqs 3.37, 3.60), reported
    beside the identity (1/n) sum (xi - xbar)^2 = mean(x^2) - xbar^2
    (eq 3.66) evaluated by its own route.

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
    """
    x_a = np.atleast_1d(np.asarray(x, dtype=float))
    lhs = _morin.population_variance(x_a)
    rhs = float(np.mean(x_a ** 2)) - float(np.mean(x_a)) ** 2
    payload = {
        "variance": lhs,
        "n": int(x_a.size),
        "lhs": lhs,
        "rhs": rhs,
        "identity_error": abs(lhs - rhs),
    }
    lines = [("s-tilde^2", lhs)]
    return RichResult(
        title="Population variance of a data set, with the computational identity.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "popvar: Population variance (1/n) sum (xi - xbar)^2. Morin (2016) eqs (3.37), (3.60), (3.66)."
