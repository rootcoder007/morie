"""Binomial with p = a/n against its Poisson limit.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (4.34)-(4.35).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["binpoislim"]


def binpoislim(k, n, a):
    """Binomial with p = a/n against its Poisson limit.

    Parameters
    ----------
    k : int
        Successes, >= 0.
    n : int
        Trials, >= 1.
    a : float
        The product n p held fixed as n grows.

    Returns
    -------
    RichResult
        Keys: binomial, poisson, abs_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (4.34)-(4.35).
    """
    exact, limit, err = _morin.binomial_poisson_limit(k, n, a)
    payload = {"binomial": exact, "poisson": limit, "abs_error": err}
    lines = [("binomial", exact), ("Poisson limit", limit)]
    return RichResult(
        title="Binomial with p = a/n against its Poisson limit.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binpoislim: Binomial with p = a/n beside the Poisson pmf of mean a. Morin (2016) eqs (4.34)-(4.35)."
