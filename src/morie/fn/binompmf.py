"""Binomial distribution P(k) = C(n,k) p^k (1-p)^(n-k).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (4.6), (4.60).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["binompmf"]


def binompmf(k, n, p):
    """Binomial distribution P(k) = C(n,k) p^k (1-p)^(n-k).

    Exact binomial coefficients up to n = 1000, log-gamma above it.

    Parameters
    ----------
    k : int
        Successes, >= 0.
    n : int
        Trials, >= 0.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: k, n, p, probability.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (4.6), (4.60).
    """
    value = _morin.binomial_pmf(k, n, p)
    payload = {"k": int(k), "n": int(n), "p": float(p), "probability": value}
    lines = [("P(k)", value)]
    return RichResult(
        title="Binomial distribution P(k) = C(n,k) p^k (1-p)^(n-k).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binompmf: Binomial pmf C(n,k) p^k q^(n-k). Morin (2016) eqs (4.6), (4.60)."
