"""Hypergeometric pmf against its binomial limit as the population grows.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (4.73), (4.75).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["hyperlim"]


def hyperlim(k, n, p, N):
    """Hypergeometric pmf against its binomial limit as the population grows.

    Draws n from a population of N of which K = round(pN) are
    successes; as N grows at fixed p the pmf tends to Binomial(n, p).

    Parameters
    ----------
    k : int
        Successes drawn, >= 0.
    n : int
        Draws, >= 0.
    p : float
        Success fraction in the population, in [0, 1].
    N : int
        Population size, >= n.

    Returns
    -------
    RichResult
        Keys: hypergeometric, binomial, abs_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (4.73), (4.75).
    """
    hyper, binom_p, err = _morin.hypergeometric_binomial_limit(k, n, p, N)
    payload = {"hypergeometric": hyper, "binomial": binom_p, "abs_error": err}
    lines = [("hypergeometric", hyper), ("binomial", binom_p)]
    return RichResult(
        title="Hypergeometric pmf against its binomial limit as the population grows.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "hyperlim: Hypergeometric pmf beside its binomial limit. Morin (2016) eqs (4.73), (4.75)."
