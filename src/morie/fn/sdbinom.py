"""sigma of the number of successes in n biased trials: sqrt(npq).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.47), (3.56).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sdbinom"]


def sdbinom(n=10000, p=1.0 / 6.0):
    """sigma of the number of successes in n biased trials: sqrt(npq).

    The defaults are the book's worked dice total: n = 10,000 rolls at
    p = 1/6 gives sigma = 37.

    Parameters
    ----------
    n : int
        Number of trials, >= 0.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: n, p, sd, sd_tot (an alias of sd kept for eq (3.56)).

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.47), (3.56).
    """
    value = _morin.sd_binomial(n, p)
    payload = {"n": int(n), "p": float(p), "sd": value, "sd_tot": value}
    lines = [("sqrt(npq)", value)]
    return RichResult(
        title="sigma of the number of successes in n biased trials: sqrt(npq).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sdbinom: sigma of a binomial count = sqrt(npq). Morin (2016) eqs (3.47), (3.56)."
