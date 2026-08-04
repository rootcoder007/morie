"""Binomial variance E(k^2) - (np)^2 = npq.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.67).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["binomvar"]


def binomvar(n, p):
    """Binomial variance E(k^2) - (np)^2 = npq.

    Parameters
    ----------
    n : int
        Trials, >= 0.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: variance.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.67).
    """
    second = _morin.binomial_second_moment(n, p)
    mean = _morin.binomial_mean(n, p)
    value = second - mean ** 2
    direct = _morin.binomial_variance(n, p)
    if abs(value - direct) > 1e-9 * max(1.0, direct):
        raise AssertionError("moment identity disagrees with npq")
    payload = {"variance": value}
    lines = [("npq", value)]
    return RichResult(
        title="Binomial variance E(k^2) - (np)^2 = npq.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binomvar: Binomial variance npq. Morin (2016) eq (4.67)."
