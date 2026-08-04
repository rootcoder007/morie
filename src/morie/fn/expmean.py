"""Exponential mean: the integral of t e^(-t/tau)/tau dt is tau.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.83).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["expmean"]


def expmean(tau):
    """Exponential mean: the integral of t e^(-t/tau)/tau dt is tau.

    Parameters
    ----------
    tau : float
        Mean waiting time, > 0.

    Returns
    -------
    RichResult
        Keys: mean.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.83).
    """
    mean, second, var = _morin.exponential_moments(tau)
    payload = {"mean": mean}
    lines = [("E(T)", mean)]
    return RichResult(
        title="Exponential mean: the integral of t e^(-t/tau)/tau dt is tau.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "expmean: E(T) = tau for an exponential waiting time. Morin (2016) eq (4.83)."
