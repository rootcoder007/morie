"""Waiting-time interval probability: e^(-lambda t) lambda dt.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (4.23), (4.25).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["expintp"]


def expintp(t, dt, lam):
    """Waiting-time interval probability: e^(-lambda t) lambda dt.

    P(the next event falls in [t, t + dt]) factorises into surviving
    to t and then firing in dt.

    Parameters
    ----------
    t : float
        Elapsed time, >= 0.
    dt : float
        Interval width, >= 0.
    lam : float
        Rate, > 0.

    Returns
    -------
    RichResult
        Keys: probability.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (4.23), (4.25).
    """
    value = _morin.exponential_interval_probability(t, dt, lam)
    payload = {"probability": value}
    lines = [("P(t, dt)", value)]
    return RichResult(
        title="Waiting-time interval probability: e^(-lambda t) lambda dt.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "expintp: P(next event in [t, t+dt]) = e^(-lambda t) lambda dt. Morin (2016) eqs (4.23), (4.25)."
