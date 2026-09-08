"""Poisson mean: the series sum k P(k) equals a.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.92).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poismean"]


def poismean(a):
    """Poisson mean: the series sum k P(k) equals a.

    Parameters
    ----------
    a : float
        Expected count, >= 0.

    Returns
    -------
    RichResult
        Keys: mean.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.92).
    """
    mean, var = _morin.poisson_mean_var(a)
    payload = {"mean": mean}
    lines = [("sum k P(k)", mean)]
    return RichResult(
        title="Poisson mean: the series sum k P(k) equals a.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poismean: Poisson mean from the series sum k P(k). Morin (2016) eq (4.92)."
