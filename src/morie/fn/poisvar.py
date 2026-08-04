"""Poisson variance: the series E(k^2) - a^2 equals a.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.94).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poisvar"]


def poisvar(a):
    """Poisson variance: the series E(k^2) - a^2 equals a.

    Parameters
    ----------
    a : float
        Expected count, >= 0.

    Returns
    -------
    RichResult
        Keys: variance.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.94).
    """
    mean, var = _morin.poisson_mean_var(a)
    payload = {"variance": var}
    lines = [("variance", var)]
    return RichResult(
        title="Poisson variance: the series E(k^2) - a^2 equals a.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poisvar: Poisson variance from the series E(k^2) - a^2. Morin (2016) eq (4.94)."
