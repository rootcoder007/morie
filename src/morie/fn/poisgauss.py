"""Gaussian limit of the Poisson: e^(-(k-a)^2/2a)/sqrt(2 pi a).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (5.23).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["poisgauss"]


def poisgauss(k, a):
    """Gaussian limit of the Poisson: e^(-(k-a)^2/2a)/sqrt(2 pi a).

    Parameters
    ----------
    k : float
        Event count (may be non-integral for the continuous form).
    a : float
        Expected count, > 0.

    Returns
    -------
    RichResult
        Keys: PG, exact.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (5.23).
    """
    value = _morin.poisson_gaussian(k, a)
    exact = _morin.poisson_pmf(int(round(float(k))), a) if float(k) >= 0 else 0.0
    payload = {"PG": value, "exact": exact}
    lines = [("PG(k)", value)]
    return RichResult(
        title="Gaussian limit of the Poisson: e^(-(k-a)^2/2a)/sqrt(2 pi a).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "poisgauss: Gaussian limit of the Poisson pmf. Morin (2016) eq (5.23)."
