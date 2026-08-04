"""Unbiased sample variance s^2 with the n-1 denominator.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.73).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sampvar"]


def sampvar(x):
    """Unbiased sample variance s^2 with the n-1 denominator.

    Parameters
    ----------
    x : array-like
        Numeric data, n >= 2.

    Returns
    -------
    RichResult
        Keys: sample_variance, population_variance.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.73).
    """
    value = _morin.sample_variance(x)
    payload = {
        "sample_variance": value,
        "population_variance": _morin.population_variance(x),
    }
    lines = [("s^2", value)]
    return RichResult(
        title="Unbiased sample variance s^2 with the n-1 denominator.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sampvar: Unbiased sample variance with the n-1 denominator. Morin (2016) eq (3.73)."
