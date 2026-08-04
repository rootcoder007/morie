"""Variance of a fair k-sided die roll.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.20).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["dievar"]


def dievar(sides=6):
    """Variance of a fair k-sided die roll.

    For a uniform roll on 1..k the mean is (k+1)/2 and the variance is
    (k^2 - 1)/12; the book's worked six-sided value is 2.92.

    Parameters
    ----------
    sides : int
        Number of faces, >= 1.

    Returns
    -------
    RichResult
        Keys: variance, mean, sides.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.20).
    """
    k = int(sides)
    if k < 1:
        raise ValueError("sides must be >= 1")
    values = np.arange(1, k + 1, dtype=float)
    probs = np.full(k, 1.0 / k)
    variance, mu = _morin.pmf_variance(values, probs)
    payload = {"variance": variance, "mean": mu, "sides": k}
    lines = [("mean", mu), ("variance", variance)]
    return RichResult(
        title="Variance of a fair k-sided die roll.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "dievar: Variance of a fair k-sided die: (k^2-1)/12. Morin (2016) eq (3.20)."
