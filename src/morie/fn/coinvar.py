"""Variance of one fair coin flip (Heads = 1, Tails = 0) is 1/4.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.21).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["coinvar"]


def coinvar():
    """Variance of one fair coin flip (Heads = 1, Tails = 0) is 1/4.

    Returns
    -------
    RichResult
        Keys: variance, mean.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.21).
    """
    variance, mu = _morin.pmf_variance([0.0, 1.0], [0.5, 0.5])
    payload = {"variance": variance, "mean": mu}
    lines = [("variance", variance)]
    return RichResult(
        title="Variance of one fair coin flip (Heads = 1, Tails = 0) is 1/4.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "coinvar: Var of one fair coin flip = 1/4. Morin (2016) eq (3.21)."
