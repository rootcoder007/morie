"""Variance of the sum of two fair coin flips is 1/2.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.28).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["twocoinvar"]


def twocoinvar():
    """Variance of the sum of two fair coin flips is 1/2.

    Convolves two Bernoulli(1/2) pmfs and takes the variance of the
    resulting 0/1/2 distribution: mean 1, variance 1/2.

    Returns
    -------
    RichResult
        Keys: variance, mean.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.28).
    """
    values, probs = _morin.pmf_sum_convolution([0.0, 1.0], [0.5, 0.5],
                                               [0.0, 1.0], [0.5, 0.5])
    variance, mu = _morin.pmf_variance(values, probs)
    payload = {"variance": variance, "mean": mu}
    lines = [("Var(X+Y)", variance)]
    return RichResult(
        title="Variance of the sum of two fair coin flips is 1/2.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "twocoinvar: Var of the sum of two fair coin flips = 1/2. Morin (2016) eq (3.28)."
