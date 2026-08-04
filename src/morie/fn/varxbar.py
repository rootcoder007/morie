"""Variance of the sample mean: sigma^2 / N.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.92).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["varxbar"]


def varxbar(sigma, N):
    """Variance of the sample mean: sigma^2 / N.

    Parameters
    ----------
    sigma : float
        Per-observation standard deviation, >= 0.
    N : int
        Sample size, >= 1.

    Returns
    -------
    RichResult
        Keys: var_mean.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.92).
    """
    value = _morin.var_of_sample_mean(sigma, N)
    payload = {"var_mean": value}
    lines = [("sigma^2/N", value)]
    return RichResult(
        title="Variance of the sample mean: sigma^2 / N.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "varxbar: Var(xbar) = sigma^2 / N. Morin (2016) eq (3.92)."
