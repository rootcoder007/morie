"""sigma of a sum of independent variables: sqrt(sum sigma_i^2).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.42)-(3.43).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sdsum"]


def sdsum(sigmas):
    """sigma of a sum of independent variables: sqrt(sum sigma_i^2).

    Parameters
    ----------
    sigmas : array-like
        Per-variable standard deviations, each >= 0.

    Returns
    -------
    RichResult
        Keys: sd_sum.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.42)-(3.43).
    """
    value = _morin.sd_sum_independent(sigmas)
    payload = {"sd_sum": value}
    lines = [("sigma_sum", value)]
    return RichResult(
        title="sigma of a sum of independent variables: sqrt(sum sigma_i^2).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sdsum: sigma_sum = sqrt(sum sigma_i^2) for independent terms. Morin (2016) eqs (3.42)-(3.43)."
