"""Standard deviation of a sum of n i.i.d. variables: sqrt(n) sigma.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (1.71), (3.45).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["sdiidsum"]


def sdiidsum(sigma, n):
    """sigma_sum = sqrt(n) sigma for a sum of n i.i.d. variables.

    Parameters
    ----------
    sigma : float
        Per-variable standard deviation, >= 0.
    n : int
        Number of i.i.d. terms, >= 0.

    Returns
    -------
    RichResult
        Keys: sigma, n, sd_sum.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.71), (3.45).
    """
    value = _morin.sd_of_iid_sum(sigma, n)
    payload = {"sigma": float(sigma), "n": int(n), "sd_sum": value}
    lines = [("per-variable sigma", float(sigma)), ("sd of sum", value)]
    return RichResult(
        title="Standard deviation of a sum of n i.i.d. variables: sqrt(n) sigma.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sdiidsum: sigma_sum = sqrt(n) sigma. Morin (2016) eqs (1.71), (3.45)."
