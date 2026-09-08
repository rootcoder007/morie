"""sd of the sample mean sigma/sqrt(N) never exceeds sigma.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.93).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sdxbar"]


def sdxbar(sigma, N):
    """sd of the sample mean sigma/sqrt(N) never exceeds sigma.

    Parameters
    ----------
    sigma : float
        Per-observation standard deviation, >= 0.
    N : int
        Sample size, >= 1.

    Returns
    -------
    RichResult
        Keys: sd_mean, sigma, bounded.  ``bounded`` is the decision
        sd_mean <= sigma, which holds for every N >= 1.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.93).
    """
    var_mean = _morin.var_of_sample_mean(sigma, N)
    value = var_mean ** 0.5
    if value > float(sigma) + 1e-12:
        raise AssertionError("sd of the mean exceeded sigma")
    payload = {"sd_mean": value, "sigma": float(sigma), "bounded": True}
    lines = [("sigma/sqrt(N)", value)]
    return RichResult(
        title="sd of the sample mean sigma/sqrt(N) never exceeds sigma.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sdxbar: sd(xbar) = sigma/sqrt(N) <= sigma. Morin (2016) eq (3.93)."
