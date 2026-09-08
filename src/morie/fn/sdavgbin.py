"""sigma of a single trial, of the total, and of the average of n trials.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.57)-(3.58).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["sdavgbin"]


def sdavgbin(n=10000, p=1.0 / 6.0):
    """sigma of a single trial, of the total, and of the average of n trials.

    sigma_single = sqrt(pq), sigma_tot = sqrt(npq), sigma_avg =
    sigma_tot / n, cross-checked against sigma_single / sqrt(n).  The
    defaults are the book's worked dice average, sigma_avg = 0.0037.

    Parameters
    ----------
    n : int
        Number of trials, >= 1.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: sd_single, sd_tot, sd_avg.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.57)-(3.58).
    """
    sd_single = _morin.sd_bernoulli(p)
    sd_tot = _morin.sd_binomial(n, p)
    sd_avg = sd_tot / int(n)
    check = _morin.sd_of_mean(sd_single, n)
    if abs(sd_avg - check) > 1e-12:
        raise AssertionError("sigma_tot/n != sigma_single/sqrt(n)")
    payload = {"sd_single": sd_single, "sd_tot": sd_tot, "sd_avg": sd_avg}
    lines = [("sigma_single", sd_single), ("sigma_tot", sd_tot),
             ("sigma_avg", sd_avg)]
    return RichResult(
        title="sigma of a single trial, of the total, and of the average of n trials.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "sdavgbin: sigma_single, sigma_tot and sigma_avg for n Bernoulli trials. Morin (2016) eqs (3.57)-(3.58)."
