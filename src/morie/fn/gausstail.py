"""Gaussian ordinate at n sigma, as a fraction of the peak area scale.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (5.25).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["gausstail"]


def gausstail(n_sigmas=20.0, sigma=1.0):
    """Gaussian ordinate at n sigma, as a fraction of the peak area scale.

    sigma f(n sigma) = e^(-n^2/2)/sqrt(2 pi), independent of sigma.
    The default n = 20 is the book's 1e-87 illustration.

    Parameters
    ----------
    n_sigmas : float
        How many standard deviations out to evaluate.
    sigma : float
        Standard deviation, > 0.

    Returns
    -------
    RichResult
        Keys: n_sigmas, area_fraction.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (5.25).
    """
    x = float(n_sigmas) * float(sigma)
    value = _morin.normal_pdf(x, 0.0, sigma) * float(sigma)
    payload = {"n_sigmas": float(n_sigmas), "area_fraction": value}
    lines = [("sigma * f(x)", value)]
    return RichResult(
        title="Gaussian ordinate at n sigma, as a fraction of the peak area scale.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "gausstail: sigma f(n sigma) = e^(-n^2/2)/sqrt(2 pi). Morin (2016) eq (5.25)."
