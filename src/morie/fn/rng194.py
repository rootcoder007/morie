# morie.fn -- function file (rootcoder007/morie)
"""Heart rate from RR interval."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch4_heart_rate_from_rr"]


def rangayyan_ch4_heart_rate_from_rr(RR_a):
    r"""Instantaneous heart rate from the RR interval (Rangayyan
    Ch. 4):

    .. math:: HR = \frac{60}{RR_a},

    with RR in seconds and HR in beats per minute. Vectorised, so a
    series of RR intervals gives the instantaneous rate at each beat;
    the mean of those is NOT the same as 60 / mean(RR) (Jensen), and
    both are returned.

    Parameters
    ----------
    RR_a : float or array-like
        RR interval(s) in seconds, strictly positive.

    Returns
    -------
    RichResult
        keys: ``heart_rate``, ``mean_instantaneous_hr``,
        ``hr_from_mean_rr``, ``n``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 4.
    """
    rr = np.atleast_1d(np.asarray(RR_a, dtype=float))
    if np.any(rr <= 0):
        raise ValueError("RR intervals must be strictly positive.")
    hr = 60.0 / rr
    scalar = np.ndim(RR_a) == 0
    return RichResult(
        payload={"heart_rate": float(hr[0]) if scalar else hr,
                 "mean_instantaneous_hr": float(np.mean(hr)),
                 "hr_from_mean_rr": float(60.0 / np.mean(rr)), "n": int(rr.size),
                 "method": "HR = 60/RR; mean of rates != rate of mean interval"})


def cheatsheet():
    return "rng194: HR = 60/RR; mean(60/RR) != 60/mean(RR)"
