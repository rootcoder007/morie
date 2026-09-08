# morie.fn -- function file (rootcoder007/morie)
"""Directly age-standardised rate."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['agestd', 'age_standardize']


def agestd(rates, standard_pop, person_time=None):
    """Directly age-standardised rate.

    Direct standardisation applies the study population's age-specific rates to a standard population's age structure, so two populations become comparable without their age structures doing the comparing. The variance treats each stratum's numerator as Poisson, which needs the person-time denominators; without them a rate is returned but no interval, rather than an interval built on an assumption the caller never made.


    Formula: ASR = sum_i w_i r_i / sum_i w_i; var(ASR) = sum_i w_i^2 r_i / n_i / (sum_i w_i)^2

    Parameters
    ----------
    rates : array-like
        Age-specific rates.
    standard_pop : array-like
        Standard population weights by age band.
    person_time : array-like, optional
        Person-time in each band of the study population.

    Returns
    -------
    RichResult
        ``asr``, ``variance``, ``se``, ``ci_lower``, ``ci_upper``, ``weights``, ``k``.

    References
    ----------
    Boyle and Parkin (1991), Statistical methods for registries, in
    Jensen et al (eds), Cancer Registration: Principles and Methods,
    IARC Scientific Publications 95.  Not held locally; the direct
    standardisation estimator and its Poisson variance are the standard
    published forms.
    """
    r = C.vec(rates); w = C.vec(standard_pop)
    k = len(r)
    if k != len(w):
        raise ValueError("rates and standard_pop must be the same length")
    sw = sum(w)
    if sw <= 0:
        raise ValueError("standard population must have positive total")
    asr = sum(w[i] * r[i] for i in range(k)) / sw
    var = se = lo = hi = float("nan")
    if person_time is not None:
        n = C.vec(person_time)
        if any(t <= 0 for t in n):
            raise ValueError("person-time must be positive")
        var = sum(w[i] * w[i] * r[i] / n[i] for i in range(k)) / (sw * sw)
        se = math.sqrt(var)
        z = C.qnorm(0.975)
        lo, hi = asr - z * se, asr + z * se
    return RichResult(payload={
        "asr": asr, "variance": var, "se": se, "ci_lower": lo,
        "ci_upper": hi, "weights": [v / sw for v in w], "k": k,
        "method": "Directly age-standardised rate"})


age_standardize = agestd


def cheatsheet():
    return "agstds: Directly age-standardised rate."
