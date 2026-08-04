# morie.fn -- function file (rootcoder007/morie)
"""Sample RMS, MS, and SD of an observed signal (Rangayyan eqs. 3.8-3.10)."""


from math import fsum, sqrt

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["srms", "rangayyan_ch3_sample_rms"]


def srms(x):
    """Sample RMS value, with the MS and SD it is bracketed by.

    Rangayyan (2024) eqs. (3.8)-(3.10):
        MS  = (1/N) sum x(n)^2
        RMS = sqrt(MS)
        SD  = sqrt( (1/N) sum [x(n) - mu]^2 )

    Note the divisor is N in all three -- eq. (3.10) is the population
    form, not the N-1 unbiased one; a caller wanting the unbiased
    variance should rescale by N/(N-1).  The book reads MS as average
    power and RMS as average signal level.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    mu = fsum(xs) / n
    ms = fsum(v * v for v in xs) / n
    var = fsum((v - mu) ** 2 for v in xs) / n
    return RichResult(payload={
        "rms": sqrt(ms), "ms": ms, "sd": sqrt(var), "mean": mu, "n": n,
        "ddof": 0, "method": "Rangayyan (2024) eqs. (3.8)-(3.10)"})


rangayyan_ch3_sample_rms = srms  # pre-policy spelling


def cheatsheet():
    return "rng009: sample RMS/MS/SD, Rangayyan eqs. (3.8)-(3.10)"
