# morie.fn -- function file (rootcoder007/morie)
"""Sample mean of an observed signal (Rangayyan eq. 3.7)."""


from math import fsum

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["smean", "rangayyan_ch3_sample_mean"]


def smean(x):
    """Sample mean of N observed values.

    Rangayyan (2024) eq. (3.7):  mu = (1/N) sum_{n=0}^{N-1} eta(n).

    The book calls this the DC component of the signal.  Summed with
    math.fsum so the result does not depend on the order of the samples.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    return RichResult(payload={"mean": fsum(xs) / n, "n": n,
                               "method": "Rangayyan (2024) eq. (3.7)"})


rangayyan_ch3_sample_mean = smean  # pre-policy spelling


def cheatsheet():
    return "rng007: sample mean, Rangayyan eq. (3.7)"
