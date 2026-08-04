# morie.fn -- function file (rootcoder007/morie)
"""Dominant periods from the amplitude spectrum."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["fftperiod", "timesnet"]


def fftperiod(x, k=1):
    """Dominant periods from the amplitude spectrum.

    Dominant periods from the amplitude spectrum.

    Wu et al. (2023), TimesNet.  The periods that carry the most
    amplitude are read off the discrete Fourier transform and used to
    fold the 1-D series into a 2-D tensor, which is how intraperiod and
    interperiod variation get separated.  Frequency 0 is excluded and
    only the first half of the spectrum is used, since the rest is its
    mirror image.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Dominant periods from the amplitude spectrum", payload=_c.fftperiod(x=x, k=k))


timesnet = fftperiod


def cheatsheet():
    return "timesnet: Dominant periods from the amplitude spectrum"
