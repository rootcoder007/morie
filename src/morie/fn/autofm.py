# morie.fn -- function file (rootcoder007/morie)
"""Series decomposition and autocorrelation."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["serdecomp", "autoformer"]


def serdecomp(x, kernel):
    """Series decomposition and autocorrelation.

    Series decomposition and autocorrelation.

    Wu et al. (2021), Autoformer.  A moving average of odd width
    ``kernel``, replicate-padded at both ends, is the trend; the
    remainder is the seasonal part.  The autocorrelation of the
    seasonal part is what the architecture's auto-correlation block
    scores periods with, so it is returned alongside.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Series decomposition and autocorrelation", payload=_c.serdecomp(x=x, kernel=kernel))


autoformer = serdecomp


def cheatsheet():
    return "autofm: Series decomposition and autocorrelation"
