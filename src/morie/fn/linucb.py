# morie.fn -- function file (rootcoder007/morie)
"""LinUCB arm scores."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["linucb", "linucb"]


def linucb(x, theta, Ainv, alpha=1.0):
    """LinUCB arm scores.

    LinUCB arm scores: p_a = theta_a' x + alpha sqrt(x' A_a^-1 x).

    Li et al. (2010).  The bonus term is a confidence radius, not noise:
    it is large exactly for arms whose design matrix has seen little
    variation along x, so exploration is directed rather than random.
    ``theta[a]`` and ``Ainv[a]`` are the per-arm parameter and inverse
    design matrix.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="LinUCB arm scores", payload=_c.linucb(x=x, theta=theta, Ainv=Ainv, alpha=alpha))


linucb = linucb


def cheatsheet():
    return "linucb: LinUCB arm scores"
