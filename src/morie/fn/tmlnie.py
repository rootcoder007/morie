# morie.fn -- function file (rootcoder007/morie)
"""Natural indirect effect."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["nieff", "tmle_natural_indirect"]


def nieff(y11, y10):
    """Natural indirect effect.

    Natural indirect effect: E[Y(1, M(1))] - E[Y(1, M(0))].

    The effect that runs through the mediator, holding treatment fixed
    at 1 and moving only the mediator's distribution.  Adding this to
    the natural direct effect recovers the total effect exactly, which
    is the decomposition the pair exists for.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Natural indirect effect", payload=_c.nieff(y11=y11, y10=y10))


tmle_natural_indirect = nieff


def cheatsheet():
    return "tmlnie: Natural indirect effect"
