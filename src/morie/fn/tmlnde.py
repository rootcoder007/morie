# morie.fn -- function file (rootcoder007/morie)
"""Natural direct effect."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["ndeff", "tmle_natural_direct"]


def ndeff(y10, y00):
    """Natural direct effect.

    Natural direct effect: E[Y(1, M(0))] - E[Y(0, M(0))].

    The effect of treatment holding the mediator at the distribution it
    would have taken under control -- the path that does not run through
    the mediator.  Both cross-world quantities are supplied by the
    caller because neither is identified from data without an
    assumption; this routine contrasts them and does not assert one.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Natural direct effect", payload=_c.ndeff(y10=y10, y00=y00))


tmle_natural_direct = ndeff


def cheatsheet():
    return "tmlnde: Natural direct effect"
