# morie.fn -- function file (rootcoder007/morie)
"""X-learner heterogeneous treatment effect."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["xlearn", "xlearner", "x_learner"]


def xlearn(tau1, tau0, g):
    """X-learner heterogeneous treatment effect.

    X-learner combination: tau = g tau0 + (1 - g) tau1.

    The X-learner imputes each unit's missing arm from the other arm's
    fitted response, then blends the two resulting effect estimates by
    the propensity.  The weighting is the point: when one arm is much
    smaller, its own imputed effect is noisy, and weighting by g pushes
    the estimate toward the estimate built from the larger arm.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="X-learner heterogeneous treatment effect", payload=_c.xlearn(tau1=tau1, tau0=tau0, g=g))


x_learner = xlearn


def cheatsheet():
    return "xlrnir: X-learner heterogeneous treatment effect"


# compact alias per ledger/NAMING.md (pre-existing spelling, kept working)
xlearner = xlearn
