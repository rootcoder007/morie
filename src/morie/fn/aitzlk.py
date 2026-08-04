# morie.fn -- function file (rootcoder007/morie)
"""Log-ratio data augmentation below a detection limit."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["lrda", "compositional_zero_lrda"]


def lrda(X, dl, draw, n_iter=20):
    """Log-ratio data augmentation below a detection limit.

    Log-ratio data augmentation for values below a detection limit.

    Palarea-Albaladejo, Martin-Fernandez & Olea (2013).  The Bayesian
    counterpart of :func:`lrem`: each step draws the censored part from
    its truncated conditional rather than taking the mean, so the
    imputation carries the right uncertainty.  Standard normal variates
    are supplied by the caller, so a run is reproducible.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Log-ratio data augmentation below a detection limit", payload=_c.lrda(X=X, dl=dl, draw=draw, n_iter=n_iter))


compositional_zero_lrda = lrda


def cheatsheet():
    return "aitzlk: Log-ratio data augmentation below a detection limit"
