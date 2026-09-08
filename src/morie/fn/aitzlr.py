# morie.fn -- function file (rootcoder007/morie)
"""Log-ratio EM for values below a detection limit."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["lrem", "compositional_zero_lrem"]


def lrem(X, dl, n_iter=20):
    """Log-ratio EM for values below a detection limit.

    Log-ratio EM imputation of values below a detection limit.

    Palarea-Albaladejo & Martin-Fernandez (2008).  Replacing rounded
    zeros by a fraction of the detection limit distorts the covariance
    structure; lrEM instead imputes the conditional expectation under a
    normal model in alr coordinates, which preserves the ratios that
    compositional analysis actually uses.  ``n_iter`` is fixed.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Log-ratio EM for values below a detection limit", payload=_c.lrem(X=X, dl=dl, n_iter=n_iter))


compositional_zero_lrem = lrem


def cheatsheet():
    return "aitzlr: Log-ratio EM for values below a detection limit"
