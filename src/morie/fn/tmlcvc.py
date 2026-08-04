# morie.fn -- function file (rootcoder007/morie)
"""Cross-validated TMLE of the ATE."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["cvtmle", "tmle_cv_targeting"]


def cvtmle(y, a, q0, q1, g, fold, n_newton=50):
    """Cross-validated TMLE of the ATE.

    Cross-validated TMLE of the ATE.

    Within each fold the initial fit is the one trained on the other
    folds, so the targeting step never sees the data it is evaluated
    on: that is what removes the empirical-process condition and lets
    data-adaptive nuisance fits be used honestly.  The fluctuation
    parameter solves the score equation for the clever covariate
    H = A/g - (1-A)/(1-g) by a fixed number of Newton steps.

    ``y`` must lie in [0, 1] (bounded outcomes, or already rescaled).

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Cross-validated TMLE of the ATE", payload=_c.cvtmle(y=y, a=a, q0=q0, q1=q1, g=g, fold=fold, n_newton=n_newton))


tmle_cv_targeting = cvtmle


def cheatsheet():
    return "tmlcvc: Cross-validated TMLE of the ATE"
