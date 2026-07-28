# morie.fn -- function file (rootcoder007/morie)
"""MM-estimator regression -- alias entry point."""

from ._richresult import RichResult

__all__ = ["mm_estimator"]


def mm_estimator(X, y, n_subsets=200, seed=0):
    """Yohai's (1987) MM-estimator. This module and
    ``morie.fn.mmreg`` are ONE estimator with two catalogue entries;
    the computation lives in ``morie.fn._robust.mm_regression`` and
    is invoked through :func:`morie.fn.mmreg.mm_regression_estimator`
    so the two cannot drift apart. See that module for the
    construction and the reason the scale is held fixed through the
    efficiency stage.

    References
    ----------
    Yohai, V. J. (1987), *Annals of Statistics* 15:642-656.
    """
    from .mmreg import mm_regression_estimator

    out = mm_regression_estimator(X, y, n_subsets=n_subsets, seed=seed)
    payload = dict(out)
    payload["alias_of"] = "morie.fn.mmreg.mm_regression_estimator"
    return RichResult(payload=payload)


def cheatsheet():
    return "mmestr: same estimator as mmreg, one implementation -- see mmreg"
