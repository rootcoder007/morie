# morie.fn -- function file (rootcoder007/morie)
"""Hill tail-index estimator -- alias entry point."""

from ._richresult import RichResult

__all__ = ["hill_estimator"]


def hill_estimator(x, k=None):
    """Hill (1975) tail-index estimator. This module and
    ``morie.fn.evhill`` are ONE estimator with two catalogue entries;
    the computation lives in :func:`morie.fn.evhill.ev_hill`, so the
    two cannot drift apart. See that module for the xi > 0
    restriction, the bias-variance role of k, and the Hill plot.

    References
    ----------
    Hill, B. M. (1975), *Annals of Statistics* 3:1163-1174.
    """
    from .evhill import ev_hill

    out = ev_hill(x, k=k)
    payload = dict(out)
    payload["alias_of"] = "morie.fn.evhill.ev_hill"
    return RichResult(payload=payload)


def cheatsheet():
    return "hillEst: same estimator as evhill, one implementation"
