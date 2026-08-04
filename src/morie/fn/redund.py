"""Redundancy 1 - H(X)/H_max."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["redundancy"]


def redundancy(p, base=2.0):
    """
    Redundancy of a source.

    Formula: R = 1 - H(X)/log|alphabet|

    Verified against Shannon (1948) Section 7 -- source consulted.
    Shannon writes: "The ratio of the entropy of a source to the
    maximum value it could have while still restricted to the same
    symbols will be called its relative entropy ... One minus the
    relative entropy is the redundancy."

    Parameters
    ----------
    p : array-like
        Non-negative source pmf; closed to unit sum internally.
    base : float, optional
        Log base for the reported entropies; the redundancy itself is a
        ratio and does not depend on it.

    Returns
    -------
    RichResult
        Keys: estimate, entropy, hmax, relative, n, method.

    References
    ----------
    Shannon, C.E. (1948). A Mathematical Theory of Communication.
    Bell System Technical Journal 27:379-423, 623-656. Section 7.
    """
    v = _big2.pnorm(np.atleast_1d(np.asarray(p, dtype=float)))
    n = len(v)
    if n < 2:
        raise ValueError("redundancy needs an alphabet of at least two symbols")
    h = _big2.entropy(v, base)
    hmax = float(_big2.logb(float(n), base))
    rel = h / hmax
    return RichResult(
        payload={
            "estimate": 1.0 - rel,
            "entropy": h,
            "hmax": hmax,
            "relative": rel,
            "n": n,
            "method": "Redundancy 1 - H/Hmax -- Shannon (1948) Sec. 7",
        }
    )


def cheatsheet():
    return "redund: Redundancy 1 - H(X)/H_max"
