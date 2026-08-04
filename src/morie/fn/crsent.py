"""Cross entropy H(p, q)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["cross_entropy"]


def cross_entropy(p, q, base=2.0):
    """
    Cross entropy of p relative to q.

    Formula: -sum p log q

    Verified against Cover & Thomas (2006): the cross entropy is
    H(p) + D(p||q) with H the entropy of eq. (2.1) and D the relative
    entropy of eq. (2.26) -- source consulted. Both pieces are returned
    so the decomposition stays visible.

    Parameters
    ----------
    p, q : array-like
        Non-negative vectors of the same length; each closed to unit
        sum internally.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate, entropy, kl, n, method.

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Eq. (2.1), (2.26).
    """
    pv = _big2.pnorm(np.atleast_1d(np.asarray(p, dtype=float)))
    qv = _big2.pnorm(np.atleast_1d(np.asarray(q, dtype=float)))
    n = len(pv)
    if len(qv) != n:
        raise ValueError("p and q must have the same length")
    inf = float("inf")
    ce = 0.0
    for i in range(n):
        pi = float(pv[i])
        if pi <= 0.0:
            continue
        qi = float(qv[i])
        if qi <= 0.0:
            ce = inf
            break
        ce -= pi * float(_big2.logb(qi, base))
    h = _big2.entropy(pv, base)
    return RichResult(
        payload={
            "estimate": ce,
            "entropy": h,
            "kl": inf if ce == inf else ce - h,
            "n": n,
            "method": "Cross entropy H(p) + D(p||q) -- Cover & Thomas (2006) eq. (2.26)",
        }
    )


def cheatsheet():
    return "crsent: Cross entropy H(p, q)"
