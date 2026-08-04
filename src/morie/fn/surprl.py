"""Surprisal -log p(x)."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["surprisal"]


def surprisal(p, x, base=2.0):
    """
    Surprisal (self-information) of one or more outcomes.

    Formula: I(x) = -log p(x)

    Verified against Shannon (1948) Section 6 and Cover & Thomas (2006)
    p. 14, where entropy is defined as the expectation of -log p(X) --
    sources consulted. The mean surprisal returned here is exactly that
    entropy when the outcomes are drawn from p.

    Parameters
    ----------
    p : array-like
        Non-negative pmf; closed to unit sum internally.
    x : array-like or int
        Outcome index or indices into that pmf.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate (mean surprisal), values (per outcome), entropy,
        n, method.

    References
    ----------
    Shannon, C.E. (1948). A Mathematical Theory of Communication.
    Bell System Technical Journal 27:379-423, 623-656. Section 6.
    """
    v = _big2.pnorm(np.atleast_1d(np.asarray(p, dtype=float)))
    k = len(v)
    idx = [int(t) for t in np.atleast_1d(np.asarray(x, dtype=float))]
    if not idx:
        raise ValueError("x must be non-empty")
    inf = float("inf")
    vals = []
    for i in idx:
        if i < 0 or i >= k:
            raise ValueError("outcome index outside the alphabet")
        pi = float(v[i])
        vals.append(inf if pi <= 0.0 else -float(_big2.logb(pi, base)))
    mean = inf if inf in vals else float(sum(vals) / len(vals))
    return RichResult(
        payload={
            "estimate": mean,
            "values": vals,
            "entropy": _big2.entropy(v, base),
            "n": len(idx),
            "method": "Surprisal -log p(x) -- Shannon (1948) Sec. 6",
        }
    )


def cheatsheet():
    return "surprl: Surprisal -log p(x)"
