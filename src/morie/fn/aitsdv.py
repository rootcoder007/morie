"""Shannon diversity of a closed composition."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["compositional_shannon"]


def compositional_shannon(x, base=2.0):
    """
    Shannon entropy of a composition (a vector closed to unit sum).

    Formula: H(x) = -sum x_i log x_i

    Verified against Shannon (1948), *A Mathematical Theory of
    Communication*, Section 6, and Cover & Thomas (2006) eq. (2.1) --
    sources consulted. The only difference from ``shannon_entropy`` is
    that the input is explicitly treated as a composition: it is closed
    to unit total before the entropy is taken, and the closure constant
    is reported.

    Parameters
    ----------
    x : array-like
        Non-negative parts of a composition.
    base : float, optional
        Log base; 2 gives bits.

    Returns
    -------
    RichResult
        Keys: estimate, closure, evenness, n, method. ``evenness`` is
        H / log(K), the relative entropy of Shannon's Section 7.

    References
    ----------
    Shannon, C.E. (1948). A Mathematical Theory of Communication.
    Bell System Technical Journal 27:379-423, 623-656. Sections 6-7.
    """
    v = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(v)
    if n < 1:
        raise ValueError("x must be non-empty")
    closure = float(np.sum(v))
    p = _big2.pnorm(v)
    h = _big2.entropy(p, base)
    hmax = float(_big2.logb(float(n), base))
    return RichResult(
        payload={
            "estimate": h,
            "closure": closure,
            "evenness": (h / hmax) if hmax > 0.0 else float("nan"),
            "n": n,
            "method": "Shannon entropy of a composition -- Shannon (1948) Sec. 6",
        }
    )


def cheatsheet():
    return "aitsdv: Shannon diversity of a closed composition"
