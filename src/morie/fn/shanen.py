"""Shannon entropy of a discrete distribution."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["shannon_entropy"]


def shannon_entropy(y, base=2.0):
    """
    Shannon entropy of a discrete distribution.

    Formula: H(X) = -sum_x p(x) log p(x)

    Verified against Shannon (1948) Section 6, eq. (H = -K sum p_i log
    p_i), and Cover & Thomas (2006) eq. (2.1) -- sources consulted.
    Counts are accepted as well as probabilities: the input is closed
    to unit sum first.

    Parameters
    ----------
    y : array-like
        Non-negative probabilities or counts.
    base : float, optional
        Log base; 2 gives bits (Shannon's own unit), ``None`` nats.

    Returns
    -------
    RichResult
        Keys: estimate, hmax, evenness, n, method.

    References
    ----------
    Shannon, C.E. (1948). A Mathematical Theory of Communication.
    Bell System Technical Journal 27:379-423, 623-656. Section 6.
    """
    v = _big2.pnorm(np.atleast_1d(np.asarray(y, dtype=float)))
    n = len(v)
    h = _big2.entropy(v, base)
    hmax = float(_big2.logb(float(n), base)) if n > 1 else 0.0
    return RichResult(
        payload={
            "estimate": h,
            "hmax": hmax,
            "evenness": (h / hmax) if hmax > 0.0 else float("nan"),
            "n": n,
            "method": "Shannon entropy of a discrete distribution -- Shannon (1948) Sec. 6",
        }
    )


def cheatsheet():
    return "shanen: Shannon entropy of a discrete distribution"


# compact alias per ledger/NAMING.md
shannonentropy = shannon_entropy
