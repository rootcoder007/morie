"""Prediction-compression duality."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["prediction_compression"]


def prediction_compression(model, data, base=2.0):
    """
    Expected code length of a predictive model -- the wrong-code bound.

    Formula: compression rate ~ -E[log p_model(x)] = H(p) + D(p||q)

    Verified against Cover & Thomas (2006) Theorem 5.4.3 ("Wrong code"),
    eq. (5.42) p. 115 -- source consulted. Coding a source p with the
    code designed for q costs between H(p) + D(p||q) and
    H(p) + D(p||q) + 1 bits per symbol.

    Parameters
    ----------
    model : array-like
        Predictive pmf q over an alphabet of size K; normalised
        internally.
    data : array-like
        Observed symbols, as integer indices into that alphabet.
    base : float, optional
        Log base; 2 gives bits per symbol.

    Returns
    -------
    RichResult
        Keys: estimate, entropy, kl, upper, n, method. ``estimate`` is
        the empirical rate ``-mean log q(x_i)``; ``upper`` is the
        H + D + 1 bound of eq. (5.42).

    References
    ----------
    Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory,
    2nd ed. Wiley. Theorem 5.4.3, eq. (5.42).
    """
    q = _big2.pnorm(np.atleast_1d(np.asarray(model, dtype=float)))
    k = len(q)
    idx = [int(v) for v in np.atleast_1d(np.asarray(data, dtype=float))]
    n = len(idx)
    if n == 0:
        raise ValueError("data must be non-empty")
    for i in idx:
        if i < 0 or i >= k:
            raise ValueError("data index outside the model alphabet")
    counts = [0.0] * k
    for i in idx:
        counts[i] += 1.0
    phat = [c / n for c in counts]
    inf = float("inf")
    rate = 0.0
    for i in idx:
        qi = float(q[i])
        if qi <= 0.0:
            rate = inf
            break
        rate -= float(_big2.logb(qi, base))
    rate = inf if rate == inf else rate / n
    hp = _big2.entropy(phat, base)
    d = _big2.kldiv(phat, [float(v) for v in q], base)
    return RichResult(
        payload={
            "estimate": rate,
            "entropy": hp,
            "kl": d,
            "upper": inf if d == inf else hp + d + 1.0,
            "n": n,
            "method": "Wrong-code expected length H(p)+D(p||q) -- Cover & Thomas (2006) Thm 5.4.3",
        }
    )


def cheatsheet():
    return "pcmpr1: Prediction-compression duality"
