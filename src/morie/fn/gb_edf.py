# morie.fn -- function file (rootcoder007/morie)
"""Empirical distribution function S_n(x) -- Gibbons eq. (2.3.1)."""

import math

from ._richresult import RichResult

__all__ = ['edfstep', 'gibbons_edf_def']


def edfstep(x, t):
    """Empirical cdf of a sample evaluated at one or more points.

    Equation (2.3.1) (book p. 32): with X_(1) <= ... <= X_(n),

    .. math::
        S_n(x) = 0,\\; x < X_{(1)};\\quad
        S_n(x) = k/n,\\; X_{(k)} \\le x < X_{(k+1)};\\quad
        S_n(x) = 1,\\; x \\ge X_{(n)}.

    Parameters
    ----------
    x : sequence of float
        The sample, n >= 1.
    t : float or sequence of float
        Evaluation point(s).

    Returns
    -------
    RichResult
        keys ``edf`` (list, one per t), ``count`` (list of n*S_n),
        ``n``, ``sorted`` (the order statistics), ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (2.3.1), p. 32.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 1:
        raise ValueError("x must be non-empty.")
    ts = [float(t)] if not hasattr(t, "__iter__") else [float(v) for v in t]
    counts = []
    for v in ts:
        c = 0
        for xi in xs:
            if xi <= v:
                c += 1
            else:
                break
        counts.append(c)
    return RichResult(
        payload={
            "edf": [c / n for c in counts],
            "count": counts,
            "n": n,
            "sorted": xs,
            "method": "S_n(x) = (# X_i <= x)/n (Gibbons eq. 2.3.1)",
        }
    )


gibbons_edf_def = edfstep
