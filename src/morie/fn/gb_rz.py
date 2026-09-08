# morie.fn -- function file (rootcoder007/morie)
"""Randomized decision rule attaining an exact significance level."""

import math

from ._richresult import RichResult

__all__ = ['randtest', 'gibbons_rz_test']


def randtest(pmf, alpha=0.05, pmf_alt=None):
    """Randomized test: reject above t2, reject with probability p at t1.

    Section 1.2.9 (book p. 26-27).  The rule rejects always when
    T >= t2, rejects with probability p when t1 <= T < t2, and never
    otherwise, with p chosen so the size is exactly alpha:

    .. math:: p = \\frac{\\alpha - P(T \\ge t_2)}{P(T = t_1)}.

    Its power against an alternative is
    P(T >= t2 | H1) + p P(t1 <= T < t2 | H1), evaluated when
    ``pmf_alt`` is supplied.

    Parameters
    ----------
    pmf : sequence of float
        Null probabilities over the support, increasing in T.
    alpha : float, optional
        Target exact size (default 0.05).
    pmf_alt : sequence of float, optional
        Alternative probabilities on the same support.

    Returns
    -------
    RichResult
        keys ``gamma`` (the randomization probability p), ``t2``,
        ``t1`` (indices into the support), ``size_hard`` (P(T >= t2)),
        ``size``, ``power``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 1.2.9, pp. 26-27.
    """
    p = [float(v) for v in pmf]
    k = len(p)
    if k < 2:
        raise ValueError("pmf needs at least 2 support points.")
    alpha = float(alpha)
    t2 = k
    hard = 0.0
    for i in range(k - 1, -1, -1):
        tail = sum(p[i:])
        if tail <= alpha:
            t2 = i
            hard = tail
        else:
            break
    t1 = t2 - 1
    if t1 < 0:
        gamma = 0.0
        t1 = 0
    else:
        gamma = (alpha - hard) / p[t1] if p[t1] > 0.0 else 0.0
        gamma = min(1.0, max(0.0, gamma))
    power = float("nan")
    if pmf_alt is not None:
        q = [float(v) for v in pmf_alt]
        if len(q) != k:
            raise ValueError("pmf_alt must match pmf in length.")
        power = sum(q[t2:]) + gamma * (q[t1] if t2 > 0 else 0.0)
    return RichResult(
        payload={
            "gamma": float(gamma),
            "t2": int(t2),
            "t1": int(t1),
            "size_hard": float(hard),
            "size": float(hard + gamma * (p[t1] if t2 > 0 else 0.0)),
            "power": float(power),
            "method": "randomized decision rule of exact size (Sec. 1.2.9)",
        }
    )


gibbons_rz_test = randtest
