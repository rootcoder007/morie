# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hellinger distance between two discrete distributions.

Source consulted: Hellinger, E. (1909). Neue Begruendung der Theorie
quadratischer Formen von unendlichvielen Veraenderlichen.  *Journal fuer die
reine und angewandte Mathematik* 136, 210-271, where the integral that now
carries his name is introduced.  In the discrete case the distance is

    H(P, Q) = (1/sqrt 2) sqrt( sum_x ( sqrt(p(x)) - sqrt(q(x)) )^2 )

which lies in [0, 1], equals 0 exactly when P = Q and 1 when the supports are
disjoint.  The Bhattacharyya coefficient BC = sum_x sqrt(p(x) q(x)) satisfies
H^2 = 1 - BC and is reported alongside.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hellinger_distance"]


def hellinger_distance(p, q, normalise=True):
    """Hellinger distance between two discrete distributions.

    Parameters
    ----------
    p, q : array-like
        Non-negative masses over a common support.
    normalise : bool
        Rescale each argument to sum to one first.

    Returns
    -------
    RichResult
        estimate (H), h2, bc (Bhattacharyya coefficient), affinity, n, method.

    References
    ----------
    Hellinger (1909), J. reine angew. Math. 136, 210-271.
    """
    pp = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
    qq = np.atleast_1d(np.asarray(q, dtype=float)).ravel()
    n = int(min(pp.size, qq.size))
    pv = [float(pp[i]) for i in range(n)]
    qv = [float(qq[i]) for i in range(n)]
    if normalise:
        sp = sum(pv)
        sq = sum(qv)
        if sp > 0.0:
            pv = [v / sp for v in pv]
        if sq > 0.0:
            qv = [v / sq for v in qv]
    ss = sum((float(np.sqrt(pv[i])) - float(np.sqrt(qv[i]))) ** 2 for i in range(n))
    h = float(np.sqrt(ss / 2.0))
    bc = sum(float(np.sqrt(pv[i] * qv[i])) for i in range(n))
    return RichResult(
        payload={
            "estimate": h,
            "h2": float(h * h),
            "bc": float(bc),
            "affinity": float(bc),
            "n": n,
            "method": "Hellinger distance (Hellinger 1909)",
        }
    )


# CANONICAL TEST
# >>> # identical distributions are at distance zero, BC = 1
# >>> r = hellinger_distance([0.25, 0.75], [0.25, 0.75])
# >>> assert abs(r["estimate"]) < 1e-12 and abs(r["bc"] - 1.0) < 1e-12
# >>> # disjoint support is at the maximum distance 1
# >>> r2 = hellinger_distance([1.0, 0.0], [0.0, 1.0])
# >>> assert abs(r2["estimate"] - 1.0) < 1e-12


def cheatsheet():
    return "hellie(p, q): Hellinger distance + Bhattacharyya coefficient."
