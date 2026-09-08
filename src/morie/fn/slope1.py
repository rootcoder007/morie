# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Slope One and Weighted Slope One collaborative-filtering predictors."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["slope_one"]


def slope_one(R, u, i):
    r"""Predict user *u*'s rating of item *i* by the Slope One schemes.

    For two items *j* and *i*, let :math:`S_{j,i}(\chi)` be the set of users
    who rated both.  The average deviation of item *i* with respect to item
    *j* is

    .. math::

        \mathrm{dev}_{j,i}
        = \sum_{u \in S_{j,i}(\chi)}
          \frac{u_j - u_i}{\mathrm{card}(S_{j,i}(\chi))} .

    Since :math:`\mathrm{dev}_{j,i} + u_i` is itself a prediction of
    :math:`u_j`, the SLOPE ONE prediction averages them,

    .. math::

        P(u)_j = \frac{1}{\mathrm{card}(R_j)}
                 \sum_{i \in R_j} (\mathrm{dev}_{j,i} + u_i),
        \qquad
        R_j = \{ i \in S(u) : i \ne j,\ \mathrm{card}(S_{j,i}(\chi)) > 0 \},

    and the WEIGHTED SLOPE ONE prediction weights each term by how many
    users support it,

    .. math::

        P^{wS1}(u)_j
        = \frac{\sum_{i \in S(u)\setminus\{j\}}
                 (\mathrm{dev}_{j,i} + u_i)\, c_{j,i}}
               {\sum_{i \in S(u)\setminus\{j\}} c_{j,i}},
        \qquad c_{j,i} = \mathrm{card}(S_{j,i}(\chi)).

    Both are returned; ``estimate`` is the weighted form, which is the
    scheme Lemire & Maclachlan recommend.  When no co-rated pair supports
    the target item the prediction falls back to the user's own mean, and
    ``fallback`` is set to 1.

    Parameters
    ----------
    R : array-like
        Users-by-items rating matrix.  Missing ratings are NaN.
    u : int
        Zero-based row index of the user whose rating is predicted.
    i : int
        Zero-based column index of the target item.

    Returns
    -------
    RichResult
        ``estimate`` is the Weighted Slope One prediction.

    References
    ----------
    Lemire, D. & Maclachlan, A. (2005). Slope one predictors for online
    rating-based collaborative filtering. Proceedings of the 2005 SIAM
    International Conference on Data Mining, 471-475.
    doi:10.1137/1.9781611972757.43
    """
    M = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(R, dtype=float)).tolist()]
    nu = len(M)
    if nu == 0:
        raise ValueError("slope_one: R is empty")
    ni = len(M[0])
    if any(len(row) != ni for row in M):
        raise ValueError("slope_one: R must be rectangular")
    u = int(u)
    i = int(i)
    if u < 0 or u >= nu:
        raise ValueError("slope_one: u is out of range")
    if i < 0 or i >= ni:
        raise ValueError("slope_one: i is out of range")

    def rated(a, b):
        v = M[a][b]
        return not (v != v)  # NaN check without importing math.isnan on lists

    rated_items = [b for b in range(ni) if b != i and rated(u, b)]
    own = [M[u][b] for b in range(ni) if rated(u, b)]
    user_mean = sum(own) / len(own) if own else float("nan")

    num_s1 = 0.0
    cnt_s1 = 0
    num_w = 0.0
    den_w = 0.0
    support = 0
    for b in rated_items:
        c = 0
        s = 0.0
        for a in range(nu):
            if rated(a, i) and rated(a, b):
                s += M[a][i] - M[a][b]
                c += 1
        if c == 0:
            continue
        dev = s / c
        pred = dev + M[u][b]
        num_s1 += pred
        cnt_s1 += 1
        num_w += pred * c
        den_w += c
        support += c

    fallback = 0.0
    if cnt_s1 == 0 or den_w == 0.0:
        p_s1 = user_mean
        p_w = user_mean
        fallback = 1.0
    else:
        p_s1 = num_s1 / cnt_s1
        p_w = num_w / den_w

    observed = M[u][i] if rated(u, i) else float("nan")
    err = (p_w - observed) if observed == observed else float("nan")

    return RichResult(
        payload={
            "estimate": p_w,
            "weighted_slope_one": p_w,
            "slope_one": p_s1,
            "user_mean": user_mean,
            "n_pairs": float(cnt_s1),
            "support": float(support),
            "fallback": fallback,
            "observed": observed,
            "error": err,
            "n_users": float(nu),
            "n_items": float(ni),
            "user": float(u),
            "item": float(i),
            "method": "Weighted Slope One predictor (Lemire & Maclachlan 2005)",
        }
    )


def cheatsheet():
    return "slope1: Slope One / Weighted Slope One rating predictor"


# compact alias per ledger/NAMING.md
slopeone = slope_one
