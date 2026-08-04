# morie.fn -- slice s03 (rootcoder007/morie)
"""Item-based collaborative filtering.

Source consulted: Sarwar, B., Karypis, G., Konstan, J. and Riedl, J.
(2001).  Item-based collaborative filtering recommendation algorithms.
*WWW* 10, 285-295.  Their weighted-sum prediction, equation in section
3.2.1, is

    P(u, i) = sum_(j in N) ( s(i, j) * R(u, j) ) / sum_(j in N) |s(i, j)|

over the k items most similar to i that user u has rated, with the
similarity s the *adjusted cosine* of their section 3.1.3,

    s(i, j) = sum_u (R_ui - Rbar_u)(R_uj - Rbar_u)
              / sqrt( sum_u (R_ui - Rbar_u)^2 ) sqrt( sum_u (R_uj - Rbar_u)^2 )

which subtracts each *user's* mean rather than each item's, because that
is what removes the rating-scale differences between users.  The 2001
WWW proceedings were not retrievable here; both expressions are quoted
in their standard published form.

Unrated entries must be marked NaN, not zero -- zero is a rating.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["item_cf"]


def item_cf(R, u=0, i=0, k_nn=2, similarity="adjusted"):
    """Predict user u's rating of item i by item-based CF.

    Parameters
    ----------
    R : 2-D array-like
        Ratings, users in rows, items in columns; NaN where unrated.
    u, i : int
        The user and item to predict.
    k_nn : int
        Neighbourhood size.
    similarity : {"adjusted", "cosine"}
        Adjusted cosine (user-mean centred) or plain cosine.

    Returns
    -------
    RichResult with payload:
        estimate  : the predicted rating
        neighbours: the items used, most similar first
        sims      : their similarities
    """
    M = k.mat(R)
    nu = len(M)
    ni = len(M[0]) if nu else 0
    umean = []
    for a in range(nu):
        vals = [M[a][b] for b in range(ni) if M[a][b] == M[a][b]]
        umean.append(k.mean(vals) if vals else 0.0)

    def sim(p, q):
        num = 0.0
        d1 = 0.0
        d2 = 0.0
        for a in range(nu):
            if M[a][p] != M[a][p] or M[a][q] != M[a][q]:
                continue
            xp = M[a][p] - (umean[a] if similarity == "adjusted" else 0.0)
            xq = M[a][q] - (umean[a] if similarity == "adjusted" else 0.0)
            num += xp * xq
            d1 += xp * xp
            d2 += xq * xq
        d = math.sqrt(d1 * d2)
        return num / d if d > 0.0 else 0.0

    ii = int(i)
    uu = int(u)
    cand = [j for j in range(ni) if j != ii and M[uu][j] == M[uu][j]]
    sims = [sim(ii, j) for j in cand]
    order = sorted(range(len(cand)), key=lambda t: (-abs(sims[t]), cand[t]))
    take = order[:int(k_nn)]
    num = 0.0
    den = 0.0
    for t in take:
        num += sims[t] * M[uu][cand[t]]
        den += abs(sims[t])
    pred = num / den if den > 0.0 else float("nan")
    return RichResult(
        title="Item-based collaborative filtering",
        summary_lines=[("prediction", pred)],
        payload={
            "estimate": pred,
            "prediction": pred,
            "neighbours": [cand[t] for t in take],
            "sims": [sims[t] for t in take],
            "method": "Item-based CF with an adjusted-cosine weighted sum (Sarwar et al. 2001)",
        },
    )


def cheatsheet():
    return "icfR: Item-based CF"


itemcf = item_cf
