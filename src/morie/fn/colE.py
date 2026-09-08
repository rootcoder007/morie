# morie.fn -- function file (rootcoder007/morie)
"""Cold-start recommendation fallbacks."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["cold_start_user"]


def cold_start_user(user, mode="popular", R=None, item_features=None,
                    user_features=None, min_ratings=3, topn=3):
    """
    Cold-start user fallback

    Formula: fallback to popular / content / metadata

    A user with fewer than min_ratings observed interactions cannot be
    served by collaborative filtering at all, because their row of the
    rating matrix carries no signal.  The three fallbacks are: rank by
    global popularity, rank by similarity between the item features and
    whatever the user has rated, and rank by similarity to users with
    matching metadata.  The cold/warm decision is reported so it can be
    audited separately from the ranking.

    Parameters
    ----------
    user : int
        Index of the target user.
    mode : str
        One of popular, content, metadata.
    R : array-like
        n_users x n_items rating matrix; 0 means unobserved.
    item_features : array-like or None
        n_items x f matrix, needed by the content fallback.
    user_features : array-like or None
        n_users x g matrix, needed by the metadata fallback.
    min_ratings : int
        Threshold below which the user counts as cold.
    topn : int
        Length of the returned recommendation list.

    Returns
    -------
    result : dict
        Keys: estimate (score of the top item), is_cold, n_rated,
        scores, recommended, mode, n_users, n_items.

    References
    ----------
    Schein, Popescul, Ungar & Pennock (2002), Methods and Metrics for
    Cold-Start Recommendations, SIGIR 2002:253-260.
    """
    if R is None:
        raise ValueError("R is required")
    Rm = core.mat(R)
    nu = len(Rm)
    if nu == 0:
        raise ValueError("empty input: R has no rows")
    ni = len(Rm[0])
    u = int(user)
    if u < 0 or u >= nu:
        raise ValueError("user index out of range")
    mode = str(mode).lower()
    if mode not in ("popular", "content", "metadata"):
        raise ValueError("mode must be popular, content or metadata")
    rated = [j for j in range(ni) if Rm[u][j] != 0.0]
    is_cold = 1 if len(rated) < int(min_ratings) else 0
    if mode == "popular":
        scores = [sum(1.0 for i in range(nu) if Rm[i][j] != 0.0)
                  for j in range(ni)]
    elif mode == "content":
        if item_features is None:
            raise ValueError("content mode needs item_features")
        F = core.mat(item_features)
        if len(F) != ni:
            raise ValueError("item_features must have one row per item")
        f = len(F[0])
        if rated:
            prof = [sum(Rm[u][j] * F[j][t] for j in rated) /
                    sum(Rm[u][j] for j in rated) for t in range(f)]
        else:
            prof = [sum(F[j][t] for j in range(ni)) / ni for t in range(f)]
        pn = math.sqrt(sum(v * v for v in prof))
        scores = []
        for j in range(ni):
            fn = math.sqrt(sum(v * v for v in F[j]))
            scores.append(sum(prof[t] * F[j][t] for t in range(f)) / (pn * fn)
                          if pn > 0.0 and fn > 0.0 else 0.0)
    else:
        if user_features is None:
            raise ValueError("metadata mode needs user_features")
        U = core.mat(user_features)
        if len(U) != nu:
            raise ValueError("user_features must have one row per user")
        g = len(U[0])
        un = math.sqrt(sum(v * v for v in U[u]))
        sim = []
        for i in range(nu):
            vn = math.sqrt(sum(v * v for v in U[i]))
            sim.append(sum(U[u][t] * U[i][t] for t in range(g)) / (un * vn)
                       if un > 0.0 and vn > 0.0 else 0.0)
        tot = sum(sim[i] for i in range(nu) if i != u)
        scores = [sum(sim[i] * Rm[i][j] for i in range(nu) if i != u) /
                  (tot if tot != 0.0 else 1.0) for j in range(ni)]
    order = sorted(range(ni), key=lambda j: (-scores[j], j))
    rec = [j for j in order if j not in rated][:int(topn)]
    return RichResult(payload={
        "estimate": scores[rec[0]] if rec else float("nan"),
        "is_cold": is_cold,
        "n_rated": len(rated),
        "scores": scores,
        "recommended": rec,
        "mode": mode,
        "n_users": nu,
        "n_items": ni,
        "method": "cold-start recommendation fallback",
    })


def cheatsheet():
    return "colE: cold-start recommendation fallback"

# public names resolved by fn/_lazy_map.json
coldstartuser = cold_start_user
