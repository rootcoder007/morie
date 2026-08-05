# morie.fn -- function file (rootcoder007/morie)
"""Content-based recommendation."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["content_based"]


def content_based(item_feat, user_profile, ratings=None, topn=3):
    """
    Content-based recommendation

    Formula: score = sim(item profile, user profile)

    The user profile is the rating-weighted mean of the feature vectors
    of the items they liked, and every item is then scored by cosine
    similarity to it.  A user who has rated exactly one item therefore
    has that item's feature vector as their profile, so the top
    recommendation is its nearest neighbour in feature space -- the
    degenerate case that pins the weighting.

    Parameters
    ----------
    item_feat : array-like
        n_items x f matrix of item feature vectors.
    user_profile : array-like
        Either a length-f profile vector, or a length-n_items rating
        vector from which the profile is built (0 = unrated).
    ratings : array-like or None
        Explicit rating vector; overrides the second reading above.
    topn : int
        Length of the returned list.

    Returns
    -------
    result : dict
        Keys: estimate (top score), scores, ranking, recommended,
        profile, n_items, f.

    References
    ----------
    Pazzani & Billsus (2007), Content-Based Recommendation Systems, in
    The Adaptive Web, LNCS 4321:325-341.
    """
    F = core.mat(item_feat)
    ni = len(F)
    if ni == 0:
        raise ValueError("empty input: item_feat has no rows")
    f = len(F[0])
    up = core.vec(user_profile if ratings is None else ratings)
    rated = []
    if ratings is not None or len(up) == ni:
        if len(up) != ni:
            raise ValueError("the rating vector must have one entry per item")
        rated = [j for j in range(ni) if up[j] != 0.0]
        if not rated:
            raise ValueError("the user has rated nothing; no profile exists")
        w = sum(up[j] for j in rated)
        prof = [sum(up[j] * F[j][t] for j in rated) / w for t in range(f)]
    else:
        if len(up) != f:
            raise ValueError("user_profile must be a length-f profile or a "
                             "length-n_items rating vector")
        prof = list(up)
    pn = math.sqrt(sum(v * v for v in prof))
    if pn <= 0.0:
        raise ValueError("the user profile has zero norm")
    scores = []
    for j in range(ni):
        fn = math.sqrt(sum(v * v for v in F[j]))
        scores.append(sum(prof[t] * F[j][t] for t in range(f)) / (pn * fn)
                      if fn > 0.0 else 0.0)
    order = sorted(range(ni), key=lambda j: (-scores[j], j))
    rec = [j for j in order if j not in rated][:int(topn)]
    return RichResult(payload={
        "estimate": scores[rec[0]] if rec else float("nan"),
        "scores": scores,
        "ranking": order,
        "recommended": rec,
        "profile": prof,
        "n_items": ni,
        "f": f,
        "method": "content-based recommendation",
    })


def cheatsheet():
    return "contRC: content-based recommendation"
