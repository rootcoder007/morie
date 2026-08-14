# morie.fn -- function file (rootcoder007/morie)
r"""GroupLens: predict from the people who agreed with you before.

The premise is stated as an architecture, not an algorithm: people who
agreed in the past will probably agree again, so a system can predict
how much someone will like an item from the ratings of users who
correlate with them -- **without any content analysis at all**. That
is what makes it work on Usenet news, where the items are text nobody
has modelled.

**The correlation is over co-rated items only.** Two users are
compared on the items they have *both* rated; everything else is
silent, not zero. Treating unrated as zero is the standard mistake,
and ``pearson`` refuses a pair with too little overlap instead of
returning a confident number computed from two points.

**Prediction is a weighted sum of DEVIATIONS, not of ratings.**

.. math:: \hat r_{ai} = \bar r_a + \frac{\sum_u w_{au}
          (r_{ui} - \bar r_u)}{\sum_u |w_{au}|},

because users differ in how they use the scale -- one person's 3 is
another's 5. Averaging raw ratings would import the neighbour's
generosity along with their opinion, and the anchor constructs exactly
that case.

**The denominator uses absolute weights**, so a strongly *negative*
correlate contributes without cancelling the normalisation -- a
neighbour who reliably disagrees is information.

**Significance weighting is the honest patch.** A correlation of 1.0
from two co-rated items is not evidence; scaling the weight by
:math:`\min(n/50, 1)` is the standard remedy, offered here rather than
left as a footnote.

References
----------
Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P. & Riedl, J.
(1994) "GroupLens: An Open Architecture for Collaborative Filtering of
Netnews", *Proceedings of the 1994 ACM Conference on Computer
Supported Cooperative Work (CSCW '94)*, 175-186,
doi:10.1145/192844.192905. [PDF supplied by Vee.] The premise that
people who agreed in the past are likely to agree again and that
predictions can therefore be made from correlated users' ratings with
no content analysis; the Pearson correlation computed over co-rated
items as the neighbour weight; and the prediction as the user's own
mean plus a weighted average of the neighbours' deviations from their
means, normalised by the sum of the absolute weights.

Herlocker, J. L., Konstan, J. A., Borchers, A. & Riedl, J. (1999)
"An Algorithmic Framework for Performing Collaborative Filtering",
*SIGIR '99*, 230-237, doi:10.1145/312624.312682. Significance
weighting for small overlaps.

Sarwar, B., Karypis, G., Konstan, J. & Riedl, J. (2001)
"Item-based Collaborative Filtering Recommendation Algorithms",
*WWW '01*, 285-295, doi:10.1145/371920.372071. The item-based
transpose of this.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["co_rated", "pearson", "neighbours", "predict_rating",
           "significance_weight"]

_EPS = 1e-12


def co_rated(ratings_a, ratings_b):
    r"""The items BOTH users rated. Unrated is silent, not zero."""
    A = dict(ratings_a)
    B = dict(ratings_b)
    common = sorted(set(A) & set(B))
    return {"items": common, "n": len(common),
            "a": [float(A[i]) for i in common],
            "b": [float(B[i]) for i in common],
            "note": "an unrated item carries no information; scoring "
                    "it as 0 would invent a strong opinion"}


def significance_weight(n_common, threshold=50):
    r""":math:`\min(n/50, 1)`.

    A correlation of 1.0 from two co-rated items is not evidence.
    """
    n = int(n_common)
    t = int(threshold)
    if t < 1:
        raise ValueError("ucfR: the threshold must be positive")
    return min(n / float(t), 1.0)


def pearson(ratings_a, ratings_b, min_common=2, significance=False,
            threshold=50):
    r"""Correlation over the co-rated items.

    Refuses a pair with too little overlap rather than returning a
    confident number from two points.
    """
    c = co_rated(ratings_a, ratings_b)
    n = c["n"]
    if n < int(min_common):
        raise ValueError("ucfR: only %d co-rated items, below the "
                         "minimum of %d -- a correlation here would "
                         "be noise" % (n, int(min_common)))
    ma = sum(c["a"]) / n
    mb = sum(c["b"]) / n
    num = sum((c["a"][i] - ma) * (c["b"][i] - mb) for i in range(n))
    da = math.sqrt(sum((v - ma) ** 2 for v in c["a"]))
    db = math.sqrt(sum((v - mb) ** 2 for v in c["b"]))
    if da <= _EPS or db <= _EPS:
        return {"w": 0.0, "n_common": n, "degenerate": True,
                "note": "one user gave identical ratings throughout, "
                        "so no correlation is defined"}
    w = num / (da * db)
    if significance:
        w *= significance_weight(n, threshold)
    return {"w": w, "n_common": n, "degenerate": False,
            "significance_applied": bool(significance)}


def neighbours(target, others, min_common=2, top_k=None,
               significance=False):
    r"""Rank the other users by correlation with the target."""
    out = []
    for uid, r in others.items():
        try:
            p = pearson(target, r, min_common, significance)
        except ValueError:
            continue
        if p.get("degenerate"):
            continue
        out.append({"user": uid, "w": p["w"],
                    "n_common": p["n_common"]})
    out.sort(key=lambda d: -abs(d["w"]))
    if top_k is not None:
        out = out[:int(top_k)]
    return {"neighbours": out, "n": len(out),
            "note": "ranked by |w|: a reliable DISAGREER is "
                    "information too"}


def predict_rating(target, others, item, min_common=2, top_k=None,
                   significance=False):
    r""":math:`\bar r_a + \sum w(r_{ui}-\bar r_u)/\sum|w|`.

    Deviations, not raw ratings: otherwise a generous neighbour's
    scale comes along with their opinion.
    """
    A = dict(target)
    if not A:
        raise ValueError("ucfR: the target user has rated nothing, "
                         "so there is no mean to anchor on")
    mean_a = sum(float(v) for v in A.values()) / len(A)
    nb = neighbours(target, others, min_common, top_k,
                    significance)["neighbours"]
    num, den, used = 0.0, 0.0, 0
    naive_num, naive_den = 0.0, 0.0
    for d in nb:
        R = dict(others[d["user"]])
        if item not in R:
            continue
        mu = sum(float(v) for v in R.values()) / len(R)
        num += d["w"] * (float(R[item]) - mu)
        den += abs(d["w"])
        naive_num += d["w"] * float(R[item])
        naive_den += d["w"]
        used += 1
    if used == 0 or den <= _EPS:
        return RichResult(payload={
            "estimate": mean_a, "prediction": mean_a,
            "n_neighbours": 0, "fell_back": True,
            "method": "user-based collaborative filtering; Resnick "
                      "et al. (1994)",
            "note": "nobody comparable rated this item, so the "
                    "user's own mean is the honest answer",
        })
    return RichResult(payload={
        "estimate": mean_a + num / den,
        "prediction": mean_a + num / den,
        "naive_weighted_mean": naive_num / naive_den
        if abs(naive_den) > _EPS else None,
        "user_mean": mean_a, "n_neighbours": used,
        "fell_back": False,
        "method": "user-based collaborative filtering; Resnick et "
                  "al. (1994)",
        "note": "deviations from each neighbour's own mean, "
                "normalised by the sum of ABSOLUTE weights",
    })


def cheatsheet():
    return ("ucfR: people who agreed before will probably agree again "
            "-- so predict from correlated users, with NO content "
            "analysis, which is why it worked on Usenet news. "
            "Correlate over CO-RATED items only; unrated is silent, "
            "not zero. Predict the user's own mean plus a weighted "
            "average of neighbours' DEVIATIONS from their means, since "
            "one person's 3 is another's 5 -- averaging raw ratings "
            "imports the neighbour's generosity. Normalise by the sum "
            "of ABSOLUTE weights, so a reliable disagreer still "
            "counts. A correlation of 1.0 from two co-rated items is "
            "not evidence: scale by min(n/50, 1).")


# compact alias per ledger/NAMING.md
user_based_cf = predict_rating

# public names resolved by fn/_lazy_map.json
user_cf = predict_rating
usercf = predict_rating
