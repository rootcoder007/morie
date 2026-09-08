# morie.fn -- function file (rootcoder007/morie)
r"""WARP: optimise the top of the ranking, cheaply.

With tens of thousands of possible labels, the metric that matters is
precision at :math:`k` -- what appears in the top few. Pairwise
ranking losses optimise the *whole* ordering instead, so they spend
most of their gradient on distinctions far below anything a user sees;
and losses that do target the top are costly to train.

**The trick is to estimate a rank by sampling, not by computing it.**
For a positive label, draw negatives uniformly until one **violates**
the margin. If it took :math:`N` draws out of :math:`Y-1` candidates,
the violating rank is estimated as
:math:`\lfloor (Y-1)/N \rfloor` -- so a violation found on the first
draw means a badly ranked positive, and one that took many draws means
the positive is already near the top. Nothing is sorted; the estimate
falls out of the number of attempts.

**That estimate is then WEIGHTED**, and the weight is where "top of
the list" enters. With :math:`L(r) = \sum_{j\le r}\alpha_j` and
:math:`\alpha_1 \ge \alpha_2 \ge \dots \ge 0`, a decreasing
:math:`\alpha` makes an error at rank 1 cost more than an error at
rank 50. Choosing :math:`\alpha_j = 1/j` optimises the top; choosing
:math:`\alpha_j` constant recovers the ordinary pairwise loss that
weights every position alike -- and the anchor compares the two rather
than asserting the difference.

**The sampling cost is self-limiting in the right direction.** A
well-ranked positive takes many draws to violate, so it is expensive
exactly when it has least to teach; capping the draws at :math:`Y-1`
bounds that, and ``sample_violation`` reports whether the cap was hit
instead of silently returning a rank of zero.

References
----------
Weston, J., Bengio, S. & Usunier, N. (2010) "Large Scale Image
Annotation: Learning to Rank with Joint Word-Image Embeddings",
*Machine Learning and Knowledge Discovery in Databases (ECML PKDD
2010)*, LNCS 6323, 21-35, doi:10.1007/978-3-642-15939-8_2. [PDF
supplied by Vee.] The WARP (Weighted Approximate-Rank Pairwise) loss:
that measures optimising for the top annotations, such as precision at
k, are costly to train; the relation to the Ordered Weighted Pairwise
Classification loss; the use of stochastic gradient descent with a
sampling trick to APPROXIMATE ranks, giving an efficient online
strategy superior to standard SGD on the same loss and able to train
on datasets that do not fit in memory; and its applicability to
arbitrary differentiable models, unlike the OWPC loss which relies on
SVMstruct.

Usunier, N., Buffoni, D. & Gallinari, P. (2009) "Ranking with ordered
weighted pairwise classification", *ICML 2009*, 1057-1064,
doi:10.1145/1553374.1553509. The ordered weighted pairwise loss and
the alpha weights.

Weston, J., Bengio, S. & Usunier, N. (2011) "WSABIE: Scaling Up to
Large Vocabulary Image Annotation", *IJCAI 2011*, 2764-2770. The
later, more widely cited presentation of the same loss. NOTE: the
ECML 2010 paper above is the one held locally and is the text this
module follows.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["alpha_weights", "rank_weight", "estimate_rank",
           "sample_violation", "warp_loss", "warp_step"]

_EPS = 1e-12


def alpha_weights(n, scheme="reciprocal"):
    r""":math:`\alpha_1\ge\alpha_2\ge\dots\ge 0`.

    ``reciprocal`` (:math:`1/j`) optimises the top; ``uniform``
    recovers the ordinary pairwise loss that treats every rank alike.
    """
    N = int(n)
    if N < 1:
        raise ValueError("warpL: n must be at least 1")
    if scheme == "reciprocal":
        a = [1.0 / (j + 1) for j in range(N)]
    elif scheme == "uniform":
        a = [1.0] * N
    elif scheme == "top1":
        a = [1.0] + [0.0] * (N - 1)
    else:
        raise ValueError("warpL: scheme must be reciprocal, uniform "
                         "or top1, got %r" % (scheme,))
    if any(a[j] < a[j + 1] - _EPS for j in range(N - 1)):
        raise ValueError("warpL: the alpha weights must be "
                         "non-increasing")
    return a


def rank_weight(rank, alphas):
    r""":math:`L(r) = \sum_{j\le r}\alpha_j` -- the cost of being at
    rank :math:`r`."""
    r = int(rank)
    if r < 0:
        raise ValueError("warpL: the rank cannot be negative")
    return sum(alphas[:min(r, len(alphas))])


def estimate_rank(n_draws, n_labels):
    r""":math:`\lfloor (Y-1)/N\rfloor` from the number of draws.

    No sorting: a violation on the first draw means a badly ranked
    positive; many draws means it is already near the top.
    """
    N = int(n_draws)
    Y = int(n_labels)
    if N < 1 or Y < 2:
        raise ValueError("warpL: need at least one draw and two "
                         "labels")
    return int((Y - 1) // N)


def sample_violation(score_positive, negative_scorer, n_labels,
                     rng, margin=1.0, max_draws=None):
    r"""Draw negatives until one violates the margin.

    Reports whether the cap was reached, since "no violation found"
    and "rank 0" are different facts.
    """
    Y = int(n_labels)
    cap = (Y - 1) if max_draws is None else int(max_draws)
    if cap < 1:
        raise ValueError("warpL: at least one draw is required")
    for t in range(1, cap + 1):
        j = int(float(rng.uniform()) * (Y - 1)) % (Y - 1)
        s = float(negative_scorer(j))
        if s > float(score_positive) - float(margin):
            return {"violated": True, "draws": t, "negative": j,
                    "negative_score": s,
                    "estimated_rank": estimate_rank(t, Y),
                    "capped": False}
    return {"violated": False, "draws": cap, "negative": None,
            "estimated_rank": 0, "capped": True,
            "note": "no violator found within the cap: the positive "
                    "is already well ranked, which is exactly when "
                    "sampling is most expensive"}


def warp_loss(score_positive, score_negative, estimated_rank, alphas,
              margin=1.0):
    r""":math:`L(\hat r)\,|1 - f(pos) + f(neg)|_+`."""
    hinge = max(0.0, float(margin) - float(score_positive)
                + float(score_negative))
    w = rank_weight(int(estimated_rank), alphas)
    return {"loss": w * hinge, "hinge": hinge, "rank_weight": w,
            "estimated_rank": int(estimated_rank)}


def warp_step(positive, negatives, embed_user, rng, alphas,
              lr=0.05, margin=1.0):
    r"""One sampled update.

    ``negatives`` is the full candidate list; only the drawn one is
    ever scored beyond the sampling loop, which is the saving.
    """
    u = [float(v) for v in k.vec(embed_user)]
    P = [float(v) for v in k.vec(positive)]
    Y = len(negatives) + 1

    def score(v):
        w = [float(x) for x in k.vec(v)]
        return sum(u[a] * w[a] for a in range(len(u)))

    sp = score(P)
    v = sample_violation(sp, lambda j: score(negatives[j]), Y, rng,
                         margin)
    if not v["violated"]:
        return {"updated": False, "draws": v["draws"],
                "loss": 0.0, "user": u,
                "note": "nothing violated the margin, so there is "
                        "nothing to learn from this positive"}
    neg = [float(x) for x in k.vec(negatives[v["negative"]])]
    L = warp_loss(sp, score(negatives[v["negative"]]),
                  v["estimated_rank"], alphas, margin)
    g = L["rank_weight"] * float(lr)
    new_u = [u[a] + g * (P[a] - neg[a]) for a in range(len(u))]
    return RichResult(payload={
        "estimate": L["loss"], "updated": True, "loss": L["loss"],
        "user": new_u, "draws": v["draws"],
        "estimated_rank": v["estimated_rank"],
        "rank_weight": L["rank_weight"],
        "negative": v["negative"],
        "method": "WARP sampled rank approximation; Weston, Bengio & "
                  "Usunier (2010)",
        "note": "the step size scales with L(rank), so an error at "
                "the top of the list moves the model further",
    })


def cheatsheet():
    return ("warpL: with tens of thousands of labels what matters is "
            "precision at k, but pairwise losses optimise the WHOLE "
            "ordering and top-targeting losses are costly to train. "
            "Estimate the rank by SAMPLING: draw negatives until one "
            "violates the margin, and if it took N draws the rank is "
            "about (Y-1)/N -- a violation on the first draw means a "
            "badly ranked positive, many draws means it is already "
            "near the top. Nothing is sorted. Then WEIGHT by "
            "L(r) = sum_{j<=r} alpha_j with alpha non-increasing: "
            "alpha_j = 1/j optimises the top, constant alpha recovers "
            "the plain pairwise loss. Cap the draws and SAY when the "
            "cap was hit.")


# compact alias per ledger/NAMING.md
warp_rank_loss = warp_step

# public names resolved by fn/_lazy_map.json
warp = warp_step
