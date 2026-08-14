# morie.fn -- function file (rootcoder007/morie)
r"""Two-tower retrieval, and the sampling bias you must remove.

Retrieval over a corpus of millions of items cannot afford a softmax
over the corpus, so the standard trick is to treat the **other items
in the batch** as negatives. That is efficient and it is biased:
in-batch negatives are drawn from the *training distribution*, so a
popular item appears as a negative constantly and is pushed down
regardless of relevance.

**The correction is one subtraction.** With :math:`p_j` the
probability that item :math:`j` appears in a batch, use the corrected
logit

.. math:: s^c(x, y_j) = s(x, y_j) - \log p_j,

which is the standard logQ correction for sampled softmax, applied to
in-batch sampling. ``corrected_logits`` implements it, and the anchor
constructs a case where the *uncorrected* ranking puts a popular
irrelevant item above a rare relevant one and the correction restores
the right order -- so the term is shown to matter rather than
described.

**The item frequency is estimated in a stream, not counted.** The
paper's estimator tracks, per item, the number of steps since it was
last seen, and updates a running average :math:`B` of that gap; the
sampling probability is :math:`1/B`. No global count, no second pass,
and it adapts as the distribution drifts. ``streaming_frequency``
implements exactly that, and the anchor checks it converges to
:math:`1/\Delta` for an item that appears every :math:`\Delta` steps.

**Normalisation and temperature are not cosmetic.** Embeddings are
L2-normalised and the inner product divided by a temperature, which
sharpens the softmax; without normalisation the model can win the
contrastive objective by inflating norms rather than by learning
directions.

References
----------
Yi, X., Yang, J., Hong, L., Cheng, D. Z., Heldt, L., Kumthekar, A.,
Zhao, Z., Wei, L. & Chi, E. (2019) "Sampling-Bias-Corrected Neural
Modeling for Large Corpus Item Recommendations", *Proceedings of the
13th ACM Conference on Recommender Systems (RecSys '19)*, 269-277,
doi:10.1145/3298689.3346996. The two-tower architecture for
large-corpus item retrieval; batch softmax with in-batch negatives and
the resulting sampling bias toward popular items; the logQ correction
subtracting log p_j from the logit; the streaming frequency estimation
algorithm tracking the number of steps between successive hits of an
item and estimating the sampling probability as its reciprocal; and
normalisation of the embeddings with a temperature in the softmax.

Bengio, Y. & Senecal, J.-S. (2008) "Adaptive Importance Sampling to
Accelerate Training of a Neural Probabilistic Language Model", *IEEE
Transactions on Neural Networks* 19(4), 713-722,
doi:10.1109/TNN.2007.912312. The sampled-softmax correction being
applied.

Covington, P., Adams, J. & Sargin, E. (2016) "Deep Neural Networks
for YouTube Recommendations", *RecSys 2016*, 191-198,
doi:10.1145/2959100.2959190. The retrieval setting.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tower_embedding", "corrected_logits",
           "streaming_frequency", "batch_softmax_loss", "retrieve"]

_EPS = 1e-12


def tower_embedding(features, W, b=None, normalise=True):
    r"""One tower: features to a vector, L2-normalised.

    Without normalisation the contrastive objective can be won by
    growing norms instead of by learning directions.
    """
    x = [float(v) for v in k.vec(features)]
    if len(W[0]) != len(x):
        raise ValueError("twoT: the tower expects %d features but "
                         "got %d" % (len(W[0]), len(x)))
    bb = [0.0] * len(W) if b is None else [float(v) for v in k.vec(b)]
    z = [bb[o] + sum(W[o][j] * x[j] for j in range(len(x)))
         for o in range(len(W))]
    if not normalise:
        return {"embedding": z, "normalised": False}
    n = math.sqrt(sum(v * v for v in z))
    if n <= _EPS:
        raise ValueError("twoT: the tower produced a zero embedding")
    return {"embedding": [v / n for v in z], "norm": n,
            "normalised": True}


def corrected_logits(scores, probabilities, temperature=1.0):
    r""":math:`s^c = s - \log p_j`, the logQ correction.

    In-batch negatives are drawn from the training distribution, so a
    popular item is pushed down for being popular.
    """
    s = [float(v) for v in k.vec(scores)]
    p = [float(v) for v in k.vec(probabilities)]
    if len(s) != len(p):
        raise ValueError("twoT: %d scores but %d probabilities"
                         % (len(s), len(p)))
    if any(v <= 0.0 or v > 1.0 for v in p):
        raise ValueError("twoT: the sampling probabilities must lie "
                         "in (0,1]")
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("twoT: the temperature must be positive")
    raw = [v / t for v in s]
    cor = [raw[i] - math.log(p[i]) for i in range(len(s))]
    return {"corrected": cor, "raw": raw,
            "shift": [-math.log(v) for v in p],
            "note": "a frequent item gets the LARGEST positive "
                    "shift, undoing its over-representation"}


def streaming_frequency(hits, n_steps, alpha=0.05, init=None):
    r"""Estimate :math:`p_j` from the GAPS between hits.

    ``hits`` maps a step index to the items seen at that step. The
    running average :math:`B_j` of the gap since item :math:`j` was
    last seen gives :math:`p_j = 1/B_j` -- no global count and no
    second pass, so it tracks a drifting distribution.
    """
    a = float(alpha)
    if not 0.0 < a <= 1.0:
        raise ValueError("twoT: the step size must lie in (0,1]")
    last, B = {}, {}
    for t in range(int(n_steps)):
        for j in hits.get(t, ()):
            if j in last:
                gap = t - last[j]
                B[j] = (1.0 - a) * B[j] + a * gap
            else:
                B[j] = float(init) if init is not None else 1.0
            last[j] = t
    return {"B": B,
            "probability": {j: 1.0 / max(v, _EPS)
                            for j, v in B.items()},
            "n_items": len(B),
            "note": "B is the average number of steps between hits, "
                    "so 1/B is the sampling probability"}


def batch_softmax_loss(query_embeddings, item_embeddings,
                       probabilities=None, temperature=0.05):
    r"""In-batch softmax, corrected when probabilities are supplied."""
    Q = [[float(v) for v in r] for r in k.mat(query_embeddings)]
    I = [[float(v) for v in r] for r in k.mat(item_embeddings)]
    n = len(Q)
    if len(I) != n:
        raise ValueError("twoT: %d queries but %d items"
                         % (n, len(I)))
    if n < 2:
        raise ValueError("twoT: in-batch negatives need at least 2 "
                         "examples")
    tot, per = 0.0, []
    for i in range(n):
        s = [sum(Q[i][a] * I[j][a] for a in range(len(Q[0])))
             for j in range(n)]
        if probabilities is None:
            lg = [v / float(temperature) for v in s]
        else:
            lg = corrected_logits(s, probabilities,
                                  temperature)["corrected"]
        m = max(lg)
        z = sum(math.exp(v - m) for v in lg)
        li = -(lg[i] - m - math.log(z))
        per.append(li)
        tot += li
    return {"loss": tot / n, "per_example": per,
            "corrected": probabilities is not None}


def retrieve(query_embedding, item_embeddings, probabilities=None,
             top_k=5, temperature=1.0):
    r"""Rank the corpus, with and without the correction."""
    q = [float(v) for v in k.vec(query_embedding)]
    I = [[float(v) for v in r] for r in k.mat(item_embeddings)]
    s = [sum(q[a] * I[j][a] for a in range(len(q)))
         for j in range(len(I))]
    raw_order = sorted(range(len(s)), key=lambda j: -s[j])
    if probabilities is None:
        order, cor = raw_order, s
    else:
        cor = corrected_logits(s, probabilities,
                               temperature)["corrected"]
        order = sorted(range(len(cor)), key=lambda j: -cor[j])
    kk = min(int(top_k), len(order))
    return RichResult(payload={
        "estimate": order[:kk], "top_k": order[:kk],
        "uncorrected_top_k": raw_order[:kk],
        "scores": s, "corrected_scores": cor,
        "changed": order[:kk] != raw_order[:kk],
        "method": "sampling-bias-corrected two-tower retrieval; Yi "
                  "et al. (2019)",
        "note": "without the correction, popularity is mistaken for "
                "irrelevance",
    })


def cheatsheet():
    return ("twoT: a softmax over millions of items is impossible, so "
            "use IN-BATCH negatives -- which are drawn from the "
            "TRAINING distribution, so a popular item is pushed down "
            "for being popular. Correct it with one subtraction: "
            "s^c = s - log p_j. Estimate p_j in a STREAM from the "
            "average gap between an item's hits, p = 1/B -- no global "
            "count, no second pass, and it tracks drift. L2-normalise "
            "the towers and divide by a temperature, or the model wins "
            "the objective by inflating norms rather than learning "
            "directions.")


# compact alias per ledger/NAMING.md
twotower = retrieve

# public names resolved by fn/_lazy_map.json
two_tower = retrieve
