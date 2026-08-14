# morie.fn -- function file (rootcoder007/morie)
r"""DINOv2: general visual features from CURATED data.

Self-supervised learning had produced good features on ImageNet-1k and
lost them when scaled to uncurated data -- the drop is attributed to a
lack of control over data quality and diversity, not to the objective.
DINOv2's claim is that discriminative self-supervision *does* produce
general-purpose features, given enough curated data, and most of its
technical contributions exist to make training at that scale stable.

**The curation pipeline uses similarity, not metadata.** Uncurated
images are embedded, **deduplicated**, and then **retrieved** against a
small curated corpus, so the curated set is augmented by its own
nearest neighbours rather than by anyone's labels. The failure it
guards against is stated plainly: images in the wild over-represent a
few dominant modes, and a naive clustering rebalance resolves it well
enough. 142M images.

**KoLeo keeps the batch from collapsing to a few directions.** From
the Kozachenko-Leonenko differential entropy estimator,

.. math:: L_{koleo} = -\frac{1}{n}\sum_{i=1}^{n}\log d_{n,i},

with :math:`d_{n,i}` the distance from :math:`x_i` to its nearest
neighbour in the batch. It is *minimised* by a uniform spread and
blows up when two features coincide -- which ``koleo`` demonstrates by
returning a strictly larger loss for a clustered batch than for a
spread one.

**Features are learned at two levels**: image-level self-distillation
and patch-level masked prediction (iBOT), which is why the features
work for dense tasks and not only for classification.

References
----------
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M.,
Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A. et
al. (2024) "DINOv2: Learning Robust Visual Features without
Supervision", *Transactions on Machine Learning Research* (01/2024),
arXiv:2304.07193. Sec. 1 (that scaling self-supervised approaches
beyond ImageNet-1k focused on uncurated datasets and led to a
significant drop in feature quality, explained by the lack of control
over data quality and diversity; the revisiting of iBOT-style methods
learning features at both the image and patch level; and the 142M
curated corpus), Sec. 2 (the automatic data pipeline mapping curated
and uncurated images to embeddings, deduplicating the uncurated ones
and matching them to curated images through a self-supervised
retrieval system, with a naive clustering approach used to rebalance
concepts), and Sec. 4 / Appendix (the KoLeo regularizer derived from
the Kozachenko-Leonenko differential entropy estimator, defined as
-(1/n) sum log d_{n,i} and encouraging a uniform span of features
within a batch, applied with weight 0.1 between the class tokens of
the first global crop; and the Sinkhorn-Knopp centering run for 3
iterations).

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., Bojanowski,
P. & Joulin, A. (2021) "Emerging Properties in Self-Supervised Vision
Transformers", *ICCV 2021*, 9650-9660, arXiv:2104.14294. DINO, the
self-distillation being scaled.

Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A. & Kong, T.
(2022) "iBOT: Image BERT Pre-Training with Online Tokenizer",
*ICLR 2022*, arXiv:2111.07832. The patch-level objective.

Sablayrolles, A., Douze, M., Schmid, C. & Jegou, H. (2019)
"Spreading vectors for similarity search", *ICLR 2019*,
arXiv:1806.03198. The KoLeo regularizer.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["deduplicate", "retrieve_augment", "koleo",
           "sinkhorn_knopp", "self_distillation_loss"]

_EPS = 1e-12


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > _EPS else list(v)


def _cos(a, b):
    x, y = _norm(a), _norm(b)
    return sum(x[i] * y[i] for i in range(len(x)))


def deduplicate(embeddings, threshold=0.999):
    r"""Drop near-duplicates before anything else uses them."""
    E = [[float(v) for v in r] for r in k.mat(embeddings)]
    keep, dropped = [], []
    for i in range(len(E)):
        dup = None
        for j in keep:
            if _cos(E[i], E[j]) >= float(threshold):
                dup = j
                break
        if dup is None:
            keep.append(i)
        else:
            dropped.append((i, dup))
    return {"keep": keep, "dropped": dropped,
            "n_before": len(E), "n_after": len(keep),
            "note": "similarity, not metadata -- no annotation is "
                    "required"}


def retrieve_augment(curated, uncurated, per_query=2,
                     min_similarity=0.0):
    r"""Augment the curated set with its own nearest neighbours.

    Retrieval against curated images is what supplies diversity
    without a label; a cluster rebalance is what stops a few dominant
    modes from taking over.
    """
    C = [[float(v) for v in r] for r in k.mat(curated)]
    U = [[float(v) for v in r] for r in k.mat(uncurated)]
    if not C:
        raise ValueError("dnvtwo: the curated corpus is empty, so "
                         "there is nothing to retrieve against")
    picked, per = [], {}
    for qi, c in enumerate(C):
        sims = sorted(((_cos(c, U[j]), j) for j in range(len(U))),
                      reverse=True)
        got = [j for s, j in sims[:int(per_query)]
               if s >= float(min_similarity)]
        per[qi] = got
        picked.extend(got)
    counts = {}
    for j in picked:
        counts[j] = counts.get(j, 0) + 1
    return {"retrieved": sorted(set(picked)), "per_query": per,
            "duplication": counts,
            "n_added": len(set(picked)),
            "max_times_retrieved": max(counts.values()) if counts
            else 0,
            "note": "a few dominant modes would otherwise be "
                    "retrieved by every query"}


def koleo(features):
    r""":math:`-\frac{1}{n}\sum_i \log d_{n,i}`.

    Minimised by a uniform span; two coincident features send it to
    infinity, which is exactly the collapse it exists to prevent.
    """
    F = [_norm([float(v) for v in r]) for r in k.mat(features)]
    n = len(F)
    if n < 2:
        raise ValueError("dnvtwo: KoLeo needs at least 2 features")
    tot, dmin = 0.0, []
    for i in range(n):
        d = min(math.sqrt(sum((F[i][a] - F[j][a]) ** 2
                              for a in range(len(F[i]))))
                for j in range(n) if j != i)
        dmin.append(d)
        tot += math.log(max(d, _EPS))
    return {"loss": -tot / n, "nearest_distances": dmin,
            "min_distance": min(dmin),
            "note": "collapsed features give a huge loss; a uniform "
                    "span gives the smallest"}


def sinkhorn_knopp(scores, iterations=3, epsilon=0.05):
    r"""Centering by 3 Sinkhorn-Knopp iterations, as in SwAV.

    Rows and columns are pushed toward equal mass, which is what stops
    every image being assigned to the same prototype.
    """
    S = [[float(v) for v in r] for r in k.mat(scores)]
    n, K = len(S), len(S[0])
    Q = [[math.exp(S[i][j] / float(epsilon)) for j in range(K)]
         for i in range(n)]
    tot = sum(sum(r) for r in Q) or 1.0
    Q = [[v / tot for v in r] for r in Q]
    for _ in range(int(iterations)):
        for j in range(K):
            c = sum(Q[i][j] for i in range(n)) or 1.0
            for i in range(n):
                Q[i][j] = Q[i][j] / c / K
        for i in range(n):
            r = sum(Q[i]) or 1.0
            for j in range(K):
                Q[i][j] = Q[i][j] / r / n
    return {"Q": [[v * n for v in r] for r in Q],
            "iterations": int(iterations),
            "row_sums": [sum(r) * n for r in Q],
            "note": "3 iterations, as specified"}


def self_distillation_loss(student, teacher, temperature_s=0.1,
                           temperature_t=0.04, patch_level=False):
    r"""Cross entropy of student against a sharpened teacher.

    ``patch_level=True`` is the iBOT term -- the reason the features
    work for dense prediction and not only for classification.
    """
    s = [float(v) for v in k.vec(student)]
    t = [float(v) for v in k.vec(teacher)]
    if len(s) != len(t):
        raise ValueError("dnvtwo: the student and teacher outputs "
                         "differ in width")
    ts, tt = float(temperature_s), float(temperature_t)
    if ts <= 0.0 or tt <= 0.0:
        raise ValueError("dnvtwo: the temperatures must be positive")

    def soft(x, T):
        m = max(x)
        e = [math.exp((v - m) / T) for v in x]
        z = sum(e)
        return [v / z for v in e]

    ps, pt = soft(s, ts), soft(t, tt)
    loss = -sum(pt[i] * math.log(max(ps[i], _EPS))
                for i in range(len(ps)))
    return RichResult(payload={
        "estimate": loss, "loss": loss, "student": ps, "teacher": pt,
        "level": "patch" if patch_level else "image",
        "teacher_entropy": -sum(v * math.log(max(v, _EPS))
                                for v in pt),
        "method": "self-distillation with a sharpened teacher; "
                  "Oquab et al. (2024)",
        "note": "the teacher is sharper (lower temperature), which is "
                "what gives the student a target to move toward",
    })


def cheatsheet():
    return ("dnvtwo: self-supervision lost feature quality when scaled "
            "to UNCURATED data -- the cause is data quality and "
            "diversity, not the objective. So CURATE automatically: "
            "embed, DEDUPLICATE, then RETRIEVE uncurated images "
            "against a small curated corpus, using similarity rather "
            "than metadata, and rebalance clusters so a few dominant "
            "modes do not take over. Learn at BOTH image level "
            "(self-distillation) and patch level (iBOT), which is why "
            "the features work for dense tasks. KoLeo = -(1/n) sum log "
            "d_{n,i} keeps the batch spread out; coincident features "
            "send it to infinity.")


# compact alias per ledger/NAMING.md
dinov2 = self_distillation_loss
