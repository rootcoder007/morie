# morie.fn -- function file (rootcoder007/morie)
r"""COMET: MT evaluation trained against human judgement.

BLEU compares a hypothesis with a reference by n-gram overlap, and
correlates only moderately with what humans actually judge. COMET
replaces the surface comparison with a **learned** one: a
cross-lingual encoder embeds the hypothesis, the reference **and the
source**, and a feed-forward head is trained to predict human quality
scores.

**Using the source is the structural difference.** A reference-only
metric cannot tell a hypothesis that is wrong *about the source* from
one that merely words the reference differently -- the second is fine
and the first is a mistranslation. Feeding the source lets the metric
separate them, and it is also what makes a **reference-free** variant
possible at all.

**Two architectures, and they answer different questions.** The
*estimator* regresses a human score directly (trained on DA or MQM
judgements, minimising mean squared error). The *translation ranking*
model instead learns from relative judgements: given a better and a
worse hypothesis, push the better one closer to source and reference
in embedding space with a **triplet margin** loss. Ranking data is far
easier to collect than calibrated absolute scores, which is why both
exist.

**The pooled features are not just concatenation.** The estimator
receives element-wise products and absolute differences between the
hypothesis embedding and each of the source and reference -- the same
:math:`(u, v, |u-v|)` reasoning as sentence-pair models, and the part
that lets a feed-forward head express *disagreement* rather than mere
co-location.

**Segment-level correlation is the evaluation that matters.** A metric
can look good on system-level averages while being useless per
sentence, so the paper reports Kendall's tau at segment level.

References
----------
Rei, R., Stewart, C., Farinha, A. C. & Lavie, A. (2020) "COMET: A
Neural Framework for MT Evaluation", *Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP
2020)*, 2685-2702, doi:10.18653/v1/2020.emnlp-main.213,
arXiv:2009.09025. The framework training multilingual machine
translation evaluation models that exploit information from both the
SOURCE and the reference; the two architectures -- an estimator model
regressing human quality scores and a translation ranking model
trained on relative rankings with a triplet margin objective; the
pooled features combining element-wise products and differences; and
evaluation by segment-level correlation with human judgements.

Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. (2002) "BLEU", *ACL
2002*, 311-318, doi:10.3115/1073083.1073135. The surface metric being
replaced; implemented in :mod:`sacrb`.

Reimers, N. & Gurevych, I. (2019) "Sentence-BERT", *EMNLP-IJCNLP
2019*, 3980-3990, doi:10.18653/v1/D19-1410. The (u, v, |u-v|) feature
construction; implemented in :mod:`sbert`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["pooled_features", "estimator_score", "triplet_loss",
           "kendall_tau", "reference_free"]

_EPS = 1e-12


def _vec(x):
    return [float(v) for v in k.vec(x)]


def pooled_features(hyp, src, ref):
    r"""Element-wise products and absolute differences against BOTH.

    Concatenation alone cannot express disagreement; the difference
    terms are what locate it.
    """
    h, s, r = _vec(hyp), _vec(src), _vec(ref)
    if not (len(h) == len(s) == len(r)):
        raise ValueError("comet: the three embeddings differ in "
                         "length (%d, %d, %d)"
                         % (len(h), len(s), len(r)))
    d = len(h)
    hs = [h[i] * s[i] for i in range(d)]
    hr = [h[i] * r[i] for i in range(d)]
    ds = [abs(h[i] - s[i]) for i in range(d)]
    dr = [abs(h[i] - r[i]) for i in range(d)]
    return {"features": h + r + hr + dr + hs + ds,
            "dim": 6 * d, "hyp_ref_diff": dr, "hyp_src_diff": ds,
            "note": "the SOURCE enters too, which is what separates a "
                    "mistranslation from a differently-worded correct "
                    "translation"}


def estimator_score(hyp, src, ref, W, b=None):
    r"""The estimator head: regress a human quality score."""
    f = pooled_features(hyp, src, ref)["features"]
    if len(W[0]) != len(f):
        raise ValueError("comet: the head expects %d features but got "
                         "%d" % (len(W[0]), len(f)))
    bb = [0.0] * len(W) if b is None else _vec(b)
    z = [bb[o] + sum(W[o][j] * f[j] for j in range(len(f)))
         for o in range(len(W))]
    return RichResult(payload={
        "estimate": z[0] if len(z) == 1 else z,
        "score": z[0] if len(z) == 1 else z,
        "method": "COMET estimator; Rei, Stewart, Farinha & Lavie "
                  "(2020)",
        "note": "trained against HUMAN judgements, not n-gram "
                "overlap",
    })


def triplet_loss(better, worse, src, ref, margin=1.0):
    r"""The ranking model: push the better hypothesis closer.

    Relative judgements are far cheaper to collect than calibrated
    absolute scores, which is why this variant exists alongside the
    estimator.
    """
    def dist(a, b):
        x, y = _vec(a), _vec(b)
        if len(x) != len(y):
            raise ValueError("comet: embeddings differ in length")
        return math.sqrt(sum((x[i] - y[i]) ** 2
                             for i in range(len(x))))

    m = float(margin)
    if m <= 0.0:
        raise ValueError("comet: the margin must be positive")
    ls = max(0.0, dist(better, src) - dist(worse, src) + m)
    lr = max(0.0, dist(better, ref) - dist(worse, ref) + m)
    return {"loss": ls + lr, "source_term": ls, "reference_term": lr,
            "satisfied": (ls + lr) == 0.0,
            "note": "zero loss means the better hypothesis is already "
                    "closer to BOTH anchors by the margin"}


def kendall_tau(scores, human):
    r"""Segment-level Kendall's tau.

    A metric can look strong on system-level averages while being
    useless per sentence, so this is the number that matters.
    """
    a = _vec(scores)
    b = _vec(human)
    if len(a) != len(b):
        raise ValueError("comet: %d scores but %d human judgements"
                         % (len(a), len(b)))
    n = len(a)
    if n < 2:
        raise ValueError("comet: at least 2 segments are needed")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0.0 or db == 0.0:
                continue
            if (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    tot = conc + disc
    return {"tau": (conc - disc) / float(tot) if tot else 0.0,
            "concordant": conc, "discordant": disc,
            "n_segments": n}


def reference_free(hyp, src, W, b=None):
    r"""Quality estimation with no reference at all.

    Possible only because the source is already part of the input --
    a reference-only metric has nothing left to compare against.
    """
    h, s = _vec(hyp), _vec(src)
    if len(h) != len(s):
        raise ValueError("comet: the embeddings differ in length")
    d = len(h)
    f = h + s + [h[i] * s[i] for i in range(d)] + \
        [abs(h[i] - s[i]) for i in range(d)]
    if len(W[0]) != len(f):
        raise ValueError("comet: the reference-free head expects %d "
                         "features but got %d" % (len(W[0]), len(f)))
    bb = [0.0] * len(W) if b is None else _vec(b)
    z = [bb[o] + sum(W[o][j] * f[j] for j in range(len(f)))
         for o in range(len(W))]
    return {"score": z[0] if len(z) == 1 else z,
            "reference_used": False,
            "note": "only possible because the source was always part "
                    "of the model"}


def cheatsheet():
    return ("comet: replace n-gram overlap with a LEARNED metric "
            "trained on human judgements, embedding hypothesis, "
            "reference AND SOURCE. The source is the structural "
            "difference: without it you cannot separate a "
            "MISTRANSLATION from a correct translation worded "
            "differently from the reference -- and it is what makes a "
            "reference-free variant possible. Two heads: an ESTIMATOR "
            "regressing absolute scores, and a RANKING model with a "
            "triplet margin, because relative judgements are far "
            "cheaper to collect. Report SEGMENT-level Kendall tau.")


# compact alias per ledger/NAMING.md
cometmetric = estimator_score

# public names resolved by fn/_lazy_map.json
comet = estimator_score
