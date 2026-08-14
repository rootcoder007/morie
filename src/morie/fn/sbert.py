# morie.fn -- function file (rootcoder007/morie)
r"""Sentence-BERT: sentence embeddings that can actually be compared.

BERT scores a sentence *pair* by concatenating both with a ``[SEP]``
and running the whole thing through the network. That sets the state of
the art on semantic textual similarity and is useless for search.
Finding the most similar pair among :math:`n` sentences needs
:math:`\binom{n}{2}` forward passes -- for 10,000 sentences, about 50
million. The embeddings never exist separately, so nothing can be
indexed.

**The workarounds were never validated.** Passing single sentences
through BERT and averaging the outputs, or taking the ``[CLS]`` vector,
is what the popular tooling does. As of this paper there had been no
evaluation of whether either produces useful sentence embeddings -- and
the paper's finding is that they do not, on their own.

**The fix is a siamese network, and the objective decides the geometry.**
Two sentences pass through the *same* weights, each producing a pooled
vector :math:`u` and :math:`v`. What happens next depends on the task:

* **classification** (NLI): concatenate :math:`(u, v, |u-v|)` and pass
  it to a softmax layer. The element-wise difference is the term that
  matters -- it gives the classifier direct access to *where* the two
  vectors disagree, which neither :math:`u` nor :math:`v` alone
  supplies.
* **regression** (STS): score by :math:`\cos(u, v)` directly and train
  on mean squared error against the gold similarity.

The distinction is worth keeping: only the regression objective trains
the *cosine geometry* itself. The classification objective shapes the
space indirectly, which is why a model fine-tuned on NLI is usually
still evaluated by cosine at inference.

**What this buys.** Embeddings are computed once, indexed, and compared
by a dot product. The 65-hour pair-scoring problem becomes seconds,
and the comparison cost drops from :math:`O(n^2)` forward passes to
:math:`O(n)` plus vector arithmetic.

**Pooling is a real choice.** Mean, ``[CLS]`` and max are all offered
because the paper ablates all three; they give different geometries and
none is universally best.

References
----------
Reimers, N. & Gurevych, I. (2019) "Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks", *Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP)*, 3980-3990, doi:10.18653/v1/D19-1410,
arXiv:1908.10084. Sec. 2 (BERT's
sentence-pair regression setup with [SEP] and its cost; the absence of
independent sentence embeddings; the averaging and [CLS] workarounds
and the observation that they were unevaluated) and the architecture
figures giving the softmax objective over (u, v, |u-v|) and the
cosine-similarity objective.

Conneau, A., Kiela, D., Schwenk, H., Barrault, L. & Bordes, A. (2017)
"Supervised Learning of Universal Sentence Representations from
Natural Language Inference Data", *EMNLP 2017*, 670-680,
arXiv:1705.02364. InferSent: the siamese BiLSTM with max pooling
trained on NLI that this follows.

Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019) "BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding", *NAACL-HLT 2019*, 4171-4186, arXiv:1810.04805.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["pool", "cosine_similarity", "classification_features",
           "pair_cost", "rank_by_similarity", "sts_score"]

_EPS = 1e-12
_POOLING = ("mean", "cls", "max")


def pool(token_vectors, mode="mean", mask=None):
    r"""Reduce token vectors to one sentence vector.

    All three modes the paper ablates. ``mask`` excludes padding from
    the mean, which otherwise drags every embedding toward the pad
    vector.
    """
    if mode not in _POOLING:
        raise ValueError("sbert: pooling must be one of %s, got %r"
                         % (", ".join(_POOLING), mode))
    T = [[float(v) for v in r] for r in k.mat(token_vectors)]
    if not T:
        raise ValueError("sbert: no token vectors given")
    d = len(T[0])
    m = [True] * len(T) if mask is None else [bool(v) for v in mask]
    if len(m) != len(T):
        raise ValueError("sbert: %d mask entries for %d tokens"
                         % (len(m), len(T)))
    keep = [i for i in range(len(T)) if m[i]]
    if not keep:
        raise ValueError("sbert: the mask excludes every token")
    if mode == "cls":
        return list(T[keep[0]])
    if mode == "max":
        return [max(T[i][j] for i in keep) for j in range(d)]
    return [sum(T[i][j] for i in keep) / len(keep) for j in range(d)]


def cosine_similarity(u, v):
    r""":math:`\cos(u,v)`, the score the regression objective trains."""
    a = [float(x) for x in k.vec(u)]
    b = [float(x) for x in k.vec(v)]
    if len(a) != len(b):
        raise ValueError("sbert: vectors differ in length (%d, %d)"
                         % (len(a), len(b)))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= _EPS or nb <= _EPS:
        raise ValueError("sbert: cosine similarity is undefined for a "
                         "zero vector")
    return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)


def classification_features(u, v):
    r"""The softmax input: :math:`(u, v, |u-v|)`.

    The element-wise difference is what gives the classifier direct
    access to where the two sentences disagree.
    """
    a = [float(x) for x in k.vec(u)]
    b = [float(x) for x in k.vec(v)]
    if len(a) != len(b):
        raise ValueError("sbert: vectors differ in length (%d, %d)"
                         % (len(a), len(b)))
    diff = [abs(a[i] - b[i]) for i in range(len(a))]
    return {"features": a + b + diff, "u": a, "v": b,
            "abs_diff": diff, "dim": 3 * len(a),
            "note": "|u - v| is the term neither u nor v supplies"}


def pair_cost(n, mode="cross-encoder"):
    r"""Forward passes needed to compare :math:`n` sentences.

    A cross-encoder needs :math:`\binom{n}{2}`; a bi-encoder needs
    :math:`n` and then cheap vector arithmetic.
    """
    N = int(n)
    if N < 2:
        raise ValueError("sbert: need at least 2 sentences")
    if mode not in ("cross-encoder", "bi-encoder"):
        raise ValueError("sbert: mode must be cross-encoder or "
                         "bi-encoder, got %r" % (mode,))
    cross = N * (N - 1) // 2
    return {"forward_passes": cross if mode == "cross-encoder" else N,
            "cross_encoder": cross, "bi_encoder": N,
            "speedup": cross / float(N), "n": N,
            "note": "the bi-encoder also does O(n^2) dot products, "
                    "but those are arithmetic, not network passes"}


def rank_by_similarity(query, corpus_embeddings, top_k=5):
    r"""Nearest neighbours by cosine -- the operation the design
    enables."""
    E = [[float(v) for v in r] for r in k.mat(corpus_embeddings)]
    if not E:
        raise ValueError("sbert: the corpus is empty")
    scores = [(i, cosine_similarity(query, E[i]))
              for i in range(len(E))]
    scores.sort(key=lambda t: -t[1])
    return {"ranking": scores[:int(top_k)], "n_corpus": len(E),
            "forward_passes": 0,
            "note": "no network passes at query time -- the corpus "
                    "was embedded once"}


def sts_score(pairs, embed):
    r"""Cosine scores for sentence pairs, with each side embedded once.

    ``embed`` maps a sentence to a vector; it is called once per
    distinct sentence, which is the point.
    """
    cache, out, calls = {}, [], 0
    for a, b in pairs:
        for s in (a, b):
            if s not in cache:
                cache[s] = [float(v) for v in k.vec(embed(s))]
                calls += 1
        out.append(cosine_similarity(cache[a], cache[b]))
    return RichResult(payload={
        "estimate": out, "scores": out, "embed_calls": calls,
        "n_pairs": len(pairs),
        "cross_encoder_calls": len(pairs),
        "method": "siamese bi-encoder scored by cosine; Reimers & "
                  "Gurevych (2019)",
    })


def cheatsheet():
    return ("sbert: BERT scores a PAIR, so comparing n sentences needs "
            "C(n,2) forward passes -- 10k sentences is ~50M. A "
            "SIAMESE network embeds each sentence ONCE with shared "
            "weights, so it is n passes plus dot products. "
            "Classification objective: softmax over (u, v, |u-v|) -- "
            "the difference term is what locates the disagreement. "
            "Regression objective: cosine directly, and only that one "
            "trains the cosine geometry. Pooling (mean/CLS/max) is a "
            "real choice, all three ablated.")


# compact alias per ledger/NAMING.md
sentencebert = sts_score
