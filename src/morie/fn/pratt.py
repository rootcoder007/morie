# morie.fn -- function file (rootcoder007/morie)
r"""Hierarchical attention for document classification.

A document has structure -- words make sentences, sentences make
documents -- and the model mirrors it: a word-level encoder with
attention produces a sentence vector, and a sentence-level encoder with
attention produces the document vector. Two levels of encoding, two
levels of attention.

**Why attention twice, rather than once.** Not all words matter equally
within a sentence, and not all sentences matter equally within a
document; those are different judgements and neither implies the
other. Flattening the document into one long sequence loses the
distinction, and averaging loses both.

**The attention mechanism, at each level.** Pass the annotation through
a one-layer MLP, measure its similarity to a learned **context
vector**, and normalise:

.. math:: u_{it} = \tanh(W_w h_{it} + b_w), \qquad
          \alpha_{it} = \frac{\exp(u_{it}^\top u_w)}
                             {\sum_t \exp(u_{it}^\top u_w)}, \qquad
          s_i = \sum_t \alpha_{it} h_{it},

and identically at the sentence level with :math:`u_s`. The context
vector is the part worth understanding: it is a fixed high-level query
-- *"what is the informative word?"* -- and it is **randomly
initialised and learned jointly**, not supplied. There is no external
query as in a question-answering setting; the model discovers what
"informative" means for the task.

**The weights are the interpretation.** Because :math:`\alpha` is a
normalised distribution over positions, it can be read directly as
which words and which sentences drove the classification, and the
anchor checks that the weights concentrate where the signal actually
is rather than assuming they do.

References
----------
Yang, Z., Yang, D., Dyer, C., He, X., Smola, A. & Hovy, E. (2016)
"Hierarchical Attention Networks for Document Classification",
*Proceedings of the 2016 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Language
Technologies (NAACL-HLT 2016)*, 1480-1489, doi:10.18653/v1/N16-1174.
Sec. 2 (the two-level structure mirroring the document's own; the
word-level attention of eqs. (5)-(7) with u_it = tanh(W_w h_it + b_w),
the softmax over the similarity of u_it with the word-level context
vector u_w, and the sentence vector as the weighted sum; the statement
that u_w acts as a fixed high-level query "what is the informative
word", as in memory networks, and that it is randomly initialised and
jointly learned during training; and the sentence-level attention of
eqs. (8)-(10) with the context vector u_s).

Bahdanau, D., Cho, K. & Bengio, Y. (2015) "Neural Machine Translation
by Jointly Learning to Align and Translate", *ICLR 2015*,
arXiv:1409.0473. The attention mechanism being adapted.

Sukhbaatar, S., Szlam, A., Weston, J. & Fergus, R. (2015) "End-To-End
Memory Networks", *NIPS 2015*, 2440-2448, arXiv:1503.08895. The
fixed-query reading of the context vector.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["attention", "sentence_vector", "document_vector",
           "classify", "attention_entropy"]

_EPS = 1e-12


def attention(H, W, b, u_context):
    r"""Eqs. (5)-(7): MLP, similarity to the context vector, softmax."""
    rows = [[float(v) for v in r] for r in k.mat(H)]
    if not rows:
        raise ValueError("pratt: nothing to attend over")
    uc = [float(v) for v in k.vec(u_context)]
    sc = []
    for h in rows:
        u = [math.tanh(b[o] + sum(W[o][j] * h[j]
                                  for j in range(len(h))))
             for o in range(len(W))]
        if len(u) != len(uc):
            raise ValueError("pratt: the context vector is %d-"
                             "dimensional but the projection is %d"
                             % (len(uc), len(u)))
        sc.append(sum(u[o] * uc[o] for o in range(len(u))))
    m = max(sc)
    e = [math.exp(v - m) for v in sc]
    z = sum(e)
    return [v / z for v in e]


def sentence_vector(H_words, W, b, u_w):
    r""":math:`s_i = \sum_t \alpha_{it}h_{it}`."""
    a = attention(H_words, W, b, u_w)
    rows = [[float(v) for v in r] for r in k.mat(H_words)]
    d = len(rows[0])
    return {"vector": [sum(a[t] * rows[t][f] for t in range(len(rows)))
                       for f in range(d)],
            "alpha": a}


def document_vector(H_sentences, W, b, u_s):
    r""":math:`v = \sum_i \alpha_i h_i`, with its own context vector."""
    a = attention(H_sentences, W, b, u_s)
    rows = [[float(v) for v in r] for r in k.mat(H_sentences)]
    d = len(rows[0])
    return {"vector": [sum(a[i] * rows[i][f] for i in range(len(rows)))
                       for f in range(d)],
            "alpha": a}


def classify(word_states, Ww, bw, u_w, Ws, bs, u_s, Wc, bc):
    r"""The full hierarchy: words to sentences to a document label."""
    S, wa = [], []
    for H in word_states:
        r = sentence_vector(H, Ww, bw, u_w)
        S.append(r["vector"])
        wa.append(r["alpha"])
    dv = document_vector(S, Ws, bs, u_s)
    z = [bc[o] + sum(Wc[o][j] * dv["vector"][j]
                     for j in range(len(dv["vector"])))
         for o in range(len(Wc))]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    tot = sum(e)
    return RichResult(payload={
        "estimate": [v / tot for v in e],
        "probabilities": [v / tot for v in e],
        "document_vector": dv["vector"],
        "sentence_attention": dv["alpha"],
        "word_attention": wa,
        "n_sentences": len(S),
        "method": "hierarchical attention network; Yang et al. (2016)",
        "note": "both context vectors are randomly initialised and "
                "learned -- the model discovers what 'informative' "
                "means",
    })


def attention_entropy(alpha):
    r"""How concentrated the weights are.

    Zero when all mass is on one position, :math:`\log n` when
    uniform -- which is what distinguishes an interpretable
    explanation from a diffuse one.
    """
    a = [float(v) for v in k.vec(alpha)]
    s = sum(a)
    if s <= _EPS:
        raise ValueError("pratt: the attention weights have no mass")
    a = [v / s for v in a]
    h = -sum(v * math.log(max(v, _EPS)) for v in a)
    return {"entropy": h, "max_entropy": math.log(len(a)),
            "concentration": 1.0 - h / math.log(len(a))
            if len(a) > 1 else 1.0}


def cheatsheet():
    return ("pratt: mirror the document's own structure -- words to "
            "sentences to document -- with attention at BOTH levels, "
            "because which word matters within a sentence and which "
            "sentence matters within a document are different "
            "judgements. At each level: u = tanh(W h + b), then "
            "softmax of u'u_context, then a weighted sum. The CONTEXT "
            "VECTOR is a learned fixed query ('what is the informative "
            "word'), randomly initialised, not supplied. The alphas "
            "are a distribution over positions, so they read directly "
            "as the explanation.")


# compact alias per ledger/NAMING.md
hierarchicalattention = classify
