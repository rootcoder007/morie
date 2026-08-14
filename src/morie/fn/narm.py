# morie.fn -- function file (rootcoder007/morie)
r"""NARM: sequential behaviour and main purpose, together.

A session-based recommender that models only the click *sequence* is
fragile in a specific way the paper names: a user comparing shirts may
click a pair of suit trousers by accident or out of curiosity, and a
purely sequential model will happily recommend trousers next. An
experienced human shop assistant would notice that most of the
session's clicks concern short-sleeved shirts and weight those.

So NARM uses two encoders over the same GRU hidden states.

* The **global** encoder takes the last hidden state
  :math:`h_t^g` as a summary of the whole sequential behaviour.
* The **local** encoder computes attention weights over the previous
  hidden states and forms
  :math:`c_t^l = \sum_{j=1}^{t}\alpha_{tj}h_j^l`, capturing the
  session's main purpose by attending differently to more and less
  important items.

The two are concatenated, :math:`c_t = [c_t^g;\, c_t^l]` (eq. 9).
:math:`h_t^g` and :math:`h_t^l` have the *same values* and different
roles -- one is the summary, the other is the query that computes the
weights. They are complementary rather than redundant: a session can
be too short, or too aimless, for a purpose to be inferable at all,
and then the sequential part carries the prediction.

**The decoder is bilinear, and that is a parameter argument.** A
fully connected output layer costs :math:`|H| \cdot |N|` parameters
with :math:`|N|` the whole catalogue. Instead

.. math:: S_i = \mathrm{emb}_i^\top B\, c_t,

with :math:`B` of size :math:`|D| \times |H|`, cuts it to
:math:`|D|\cdot|H|` -- and the paper reports it also improves
accuracy, so it is not purely an economy. The anchor checks the
parameter counts and the scoring identity.

Training is cross-entropy over the softmax of those scores.

References
----------
Li, J., Ren, P., Chen, Z., Ren, Z., Lian, T. & Ma, J. (2017) "Neural
Attentive Session-based Recommendation", *Proceedings of the 2017 ACM
Conference on Information and Knowledge Management (CIKM '17)*,
1419-1428, doi:10.1145/3132847.3132926, arXiv:1711.04725. Sec. 1 (the
shirt-and-suit-trousers example; relying only on sequential behaviour
is dangerous when a user clicks the wrong item or is briefly
distracted; the two signals are complementary because a purpose
cannot always be inferred). Sec. 3.3-3.5 (the global and local
encoders, that h_t^g and h_t^l have the same values but different
roles, the attention-weighted c_t^l and the concatenation of eq. (9),
the bilinear decoder of eq. (10) reducing the parameter count from
|N||H| to |D||H| while improving performance, and the cross-entropy
loss of eq. (11)).

Hidasi, B., Karatzoglou, A., Baltrunas, L. & Tikk, D. (2016)
"Session-based Recommendations with Recurrent Neural Networks",
*ICLR 2016*, arXiv:1511.06939. The GRU baseline NARM extends;
implemented in :mod:`gru4r`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["attention_weights", "local_encoder", "session_repr",
           "bilinear_scores", "decoder_parameters", "softmax"]

_EPS = 1e-12


def softmax(z):
    v = [float(q) for q in k.vec(z)]
    m = max(v)
    e = [math.exp(q - m) for q in v]
    s = sum(e)
    return [q / s for q in e]


def attention_weights(h_t, H, A1, A2, v):
    r""":math:`\alpha_{tj} \propto v^\top\sigma(A_1h_t + A_2h_j)`.

    The query is the *last* hidden state, so the weights say which
    earlier clicks matter given where the session ended up.
    """
    ht = [float(q) for q in k.vec(h_t)]
    rows = [[float(q) for q in r] for r in k.mat(H)]
    sc = []
    for hj in rows:
        z = [1.0 / (1.0 + math.exp(-(
            sum(A1[o][i] * ht[i] for i in range(len(ht)))
            + sum(A2[o][i] * hj[i] for i in range(len(hj))))))
            for o in range(len(A1))]
        sc.append(sum(v[o] * z[o] for o in range(len(v))))
    return softmax(sc)


def local_encoder(H, alpha):
    r""":math:`c_t^l = \sum_j \alpha_{tj}h_j`."""
    rows = [[float(q) for q in r] for r in k.mat(H)]
    a = [float(q) for q in k.vec(alpha)]
    if len(a) != len(rows):
        raise ValueError("narm: %d weights for %d hidden states"
                         % (len(a), len(rows)))
    d = len(rows[0])
    return [sum(a[j] * rows[j][f] for j in range(len(rows)))
            for f in range(d)]


def session_repr(h_t_global, c_local):
    r"""Eq. (9): :math:`c_t = [c_t^g; c_t^l]`."""
    return [float(q) for q in k.vec(h_t_global)] + \
        [float(q) for q in k.vec(c_local)]


def bilinear_scores(embeddings, B, c_t):
    r"""Eq. (10): :math:`S_i = \mathrm{emb}_i^\top B c_t`."""
    E = [[float(q) for q in r] for r in k.mat(embeddings)]
    c = [float(q) for q in k.vec(c_t)]
    if len(B[0]) != len(c):
        raise ValueError("narm: B has %d columns for a session vector "
                         "of %d" % (len(B[0]), len(c)))
    Bc = [sum(B[d][h] * c[h] for h in range(len(c)))
          for d in range(len(B))]
    if len(E[0]) != len(Bc):
        raise ValueError("narm: embeddings are %d-dimensional but B "
                         "has %d rows" % (len(E[0]), len(Bc)))
    s = [sum(E[i][d] * Bc[d] for d in range(len(Bc)))
         for i in range(len(E))]
    return RichResult(payload={
        "estimate": s, "scores": s, "probabilities": softmax(s),
        "method": "bilinear decoder; Li et al. (2017) eq. (10)",
        "note": "|D||H| parameters instead of |N||H|, and the paper "
                "reports better accuracy too",
    })


def decoder_parameters(n_items, hidden, emb_dim):
    r"""Fully connected against bilinear."""
    N, H, D = int(n_items), int(hidden), int(emb_dim)
    if min(N, H, D) < 1:
        raise ValueError("narm: all three sizes must be at least 1")
    return {"fully_connected": N * H, "bilinear": D * H,
            "ratio": (N * H) / float(D * H),
            "note": "|D| is usually far smaller than |N|"}


def cheatsheet():
    return ("narm: a purely sequential session model recommends "
            "trousers because the shopper clicked a pair by accident. "
            "Two encoders over the SAME GRU states: the global one "
            "takes h_t as the whole-behaviour summary, the local one "
            "attends over previous states to capture the session's "
            "MAIN PURPOSE. h_t^g and h_t^l have identical values and "
            "different roles. Concatenate, then score with a BILINEAR "
            "decoder emb_i' B c_t -- |D||H| parameters instead of "
            "|N||H|, and more accurate.")


# compact alias per ledger/NAMING.md
neuralattentiverec = bilinear_scores
