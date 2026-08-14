# morie.fn -- function file (rootcoder007/morie)
r"""SASRec: self-attentive sequential recommendation.

Two traditions, each right about a different regime. **Markov chains**
predict the next action from the last one or few: parsimonious, and
best where data are extremely sparse and every parameter must earn its
place. **RNNs** allow longer-term semantics to be uncovered: better
where data are dense enough to afford the complexity. The choice is
usually made once, for the whole dataset.

**Self-attention refuses the choice.** At each step the model asks
which items in the history are *relevant* and predicts from those --
so it can reach far back like an RNN while basing a given prediction on
few actions like a Markov chain. Crucially this is not a fixed
compromise: the attention weights are computed per sequence, so the
model behaves parsimoniously where the data are sparse and uses long
context where they are not. That adaptivity is the claim, and the
anchor tests it by feeding one sequence whose signal is the last item
and another whose signal is far back, and requiring the attention mass
to move.

**Causal masking is the correctness condition.** Position :math:`t`
may attend only to :math:`\le t`; attending to :math:`t+1` leaks the
answer and produces a model that scores beautifully offline and fails
in production. The mask is not an optimisation.

**Efficiency.** All positions are attended in parallel rather than
sequentially, which is why the paper reports an order of magnitude
speed-up over comparable CNN/RNN models -- the computation is
:math:`O(n^2 d)` but fully parallel, against the RNN's :math:`O(nd^2)`
that cannot be.

References
----------
Kang, W.-C. & McAuley, J. (2018) "Self-Attentive Sequential
Recommendation", *Proceedings of the 2018 IEEE International
Conference on Data Mining (ICDM 2018)*, 197-206,
doi:10.1109/ICDM.2018.00035, arXiv:1808.09781. The abstract: Markov
chains assume the next action is predictable from the last few, while
RNNs allow longer-term semantics; MC-based methods perform best in
extremely sparse datasets where parsimony is critical, RNNs in denser
datasets where complexity is affordable; SASRec balances these by
capturing long-term semantics like an RNN while making predictions
from relatively few actions like an MC, identifying at each step which
items are relevant; outperforming MC/CNN/RNN baselines on both sparse
and dense datasets; being an order of magnitude more efficient; and
attention-weight visualisations showing adaptive handling of datasets
of various density.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, L. & Polosukhin, I. (2017) "Attention Is All You
Need", *NIPS 2017*, 5998-6008, arXiv:1706.03762.

Hidasi, B., Karatzoglou, A., Baltrunas, L. & Tikk, D. (2016)
"Session-based Recommendations with Recurrent Neural Networks",
*ICLR 2016*, arXiv:1511.06939. The RNN baseline; implemented in
:mod:`gru4r`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["causal_mask", "self_attention", "attention_span",
           "predict_next", "complexity"]

_EPS = 1e-12


def causal_mask(n):
    r"""Position :math:`t` may attend only to :math:`\le t`.

    Attending forward leaks the target -- a model that scores well
    offline and cannot work in production.
    """
    m = int(n)
    if m < 1:
        raise ValueError("sasRec: the sequence must be non-empty")
    return [[1.0 if j <= i else 0.0 for j in range(m)]
            for i in range(m)]


def self_attention(E, WQ, WK, WV, mask=None):
    r"""Masked scaled dot-product attention over the item embeddings."""
    X = [[float(v) for v in r] for r in k.mat(E)]
    n, d = len(X), len(X[0])
    M = causal_mask(n) if mask is None else mask

    def proj(W, x):
        return [sum(W[o][j] * x[j] for j in range(len(x)))
                for o in range(len(W))]

    dk = len(WQ)
    out, weights = [], []
    for i in range(n):
        q = proj(WQ, X[i])
        sc = []
        for j in range(n):
            if M[i][j] == 0.0:
                sc.append(-1e30)
                continue
            kk = proj(WK, X[j])
            sc.append(sum(q[a] * kk[a] for a in range(dk))
                      / math.sqrt(dk))
        mx = max(sc)
        e = [math.exp(v - mx) if v > -1e29 else 0.0 for v in sc]
        z = sum(e) or 1.0
        w = [v / z for v in e]
        weights.append(w)
        vs = [proj(WV, X[j]) for j in range(n)]
        out.append([sum(w[j] * vs[j][a] for j in range(n))
                    for a in range(len(vs[0]))])
    return {"output": out, "weights": weights,
            "note": "the mask is a correctness condition, not an "
                    "optimisation"}


def attention_span(weights, position=None):
    r"""How far back the prediction actually looks.

    The weighted mean distance -- small means the model is behaving
    like a Markov chain, large like an RNN, and it is the DATA that
    decides.
    """
    W = [[float(v) for v in r] for r in k.mat(weights)]
    i = len(W) - 1 if position is None else int(position)
    row = W[i]
    tot = sum(row[:i + 1])
    if tot <= _EPS:
        raise ValueError("sasRec: the attention row has no mass")
    span = sum((i - j) * row[j] for j in range(i + 1)) / tot
    return {"mean_lookback": span,
            "mass_on_last": row[i] / tot,
            "effective_order": span + 1.0,
            "note": "a short span IS Markov behaviour; a long one is "
                    "RNN behaviour, chosen per sequence"}


def predict_next(state, item_embeddings, top_k=5, exclude=()):
    r"""Score every item by its inner product with the final state."""
    s = [float(v) for v in k.vec(state)]
    E = [[float(v) for v in r] for r in k.mat(item_embeddings)]
    ex = set(int(v) for v in exclude)
    sc = [(i, sum(s[a] * E[i][a] for a in range(len(s))))
          for i in range(len(E)) if i not in ex]
    sc.sort(key=lambda t: -t[1])
    return RichResult(payload={
        "estimate": sc[:int(top_k)], "ranking": sc[:int(top_k)],
        "n_scored": len(sc),
        "method": "self-attentive sequential recommendation; Kang & "
                  "McAuley (2018)",
    })


def complexity(n, d):
    r"""Self-attention against an RNN, and why parallelism matters.

    Attention is :math:`O(n^2 d)` but fully parallel over positions;
    an RNN is :math:`O(nd^2)` and inherently sequential.
    """
    nn, dd = int(n), int(d)
    if nn < 1 or dd < 1:
        raise ValueError("sasRec: n and d must be positive")
    return {"attention_ops": nn * nn * dd, "rnn_ops": nn * dd * dd,
            "attention_sequential_steps": 1, "rnn_sequential_steps":
            nn,
            "note": "the parallelism, not the operation count, is "
                    "where the order-of-magnitude speed-up comes "
                    "from"}


def cheatsheet():
    return ("sasRec: Markov chains win where data are SPARSE (parsimony "
            "is critical), RNNs where they are DENSE (complexity is "
            "affordable) -- and the choice is normally made once for a "
            "whole dataset. Self-attention picks per sequence: it can "
            "reach far back like an RNN while predicting from FEW "
            "actions like an MC, and the attention weights show it "
            "adapting to density. Causal masking is a CORRECTNESS "
            "condition -- attending forward leaks the target. O(n^2 d) "
            "but fully parallel against an RNN's inherently sequential "
            "O(n d^2).")


# compact alias per ledger/NAMING.md
selfattentivesequential = self_attention

# public names resolved by fn/_lazy_map.json
sasrec = self_attention
