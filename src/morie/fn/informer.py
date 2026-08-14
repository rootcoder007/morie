# morie.fn -- function file (rootcoder007/morie)
r"""Informer: ProbSparse attention for long-sequence forecasting.

Self-attention costs :math:`O(L^2)` in both time and memory, which is
what stops a Transformer from reading a long history. Informer's
observation is that most of that computation is **wasted**, and it
says precisely why.

**A query whose attention is uniform contributes nothing.** Write the
attention of query :math:`i` over the keys as
:math:`p(k_j \mid q_i)`. If that distribution is close to uniform,
:math:`q(k_j\mid q_i) = 1/L_K`, then the output for that query is
essentially a plain average of the values -- redundant with the
residual connection already carrying the input forward. Only queries
whose attention is *far* from uniform do any work.

**So measure the distance from uniform.** The Kullback-Leibler
divergence between the two, with the constant dropped, gives the
sparsity measurement

.. math:: M(q_i, K) = \ln \sum_{j=1}^{L_K}
          e^{q_i k_j^{\top}/\sqrt d}
          - \frac{1}{L_K}\sum_{j=1}^{L_K}
          \frac{q_i k_j^{\top}}{\sqrt d},

a log-sum-exp minus a mean. The paper drops the additive constant
:math:`\ln L_K` from the KL divergence, so :math:`M` is not zero at
uniform attention -- it attains its **minimum** there, and that
minimum is exactly :math:`\ln L_K`. By Jensen's inequality
:math:`\mathrm{logsumexp}(z) - \bar z \ge \ln L_K` for every
:math:`z`, with equality if and only if all the logits are equal. So
:math:`M - \ln L_K \ge 0` is the KL itself, zero exactly at uniform,
and it is that quantity the anchor pins down as an identity.

**ProbSparse keeps only the top-u queries.** With
:math:`u = c \ln L_Q` for a sampling factor :math:`c`, each query-key
lookup needs :math:`O(\ln L_Q)` dot products and the layer memory
stays :math:`O(L_K \ln L_Q)`. Under multi-head attention each head
selects its own sparse query-key pairs, so what one head drops another
may keep -- which is why the information loss is not severe.

**The measurement itself would cost what it saves.** Computing
:math:`M(q_i, K)` for every query needs every dot product, i.e.
:math:`O(L_Q L_K)` -- exactly the cost being avoided -- and the
log-sum-exp is numerically delicate. Lemma 1 supplies the way out: a
max-mean approximation

.. math:: \bar M(q_i, K) = \max_j \frac{q_i k_j^{\top}}{\sqrt d}
          - \frac{1}{L_K}\sum_j \frac{q_i k_j^{\top}}{\sqrt d}

bounds :math:`M` and, evaluated on a sample of keys rather than all of
them, ranks the queries well enough to select the same top set. Both
are implemented: ``measure="exact"`` and ``measure="maxmean"``, with
the anchor measuring how often their top-:math:`u` sets agree instead
of assuming they do.

**What is given up.** Queries outside the top-:math:`u` are not
computed; they take the mean of the values, which is what their
near-uniform attention would have produced anyway. That substitution
is the approximation, and setting :math:`u = L_Q` recovers full
attention **exactly** -- another identity the anchor checks.

References
----------
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H. & Zhang,
W. (2021) "Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting", *Proceedings of the AAAI Conference on
Artificial Intelligence* 35(12), 11106-11115, arXiv:2012.07436. The
query sparsity measurement derived from the KL divergence against the
uniform distribution, eq. (3) (ProbSparse self-attention over the
top-u queries), the choice u = c ln L_Q giving O(L ln L) time and
O(L_K ln L_Q) memory, the multi-head argument for why information
loss is limited, and Lemma 1's max-mean approximation.

Vaswani, A. et al. (2017) "Attention is all you need", *Advances in
Neural Information Processing Systems* 30, arXiv:1706.03762. The
scaled dot-product attention of eq. (1).
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sparsity_measure", "kl_from_uniform", "select_queries",
           "probsparse_attention", "full_attention", "complexity"]

_EPS = 1e-12
_MEASURES = ("exact", "maxmean")


def _logits(q, K, scale):
    return [scale * sum(q[a] * K[j][a] for a in range(len(q)))
            for j in range(len(K))]


def sparsity_measure(q, K, measure="exact", scale=None):
    r"""How far this query's attention is from uniform.

    ``"exact"`` is the paper's M: log-sum-exp minus the mean. It is
    minimised at :math:`\ln L_K`, attained exactly when the attention
    is uniform, so ``M - ln L_K`` is the KL divergence and is zero
    there. ``"maxmean"`` is Lemma 1's bound, which costs no
    log-sum-exp and is what makes the selection affordable.
    """
    if measure not in _MEASURES:
        raise ValueError("informer: measure must be exact or maxmean, "
                         "got %r" % (measure,))
    Km = [[float(v) for v in r] for r in k.mat(K)]
    qv = [float(v) for v in k.vec(q)]
    if not Km:
        raise ValueError("informer: the key set is empty")
    d = len(qv)
    if len(Km[0]) != d:
        raise ValueError("informer: query has %d dimensions but keys "
                         "have %d" % (d, len(Km[0])))
    sc = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    z = _logits(qv, Km, sc)
    mean = sum(z) / len(z)
    if measure == "maxmean":
        return max(z) - mean
    return k.logsumexp(z) - mean


def kl_from_uniform(q, K, scale=None):
    r"""The KL divergence itself: :math:`M(q,K) - \ln L_K`.

    Zero exactly when the attention over the keys is uniform, and
    positive otherwise -- the quantity the sparsity measurement is a
    shifted version of.
    """
    Km = k.mat(K)
    return sparsity_measure(q, K, measure="exact",
                            scale=scale) - math.log(len(Km))


def select_queries(Q, K, factor=5, measure="maxmean", n_sample=None,
                   seed=0):
    r"""The top-:math:`u` queries, :math:`u = c \ln L_Q`.

    ``n_sample`` evaluates the measure against a random subset of keys
    rather than all of them -- the practical form, since scoring
    against every key would cost the :math:`O(L_Q L_K)` the method
    exists to avoid.
    """
    Qm = [[float(v) for v in r] for r in k.mat(Q)]
    Km = [[float(v) for v in r] for r in k.mat(K)]
    LQ, LK = len(Qm), len(Km)
    if LQ == 0 or LK == 0:
        raise ValueError("informer: queries and keys must be "
                         "non-empty")
    u = max(1, min(LQ, int(float(factor) * math.log(max(LQ, 2)))))
    if n_sample is not None and int(n_sample) < LK:
        rng = np.random.default_rng(seed)
        idx = sorted(range(LK),
                     key=lambda _i: float(rng.uniform()))[:int(n_sample)]
        Ks = [Km[j] for j in idx]
    else:
        Ks = Km
    scores = [sparsity_measure(Qm[i], Ks, measure=measure)
              for i in range(LQ)]
    order = sorted(range(LQ), key=lambda i: -scores[i])[:u]
    return {"top": sorted(order), "u": u, "scores": scores,
            "L_Q": LQ, "L_K": LK,
            "n_sample": int(n_sample) if n_sample else LK,
            "measure": measure}


def full_attention(Q, K, V, scale=None):
    r"""Ordinary scaled dot-product attention, for comparison."""
    Qm = [[float(v) for v in r] for r in k.mat(Q)]
    Km = [[float(v) for v in r] for r in k.mat(K)]
    Vm = [[float(v) for v in r] for r in k.mat(V)]
    if len(Km) != len(Vm):
        raise ValueError("informer: keys and values must match in "
                         "length (%d, %d)" % (len(Km), len(Vm)))
    d = len(Qm[0])
    sc = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    out = []
    for q in Qm:
        w = k.softmax(_logits(q, Km, sc))
        out.append([sum(w[j] * Vm[j][a] for j in range(len(Vm)))
                    for a in range(len(Vm[0]))])
    return out


def probsparse_attention(Q, K, V, factor=5, measure="maxmean",
                         n_sample=None, seed=0, scale=None):
    r"""Eq. (3): attention computed only for the dominant queries.

    Queries outside the top-:math:`u` are given the mean of the
    values, which is what their near-uniform attention would have
    produced. With :math:`u = L_Q` this reduces to full attention
    exactly.
    """
    Qm = [[float(v) for v in r] for r in k.mat(Q)]
    Km = [[float(v) for v in r] for r in k.mat(K)]
    Vm = [[float(v) for v in r] for r in k.mat(V)]
    if len(Km) != len(Vm):
        raise ValueError("informer: keys and values must match in "
                         "length (%d, %d)" % (len(Km), len(Vm)))
    sel = select_queries(Qm, Km, factor=factor, measure=measure,
                         n_sample=n_sample, seed=seed)
    d = len(Qm[0])
    sc = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    dv = len(Vm[0])
    vbar = [sum(Vm[j][a] for j in range(len(Vm))) / len(Vm)
            for a in range(dv)]
    out = [list(vbar) for _ in range(len(Qm))]
    for i in sel["top"]:
        w = k.softmax(_logits(Qm[i], Km, sc))
        out[i] = [sum(w[j] * Vm[j][a] for j in range(len(Vm)))
                  for a in range(dv)]
    return RichResult(payload={
        "estimate": out, "output": out, "selected": sel["top"],
        "u": sel["u"], "L_Q": sel["L_Q"], "L_K": sel["L_K"],
        "measure": measure,
        "complexity": complexity(sel["L_Q"], sel["L_K"], factor),
        "method": "ProbSparse self-attention, Zhou et al. (2021) "
                  "eq. (3)",
        "note": "unselected queries take the mean of V, which is what "
                "their near-uniform attention would give",
    })


def complexity(L_Q, L_K, factor=5):
    r"""Dot-product counts: full attention against ProbSparse."""
    lq, lk = int(L_Q), int(L_K)
    u = max(1, min(lq, int(float(factor) * math.log(max(lq, 2)))))
    return {"full": lq * lk, "probsparse": u * lk, "u": u,
            "ratio": (lq * lk) / max(u * lk, 1),
            "memory_full": lq * lk,
            "memory_probsparse": lk * max(1, int(math.log(max(lq, 2))))}


def cheatsheet():
    return ("informer: ProbSparse. A query whose attention is UNIFORM "
            "just averages V and is redundant with the residual. "
            "M(q,K) = logsumexp(z) - mean(z) measures the distance "
            "from uniform; it is MINIMISED at ln L_K, attained "
            "exactly when the logits are equal, so M - ln L_K is the "
            "KL and that is what is zero there. "
            "Keep only the top u = c ln L_Q queries: O(L ln L) time, "
            "O(L_K ln L_Q) memory. Computing M exactly would cost the "
            "O(L^2) being saved, so Lemma 1's max-mean bound on "
            "sampled keys is used instead. u = L_Q recovers full "
            "attention exactly.")


# compact alias per ledger/NAMING.md -- infmer and informer are the
# same ledger entry duplicated; both names resolve here
informerattention = probsparse_attention
