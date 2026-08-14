# morie.fn -- function file (rootcoder007/morie)
r"""Crossformer: cross-time and cross-dimension dependency, separately.

Every earlier Transformer for multivariate forecasting embeds *all
dimensions at one time step* into a single vector,
:math:`x_t \in \mathbb R^{D} \to h_t \in \mathbb R^{d_{model}}`, and
then attends over the :math:`T` resulting vectors. That captures
cross-time dependency and throws cross-dimension dependency away at
the very first step: once the dimensions are summed into one vector,
no later layer can ask which dimension a signal came from.

**Dimension-Segment-Wise embedding keeps them apart.** A single value
at a single step carries almost nothing -- unlike a word, which is why
the NLP recipe transfers badly. Nearby values in one dimension *do*
form a pattern. So DSW splits each dimension into segments of length
:math:`L_{seg}` and embeds each segment on its own (eq. 1-2):

.. math:: x^{(s)}_{i,d} = \{x_{t,d} : (i-1)L_{seg} < t \le i L_{seg}\},
          \qquad
          h_{i,d} = E x^{(s)}_{i,d} + E^{(pos)}_{i,d},

giving a **2D array** :math:`H = \{h_{i,d}\}` whose two axes are time
and dimension, each :math:`h_{i,d}` a univariate segment. The
motivation is empirical: attention maps from a plain Transformer on
this data are visibly blocky, close time points getting similar
weights, so the natural unit is a segment rather than a point.

**Two-Stage Attention, because the two axes are not interchangeable.**
Flattening the array and running ordinary self-attention would (a)
treat time and dimension as the same kind of axis, which they are not,
and (b) cost :math:`O(D^2 L^2)`, unaffordable for large :math:`D`. So
TSA does them in sequence.

*Cross-time* (eq. 3) applies multi-head self-attention within each
dimension, with **all dimensions sharing one MSA layer**:

.. math:: \hat Z^{time}_{:,d}
          = \mathrm{LayerNorm}\big(Z_{:,d}
            + \mathrm{MSA}(Z_{:,d}, Z_{:,d}, Z_{:,d})\big),

at cost :math:`O(D L^2)`. Sharing the weights is what makes the layer
equivariant to relabelling the dimensions -- permute the input
dimensions and the output permutes identically, which the anchor
checks exactly rather than by eye.

*Cross-dimension* uses a **router**. Direct attention across
dimensions costs :math:`O(D^2)` per time step. Instead a small
learnable array :math:`B_{i,:}` of :math:`c \ll D` vectors first
gathers from all dimensions, then broadcasts back:

.. math:: B^{gather}_{i,:}
            &= \mathrm{MSA}_1(B_{i,:}, Z^{time}_{i,:}, Z^{time}_{i,:}), \\
          Z^{dim}_{i,:}
            &= \mathrm{MSA}_2(Z^{time}_{i,:}, B^{gather}_{i,:},
               B^{gather}_{i,:}),

which is :math:`O(cD)` -- linear in :math:`D`. Information still
reaches every dimension from every other, but through a bottleneck of
size :math:`c` rather than an all-pairs matrix. That is the trade the
anchor measures: with :math:`c \ge D` the router reproduces full
attention closely, and its operation count grows linearly in
:math:`D` while full attention grows quadratically.

**Hierarchical encoder-decoder.** Each encoder layer merges adjacent
segments from the layer below, so upper layers see a coarser scale.
The decoder produces a prediction at every scale and adds them, so a
forecast is the sum of contributions from several resolutions rather
than a single one.

References
----------
Zhang, Y. & Yan, J. (2023) "Crossformer: Transformer Utilizing
Cross-Dimension Dependency for Multivariate Time Series Forecasting",
*International Conference on Learning Representations (ICLR 2023)*.
ICLR assigns no DOI. Sec. 3.1 and eq. (1)-(2) (Dimension-Segment-Wise
embedding, with Fig. 1 showing the segmented attention maps that
motivate it), Sec. 3.2 and eq. (3) (the Two-Stage Attention layer,
the O(DL^2) cross-time stage with a shared MSA, and the router that
reduces the cross-dimension stage from O(D^2) to O(cD)), Sec. 3.3
(the hierarchical encoder-decoder). Code at
github.com/Thinklab-SJTU/Crossformer.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, L. & Polosukhin, I. (2017) "Attention is all
you need", *Advances in Neural Information Processing Systems* 30,
arXiv:1706.03762. The multi-head self-attention and layer
normalisation used unchanged.

Dosovitskiy, A. et al. (2021) "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale", *International
Conference on Learning Representations*, arXiv:2010.11929. The
patching idea DSW adapts, and the flattening approach Sec. 3.2
explicitly declines to follow.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dsw_embed", "attention", "cross_time_stage",
           "cross_dimension_stage", "two_stage_attention",
           "segment_merge", "complexity"]

_EPS = 1e-12


def dsw_embed(X, seg_len, E=None, pos=None):
    r"""Eq. (1)-(2): segment each dimension, then embed each segment.

    ``X`` is :math:`T \times D`. Returns a 2D array of shape
    :math:`(T/L_{seg}) \times D \times d_{model}`. The point is that
    each output vector represents **one dimension's** segment, so the
    dimension identity survives the embedding -- which is exactly what
    the point-wise embedding of earlier models destroys.
    """
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    T = len(Xm)
    if T == 0:
        raise ValueError("crsfmr: the input series is empty")
    D = len(Xm[0])
    L = int(seg_len)
    if L < 1:
        raise ValueError("crsfmr: seg_len must be at least 1, got %d"
                         % L)
    if T % L != 0:
        raise ValueError("crsfmr: T = %d is not divisible by seg_len "
                         "= %d; pad the series first (the paper pads "
                         "to a proper length)" % (T, L))
    n_seg = T // L
    if E is None:
        # identity-ish default: the segment itself, so d_model = L
        Em = [[1.0 if a == b else 0.0 for b in range(L)]
              for a in range(L)]
    else:
        Em = [[float(v) for v in r] for r in k.mat(E)]
        if len(Em[0]) != L:
            raise ValueError("crsfmr: E must have seg_len = %d "
                             "columns, got %d" % (L, len(Em[0])))
    dm = len(Em)
    H = []
    for i in range(n_seg):
        row = []
        for d in range(D):
            seg = [Xm[i * L + q][d] for q in range(L)]
            vec = [sum(Em[a][q] * seg[q] for q in range(L))
                   for a in range(dm)]
            if pos is not None:
                p = k.mat(pos)
                vec = [vec[a] + float(p[i][d][a]) if isinstance(
                    p[i][d], (list, tuple)) else vec[a]
                    for a in range(dm)]
            row.append(vec)
        H.append(row)
    return {"H": H, "n_seg": n_seg, "D": D, "d_model": dm,
            "seg_len": L,
            "shape": (n_seg, D, dm),
            "note": "each vector is ONE dimension's segment; the "
                    "dimension axis survives embedding"}


def attention(Q, K_, V):
    r"""Scaled dot-product attention, rows of the weight matrix
    summing to 1."""
    Qm = [[float(v) for v in r] for r in k.mat(Q)]
    Km = [[float(v) for v in r] for r in k.mat(K_)]
    Vm = [[float(v) for v in r] for r in k.mat(V)]
    if len(Km) != len(Vm):
        raise ValueError("crsfmr: keys and values must have the same "
                         "length (%d, %d)" % (len(Km), len(Vm)))
    dk = len(Qm[0])
    if len(Km[0]) != dk:
        raise ValueError("crsfmr: queries and keys must share a "
                         "dimension (%d, %d)" % (dk, len(Km[0])))
    scale = 1.0 / math.sqrt(dk)
    out, W = [], []
    for q in Qm:
        logits = [scale * sum(q[a] * kk[a] for a in range(dk))
                  for kk in Km]
        w = k.softmax(logits)
        W.append(w)
        out.append([sum(w[j] * Vm[j][a] for j in range(len(Vm)))
                    for a in range(len(Vm[0]))])
    return {"out": out, "weights": W}


def cross_time_stage(Z):
    r"""Eq. (3): attention within each dimension, weights shared.

    ``Z`` is :math:`L \times D \times d_{model}`. All dimensions pass
    through the same attention, which is what makes the stage
    equivariant to permuting them.
    """
    L = len(Z)
    if L == 0:
        raise ValueError("crsfmr: the input array is empty")
    D = len(Z[0])
    out = [[None] * D for _ in range(L)]
    for d in range(D):
        seq = [Z[i][d] for i in range(L)]
        a = attention(seq, seq, seq)["out"]
        for i in range(L):
            # residual, as in eq. (3), before the norm
            out[i][d] = [seq[i][q] + a[i][q]
                         for q in range(len(seq[i]))]
    return out


def cross_dimension_stage(Z, router=None, n_router=None):
    r"""The router: gather to :math:`c` vectors, then broadcast back.

    ``n_router`` is :math:`c`. With ``router=None`` the routers are
    initialised as the first :math:`c` dimensions' vectors, which is a
    deterministic starting point rather than a random one so the
    behaviour is reproducible.

    Cost is :math:`O(cD)` per time step against :math:`O(D^2)` for
    direct all-pairs attention.
    """
    L = len(Z)
    if L == 0:
        raise ValueError("crsfmr: the input array is empty")
    D = len(Z[0])
    c = int(n_router) if n_router is not None else max(1, min(D, 3))
    if c < 1:
        raise ValueError("crsfmr: n_router must be at least 1")
    out = []
    for i in range(L):
        Zi = Z[i]
        B = ([[float(v) for v in r] for r in k.mat(router)]
             if router is not None
             else [list(Zi[d % D]) for d in range(c)])
        if len(B) != c:
            raise ValueError("crsfmr: the router array has %d rows "
                             "but n_router is %d" % (len(B), c))
        gathered = attention(B, Zi, Zi)["out"]
        back = attention(Zi, gathered, gathered)["out"]
        out.append([[Zi[d][q] + back[d][q]
                     for q in range(len(Zi[d]))] for d in range(D)])
    return out


def two_stage_attention(Z, n_router=None, router=None):
    """A full TSA layer: cross-time, then cross-dimension."""
    zt = cross_time_stage(Z)
    zd = cross_dimension_stage(zt, router=router, n_router=n_router)
    D = len(Z[0])
    c = int(n_router) if n_router is not None else max(1, min(D, 3))
    return RichResult(payload={
        "estimate": zd, "output": zd, "cross_time": zt,
        "L": len(Z), "D": D, "n_router": c,
        "complexity": complexity(len(Z), D, c),
        "method": "Two-Stage Attention, Zhang & Yan (2023) Sec. 3.2",
    })


def segment_merge(Z, factor=2):
    r"""Merge adjacent segments -- one step up the hierarchy.

    The encoder's upper layer sees half as many segments, each
    covering twice the span, which is how the model reaches a coarser
    scale (Sec. 3.3).
    """
    f = int(factor)
    if f < 2:
        raise ValueError("crsfmr: the merge factor must be at least 2")
    L = len(Z)
    if L % f != 0:
        raise ValueError("crsfmr: %d segments do not divide by a "
                         "merge factor of %d" % (L, f))
    D = len(Z[0])
    out = []
    for i in range(L // f):
        row = []
        for d in range(D):
            acc = [0.0] * len(Z[0][0])
            for q in range(f):
                v = Z[i * f + q][d]
                for a in range(len(acc)):
                    acc[a] += v[a] / f
            row.append(acc)
        out.append(row)
    return out


def complexity(L, D, c):
    r"""Operation counts for the two stages and for the alternatives.

    Cross-time is :math:`O(DL^2)`. Cross-dimension is :math:`O(cDL)`
    through the router against :math:`O(D^2L)` all-pairs, and
    flattening the whole array into one sequence would be
    :math:`O(D^2L^2)`.
    """
    Lv, Dv, cv = int(L), int(D), int(c)
    return {"cross_time": Dv * Lv * Lv,
            "cross_dimension_router": cv * Dv * Lv,
            "cross_dimension_full": Dv * Dv * Lv,
            "flattened_2d": Dv * Dv * Lv * Lv,
            "router_saving": (Dv * Dv * Lv)
            / max(cv * Dv * Lv, 1)}


def cheatsheet():
    return ("crsfmr: Crossformer. Earlier models embed ALL dimensions "
            "at one time step into one vector, destroying "
            "cross-dimension information at step one. DSW instead "
            "segments EACH dimension and embeds the segments, giving "
            "a 2D array (time x dimension). TSA then does cross-time "
            "attention per dimension with SHARED weights (O(D L^2)), "
            "then cross-dimension through a ROUTER of c << D vectors "
            "-- gather then broadcast -- which is O(cD) rather than "
            "O(D^2). Hierarchy: upper layers merge adjacent segments.")


# compact alias per ledger/NAMING.md
crossformer = two_stage_attention
