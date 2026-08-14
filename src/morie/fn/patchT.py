# morie.fn -- function file (rootcoder007/morie)
r"""PatchTST: subseries patches and channel independence.

A short paper by Zeng et al. showed that a plain linear model beat
every Transformer variant then available on standard forecasting
benchmarks, which put the whole enterprise in doubt. PatchTST's answer
is that the Transformers were being fed the wrong tokens, and it
changes two things.

**Patching: a single time step is not a word.** Point-wise tokens are
the default, but one value at one instant carries almost no meaning --
unlike a word in a sentence, which is what the architecture was
designed for. A *subseries* does carry meaning. So the series is cut
into patches of length :math:`P` with stride :math:`S`, and each patch
becomes one token.

Three consequences, and they are separable:

* local semantic information survives into the embedding, because the
  token *is* a local pattern;
* the attention map shrinks **quadratically** -- :math:`N \approx
  (L-P)/S + 1` tokens instead of :math:`L`, so cost falls by roughly
  :math:`S^2` for the same look-back;
* the same compute therefore buys a **longer history**, which is where
  the accuracy comes from.

**Channel independence: each series is its own sequence.** A
multivariate series is a multi-channel signal, and a token can be
built from one channel or from all of them. Channel-*mixing* projects
the vector of all features at a step into one embedding, blending them
before attention. Channel-*independence* gives each channel its own
univariate token stream, with the embedding and the Transformer
weights **shared** across channels.

The consequence is exact and is what the anchor tests: shared weights
plus per-channel tokens means the model is **equivariant** to
relabelling the channels -- permute the inputs and the outputs permute
identically. Channel mixing is not; it destroys the channel identity
at the first projection, exactly as it does in the point-wise
embedding case. That property had been shown to work for CNNs and
linear models but had not been tried in a Transformer.

**Why this matters against the linear baseline.** The linear model
that embarrassed the Transformers was itself channel-independent.
PatchTST keeps that inductive bias and adds attention on top of
tokens that mean something, rather than discarding the bias and hoping
attention rediscovers it.

References
----------
Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. (2023) "A Time
Series is Worth 64 Words: Long-term Forecasting with Transformers",
*International Conference on Learning Representations (ICLR 2023)*,
arXiv:2211.14730. The abstract's two components (subseries-level
patches as input tokens, and channel independence with shared
embedding and Transformer weights), the three-fold benefit of patching
(local semantics retained, attention maps quadratically reduced,
longer history attendable), and Sec. 1's contrast between
channel-mixing and channel-independent token designs.

Zeng, A., Chen, M., Zhang, L. & Xu, Q. (2023) "Are Transformers
Effective for Time Series Forecasting?", *Proceedings of the AAAI
Conference on Artificial Intelligence* 37(9), 11121-11128,
arXiv:2205.13504. The linear baseline that outperformed prior
Transformer variants and motivated this design.

Vaswani, A. et al. (2017) "Attention is all you need", *Advances in
Neural Information Processing Systems* 30, arXiv:1706.03762.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["patchify", "channel_independent_tokens",
           "channel_mixed_tokens", "instance_norm", "attention_cost",
           "patchtst_encode"]

_EPS = 1e-12


def patchify(x, patch_len, stride=None):
    r"""Cut one univariate series into patches of length :math:`P`.

    With ``stride`` equal to ``patch_len`` the patches are disjoint;
    a smaller stride overlaps them. The number of patches is
    :math:`N = \lfloor (L-P)/S \rfloor + 1`, which is what makes the
    attention map shrink.
    """
    v = [float(q) for q in k.vec(x)]
    L = len(v)
    P = int(patch_len)
    S = int(stride) if stride is not None else P
    if P < 1:
        raise ValueError("patchT: patch_len must be at least 1")
    if S < 1:
        raise ValueError("patchT: the stride must be at least 1")
    if L < P:
        raise ValueError("patchT: the series has %d points but the "
                         "patch length is %d" % (L, P))
    n = (L - P) // S + 1
    return {"patches": [v[i * S:i * S + P] for i in range(n)],
            "n_patches": n, "patch_len": P, "stride": S,
            "L": L,
            "covers": min(L, (n - 1) * S + P)}


def channel_independent_tokens(X, patch_len, stride=None):
    r"""One token stream per channel, weights shared across channels.

    ``X`` is :math:`L \times D`. Returns a list of per-channel patch
    lists. Nothing is mixed across channels, so the channel identity
    survives -- and because the downstream weights are shared, the
    whole model is equivariant to permuting them.
    """
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    if not Xm:
        raise ValueError("patchT: the input series is empty")
    D = len(Xm[0])
    out = []
    for d in range(D):
        col = [Xm[t][d] for t in range(len(Xm))]
        out.append(patchify(col, patch_len, stride)["patches"])
    n = len(out[0])
    return {"tokens": out, "D": D, "n_patches": n,
            "patch_len": int(patch_len),
            "n_tokens_total": D * n,
            "design": "channel-independent",
            "note": "each token holds ONE channel's subseries; the "
                    "embedding and Transformer weights are shared "
                    "across channels"}


def channel_mixed_tokens(X, patch_len, stride=None):
    r"""The alternative: one token per time position, all channels
    blended.

    Provided for contrast. Summing across channels at the first
    projection is what makes such a model permutation-**invariant**
    and so unable to say which channel a signal came from.
    """
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    if not Xm:
        raise ValueError("patchT: the input series is empty")
    mixed = [sum(row) for row in Xm]
    p = patchify(mixed, patch_len, stride)
    return {"tokens": p["patches"], "n_patches": p["n_patches"],
            "n_tokens_total": p["n_patches"],
            "design": "channel-mixing",
            "note": "channels are blended before attention, so the "
                    "channel identity is gone"}


def instance_norm(x):
    r"""Standardise one series to zero mean and unit variance.

    Applied per instance before patching and reversed afterwards, so
    a shift in level between training and test does not have to be
    learned.
    """
    v = [float(q) for q in k.vec(x)]
    if len(v) < 2:
        raise ValueError("patchT: need at least 2 points to normalise")
    m = sum(v) / len(v)
    sd = math.sqrt(sum((q - m) ** 2 for q in v) / (len(v) - 1))
    if sd <= _EPS:
        return {"normalised": [0.0] * len(v), "mean": m, "sd": 0.0,
                "degenerate": True}
    return {"normalised": [(q - m) / sd for q in v], "mean": m,
            "sd": sd, "degenerate": False}


def attention_cost(L, patch_len, stride=None, D=1,
                   channel_independent=True):
    r"""Attention-map size, patched against point-wise.

    Point-wise attention over a look-back of :math:`L` costs
    :math:`L^2` per channel. Patching reduces the sequence to
    :math:`N` tokens, so the cost falls to :math:`N^2` -- roughly a
    factor of :math:`S^2`.
    """
    P = int(patch_len)
    S = int(stride) if stride is not None else P
    Lv = int(L)
    if Lv < P:
        raise ValueError("patchT: the look-back is shorter than the "
                         "patch")
    n = (Lv - P) // S + 1
    per_channel = n * n
    return {"n_patches": n,
            "pointwise": Lv * Lv * (int(D) if channel_independent
                                    else 1),
            "patched": per_channel * (int(D) if channel_independent
                                      else 1),
            "reduction": (Lv * Lv) / max(per_channel, 1),
            "stride": S, "patch_len": P,
            "note": "the reduction is about S^2 for the same "
                    "look-back, which is what lets the model attend "
                    "a LONGER history at equal cost"}


def patchtst_encode(X, patch_len, stride=None, normalise=True):
    r"""The full front end: instance norm, patch, per-channel tokens.
    """
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    if not Xm:
        raise ValueError("patchT: the input series is empty")
    D = len(Xm[0])
    stats, cols = [], []
    for d in range(D):
        col = [Xm[t][d] for t in range(len(Xm))]
        if normalise:
            nz = instance_norm(col)
            stats.append({"mean": nz["mean"], "sd": nz["sd"]})
            col = nz["normalised"]
        else:
            stats.append({"mean": 0.0, "sd": 1.0})
        cols.append(col)
    Xn = [[cols[d][t] for d in range(D)] for t in range(len(Xm))]
    tok = channel_independent_tokens(Xn, patch_len, stride)
    return RichResult(payload={
        "estimate": tok["tokens"], "tokens": tok["tokens"],
        "D": D, "n_patches": tok["n_patches"],
        "n_tokens_total": tok["n_tokens_total"],
        "norm_stats": stats, "normalised": bool(normalise),
        "cost": attention_cost(len(Xm), patch_len, stride, D),
        "method": "PatchTST front end; Nie, Nguyen, Sinthong & "
                  "Kalagnanam (2023)",
    })


def cheatsheet():
    return ("patchT: PatchTST. A single time step is not a word, so "
            "tokenise SUBSERIES: patches of length P, stride S, "
            "giving N = (L-P)/S + 1 tokens instead of L -- attention "
            "shrinks by about S^2, which buys a longer look-back at "
            "the same cost. CHANNEL-INDEPENDENT: one token stream per "
            "channel with SHARED weights, so the model is equivariant "
            "to permuting channels. Channel-mixing blends them at the "
            "first projection and is permutation-INVARIANT instead.")


# compact alias per ledger/NAMING.md
patchtst = patchtst_encode
