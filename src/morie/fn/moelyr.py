# morie.fn -- slice s03 (rootcoder007/morie)
"""Sparsely-gated mixture-of-experts layer.

Source consulted (FETCHED): Shazeer, N. et al. (2017).  Outrageously
large neural networks: the sparsely-gated mixture-of-experts layer.
*ICLR* (arXiv:1701.06538).  Equation (3):

    y = sum_(i=1)^n G(x)_i E_i(x)

with only the nonzero gates evaluated.  The gate is noisy top-k,
equations (4)-(6):

    G(x)        = Softmax( KeepTopK( H(x), k ) )
    H(x)_i      = (x . W_g)_i + StandardNormal() . Softplus((x . W_noise)_i)
    KeepTopK(v, k)_i = v_i if v_i is among the top k, else -infinity

DETERMINISM.  StandardNormal() would make the routing irreproducible, so
the noise is supplied by the caller as ``noise`` and defaults to zero,
which is the paper's own inference-time behaviour.  Nothing consults a
generator.

Ties in the top-k selection break to the lowest expert index, so a batch
with equal logits routes identically in both arms.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["moe_layer"]


def moe_layer(y, x=None, W_g=None, experts=None, top_k=2, W_noise=None,
              noise=None):
    """Route x to the top-k experts and mix their outputs.

    Parameters
    ----------
    y : array-like
        The input x.  (First slot, for signature stability.)
    x : array-like, optional
        The input; wins over ``y``.
    W_g : 2-D array-like
        Gate weights, rows indexed by input unit, columns by expert.
    experts : list of callable or 2-D array-like
        Either callables ``x -> vector``, or a matrix whose row i is the
        output of expert i.
    top_k : int
        Number of experts kept.
    W_noise : 2-D array-like, optional
        Noise-scale weights of equation (5).
    noise : array-like, optional
        The StandardNormal() draws; zeros by default.

    Returns
    -------
    RichResult with payload:
        estimate : the first output unit
        out      : the mixed output
        gate     : G(x), zero off the top-k
        chosen   : indices of the kept experts, ascending
        h        : H(x) before KeepTopK
    """
    v = k.vec(x if x is not None else y)
    Wg = k.mat(W_g)
    h = k.matvec(k.tr(Wg), v)
    m = len(h)
    if W_noise is not None:
        wn = k.matvec(k.tr(k.mat(W_noise)), v)
        z = k.vec(noise) if noise is not None else [0.0] * m
        for i in range(m):
            sp = math.log1p(math.exp(-abs(wn[i]))) + max(wn[i], 0.0)
            h[i] = h[i] + z[i] * sp
    kk = int(top_k)
    if kk > m:
        kk = m
    order = sorted(range(m), key=lambda i: (-h[i], i))
    chosen = sorted(order[:kk])
    keep = [h[i] if i in chosen else float("-inf") for i in range(m)]
    sub = k.softmax([h[i] for i in chosen])
    gate = [0.0] * m
    for j, i in enumerate(chosen):
        gate[i] = sub[j]
    outs = []
    for i in range(m):
        if callable(experts[i]):
            outs.append(k.vec(experts[i](v)))
        else:
            outs.append(k.vec(experts[i]))
    d = len(outs[0]) if outs else 0
    out = [0.0] * d
    for i in chosen:
        for j in range(d):
            out[j] += gate[i] * outs[i][j]
    return RichResult(
        title="Mixture-of-experts layer",
        summary_lines=[("experts", m), ("top-k", kk)],
        payload={
            "estimate": out[0] if out else float("nan"),
            "out": out,
            "gate": gate,
            "chosen": chosen,
            "h": h,
            "keep": keep,
            "method": "Sparsely-gated MoE layer (Shazeer et al. 2017, eqs. 3-6)",
        },
    )


def cheatsheet():
    return "moelyr: MoE feed-forward layer with router + expert mix"


moelayer = moe_layer
