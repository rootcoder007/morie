"""ViT self-attention block (single head, Eqs. (5)-(7))."""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["vit_self_attention"]


def _weights(r, c, base=2, scale=None):
    """Deterministic stand-in for a trained projection.

    Entries are ``core.normdraws(r * c, base)`` laid out row by row and
    divided by sqrt(fan-in).  Both language arms call the same van der
    Corput / AS 241 sequence, so they land on identical numbers without
    either having to reproduce the other's RNG.
    """
    z = core.normdraws(r * c, base)
    s = (1.0 / math.sqrt(r)) if scale is None else float(scale)
    return [[z[i * c + j] * s for j in range(c)] for i in range(r)]


def _rect(A, name):
    w = len(A[0])
    for r in A:
        if len(r) != w:
            raise ValueError("vit_self_attention: " + name + " is ragged")
    return w


def vit_self_attention(q, k, v, mask=None):
    """Scaled dot-product self-attention.

    Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale*, ICLR 2021,
    arXiv:2010.11929v2, Appendix A, p. 13:

        A     = softmax(q k^T / sqrt(D_h)),  A in R^{N x N}     (6)
        SA(z) = A v                                             (7)

    q, k and v are the projections of Eq. (5); this function takes them
    already projected, which is what the module name (a block, not a
    whole encoder) asks for.

    Parameters
    ----------
    q, k : array-like
        N-by-D_h query and key matrices.
    v : array-like
        N-by-D_v value matrix.
    mask : array-like or None
        One row per query, one column per key; a zero entry blocks that
        key for that query, a non-zero entry keeps it (so a boolean
        keep-mask works unchanged).  ``None`` keeps everything.

    Returns
    -------
    estimate : mean of the attention output
    attn     : the N-by-N matrix A of Eq. (6); every row sums to 1
    output   : the N-by-D_v matrix A v of Eq. (7)
    """
    Q = core.mat(q)
    K = core.mat(k)
    V = core.mat(v)
    n = len(Q)
    nk = len(K)
    if n == 0 or nk == 0 or len(V) == 0:
        raise ValueError("vit_self_attention: q, k and v must be non-empty")
    if len(V) != nk:
        raise ValueError("vit_self_attention: k and v must have the same number of rows")
    dh = _rect(Q, "q")
    if _rect(K, "k") != dh:
        raise ValueError("vit_self_attention: q and k must have the same width")
    dv = _rect(V, "v")
    M = None
    if mask is not None:
        M = core.mat(mask)
        if len(M) != n or len(M[0]) != nk:
            raise ValueError("vit_self_attention: mask must be one row per query and one column per key")
    scale = 1.0 / math.sqrt(dh)
    A = []
    for i in range(n):
        s = []
        keep = []
        for j in range(nk):
            if M is not None and M[i][j] == 0.0:
                continue
            acc = 0.0
            for p in range(dh):
                acc += Q[i][p] * K[j][p]
            s.append(acc * scale)
            keep.append(j)
        if not keep:
            raise ValueError("vit_self_attention: mask leaves a query with no keys")
        w = core.softmax(s)
        row = [0.0] * nk
        for t, j in enumerate(keep):
            row[j] = w[t]
        A.append(row)
    out = core.matmul(A, V)
    tot = 0.0
    for r in out:
        for e in r:
            tot += e
    return RichResult(payload={
        "estimate": tot / (n * dv),
        "attn": A,
        "output": out,
        "n": n,
        "d_head": dh,
        "d_value": dv,
        "method": "ViT self-attention block",
    })


def cheatsheet():
    return "vitatt: ViT self-attention block"
