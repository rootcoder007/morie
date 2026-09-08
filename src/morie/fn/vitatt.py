# morie.fn -- slice s01 (rootcoder007/morie)
"""ViT self-attention: A = softmax(q k^T / sqrt(D_h)), SA(z) = A v.

SOURCE.  Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale", ICLR 2021;
arXiv:2010.11929v2, Appendix A "Multihead Self-attention", p. 13.  Read
from the PDF rendered as a page image.

    [q, k, v] = z U_qkv,   U_qkv in R^{D x 3 D_h}          (5)
    A = softmax(q k^T / sqrt(D_h)),   A in R^{N x N}       (6)
    SA(z) = A v                                            (7)
    MSA(z) = [SA_1(z); SA_2(z); ...; SA_k(z)] U_msa,
        U_msa in R^{(k . D_h) x D}                         (8)

with D_h "typically set to D/k" (text under Eq. (8), p. 13).  The
underlying construction is Vaswani et al. (2017), "Attention Is All You
Need", NeurIPS 30, which the appendix cites; the paper reproduces it
unchanged.

This module is Eqs. (6) and (7): it takes q, k, v already projected and
returns the attention matrix and A v.  The projection (5) and the
multi-head concatenation (8) are done by vitfwd, which is where U_qkv
and U_msa live.

The softmax in Eq. (6) is over the key axis, i.e. row-wise on q k^T, so
that each row of A is a set of weights summing to one -- "a weighted sum
over all values v in the sequence" (p. 13).  The scale is 1/sqrt(D_h)
where D_h is the query/key width, not the value width; q and k must
therefore share a width, while v need not.

The mask is not in the paper -- ViT attends over the whole sequence --
and is carried because the stub's signature declares it.  A zero (or
FALSE) entry means "this key is not visible to this query" and is set to
-Inf before the softmax; an all-zero mask row is an error rather than a
silent NaN.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _vitcore as vc  # noqa: F401

from ._richresult import RichResult

__all__ = ["vit_self_attention"]


def vit_self_attention(q, k, v, mask=None):
    """Scaled dot-product attention, Eqs. (6)-(7) p. 13.

    Parameters
    ----------
    q : array-like
        N-by-D_h queries.
    k : array-like
        M-by-D_h keys.  D_h must match q.
    v : array-like
        M-by-D_v values.
    mask : array-like or None
        N-by-M of 0/1 (or FALSE/TRUE).  Zero entries are excluded.

    Returns
    -------
    estimate : mean of the attention output
    attn     : A, N-by-M, rows summing to 1
    output   : SA = A v, N-by-D_v
    scale    : 1/sqrt(D_h)
    """
    Q = core.mat(q)
    K = core.mat(k)
    V = core.mat(v)
    n = len(Q)
    m = len(K)
    if n < 1 or m < 1:
        raise ValueError("vit_self_attention: q and k must have at least one row")
    dh = len(Q[0])
    if len(K[0]) != dh:
        raise ValueError("vit_self_attention: q and k must have the same width D_h")
    if len(V) != m:
        raise ValueError("vit_self_attention: k and v must have the same number of rows")
    dv = len(V[0])
    scale = 1.0 / math.sqrt(dh)
    if mask is not None:
        M = core.mat(mask)
        if len(M) != n or len(M[0]) != m:
            raise ValueError("vit_self_attention: mask must be N-by-M")
    else:
        M = None
    A = []
    for i in range(n):
        logits = []
        for j in range(m):
            s = 0.0
            for t in range(dh):
                s += Q[i][t] * K[j][t]
            logits.append(s * scale)
        if M is not None:
            allowed = 0
            for j in range(m):
                if M[i][j] != 0.0:
                    allowed += 1
                else:
                    logits[j] = float("-inf")
            if allowed == 0:
                raise ValueError("vit_self_attention: a mask row excludes every key")
        A.append(core.softmax(logits))
    out = []
    for i in range(n):
        row = []
        for c in range(dv):
            s = 0.0
            for j in range(m):
                s += A[i][j] * V[j][c]
            row.append(s)
        out.append(row)
    tot = 0.0
    for r in out:
        for x in r:
            tot += x
    return RichResult(
        title="ViT self-attention",
        summary_lines=[("queries", n), ("keys", m), ("D_h", dh)],
        payload={
            "estimate": tot / (n * dv),
            "attn": A,
            "output": out,
            "scale": scale,
            "n_query": n,
            "n_key": m,
            "d_head": dh,
            "d_value": dv,
            "n": n,
            "method": "A = softmax(q k^T / sqrt(D_h)); SA(z) = A v (Dosovitskiy et al. 2021, Eqs. (6)-(7) p. 13)",
        },
    )


def cheatsheet():
    return "vitatt: ViT scaled dot-product self-attention, softmax(qk'/sqrt(D_h))v"


# compact alias per ledger/NAMING.md
vitselfattention = vit_self_attention
