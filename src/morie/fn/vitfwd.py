"""Vision Transformer forward pass (Eqs. (1)-(4) and (8))."""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult
from .vitatt import vit_self_attention
from .vitcls import vit_cls_token
from .vitmlp import vit_mlp_block
from .vitptm import vit_patch_embed

__all__ = ["vit_forward"]


def _w(r, c, base=2, mult=1.0):
    z = core.normdraws(r * c, base)
    s = mult / math.sqrt(r)
    return [[z[i * c + j] * s for j in range(c)] for i in range(r)]


def _layernorm(u, eps):
    """LN with unit gain and zero bias (Ba, Kiros and Hinton 2016)."""
    d = len(u)
    m = 0.0
    for e in u:
        m += e
    m /= d
    s = 0.0
    for e in u:
        s += (e - m) * (e - m)
    s = math.sqrt(s / d + eps)
    if s == 0.0:
        return [0.0] * d
    return [(e - m) / s for e in u]


def vit_forward(x, patch_size, embed_dim, num_heads, num_layers,
                mlp_ratio=4, scale=None, eps=1e-5, E=None, pos=None):
    """The encoder of Eqs. (1)-(4), with MSA as Eq. (8).

    Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words*, ICLR
    2021, arXiv:2010.11929v2, p. 4:

        z_0    = [x_class; x_p^1 E; ... ; x_p^N E] + E_pos            (1)
        z'_l   = MSA(LN(z_{l-1})) + z_{l-1},        l = 1 ... L       (2)
        z_l    = MLP(LN(z'_l)) + z'_l,              l = 1 ... L       (3)
        y      = LN(z_L^0)                                            (4)

    and Appendix A, p. 13, Eq. (8):

        MSA(z) = [SA_1(z); SA_2(z); ... ; SA_k(z)] U_msa,
        U_msa in R^{(k . D_h) x D},  D_h = D / k.

    Layer normalisation is not defined in the paper, which cites Ba,
    J. L., Kiros, J. R. and Hinton, G. E. (2016), *Layer
    Normalization*, arXiv:1607.06450: LN(u) = (u - mean(u)) /
    sqrt(var(u) + eps), with the population variance, unit gain and
    zero bias.  ``eps`` is an implementation choice, not the paper's.

    All weights are deterministic (see ``vitatt._weights``) so that the
    Python and R arms return the same numbers.  ``scale`` multiplies the
    attention and MLP weights only; ``scale = 0`` switches both blocks
    off, leaving the residual stream of Eqs. (2)-(3) untouched, which is
    what the closed-form check uses.

    Parameters
    ----------
    x : array-like
        H-by-W single-channel image, patch_size dividing H and W.
    patch_size, embed_dim, num_heads, num_layers : int
        P, D, k and L.  k must divide D.
    mlp_ratio : int
        Hidden width of the MLP as a multiple of D (4 in the paper's
        configurations, Table 1).
    scale : float or None
        Multiplier on the block weights; ``None`` means 1.
    eps : float
        LN epsilon.
    E, pos : array-like or None
        Patch projection and position embedding; ``None`` uses the
        deterministic ones.

    Returns
    -------
    estimate : mean of y
    y        : the length-D image representation LN(z_L^0)
    tokens   : the final (N+1)-by-D token matrix z_L
    """
    D = int(embed_dim)
    k = int(num_heads)
    L = int(num_layers)
    P = int(patch_size)
    if k < 1:
        raise ValueError("vit_forward: num_heads must be at least 1")
    if L < 1:
        raise ValueError("vit_forward: num_layers must be at least 1")
    if D % k != 0:
        raise ValueError("vit_forward: num_heads must divide embed_dim")
    if int(mlp_ratio) < 1:
        raise ValueError("vit_forward: mlp_ratio must be at least 1")
    mult = 1.0 if scale is None else float(scale)
    dh = D // k
    Emat = _w(P * P, D, 2) if E is None else core.mat(E)
    pe = vit_patch_embed(x, P, D, E=Emat)
    zp = pe["embeddings"]
    N = pe["n_patches"]
    posm = _w(N + 1, D, 3) if pos is None else core.mat(pos)
    z = vit_cls_token(zp, N, cls=None, pos=posm)["tokens"]
    nt = N + 1
    Hd = int(mlp_ratio) * D
    for l in range(1, L + 1):
        u = [_layernorm(r, eps) for r in z]
        heads = []
        for h in range(1, k + 1):
            U = _w(D, 3 * dh, 2 + (l - 1) * k + (h - 1), mult)
            proj = core.matmul(u, U)
            q = [[r[j] for j in range(dh)] for r in proj]
            kk = [[r[dh + j] for j in range(dh)] for r in proj]
            vv = [[r[2 * dh + j] for j in range(dh)] for r in proj]
            heads.append(vit_self_attention(q, kk, vv)["output"])
        cat = [[heads[h][i][j] for h in range(k) for j in range(dh)] for i in range(nt)]
        Umsa = _w(k * dh, D, 101 + l, mult)
        msa = core.matmul(cat, Umsa)
        zp1 = [[msa[i][j] + z[i][j] for j in range(D)] for i in range(nt)]
        un = [_layernorm(r, eps) for r in zp1]
        mlp = vit_mlp_block(un, Hd,
                            W1=_w(D, Hd, 201 + l, mult),
                            W2=_w(Hd, D, 301 + l, mult))["output"]
        z = [[mlp[i][j] + zp1[i][j] for j in range(D)] for i in range(nt)]
    y = _layernorm(z[0], eps)
    m = 0.0
    for e in y:
        m += e
    return RichResult(payload={
        "estimate": m / D,
        "y": y,
        "tokens": z,
        "n_patches": N,
        "n_tokens": nt,
        "embed_dim": D,
        "n_heads": k,
        "n_layers": L,
        "n": N,
        "method": "Vision Transformer forward pass",
    })


def cheatsheet():
    return "vitfwd: Vision Transformer forward pass"
