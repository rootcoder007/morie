# morie.fn -- slice s01 (rootcoder007/morie)
"""Vision Transformer forward pass, Equations (1)-(4).

SOURCE.  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D.,
Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G.,
Gelly, S., Uszkoreit, J. and Houlsby, N. (2021), "An Image is Worth
16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021;
arXiv:2010.11929v2.  Read from the PDF rendered as page images, not from
the text layer.

Equations (1)-(4), p. 4:

    z_0   = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos,
            E in R^{(P^2 . C) x D},  E_pos in R^{(N+1) x D}      (1)
    z'_l  = MSA(LN(z_{l-1})) + z_{l-1},        l = 1 ... L       (2)
    z_l   = MLP(LN(z'_l)) + z'_l,              l = 1 ... L       (3)
    y     = LN(z_L^0)                                            (4)

Section 3.1, p. 3: "The Transformer encoder (Vaswani et al., 2017)
consists of alternating layers of multiheaded self-attention (MSA, see
Appendix A) and MLP blocks (Eq. 2, 3).  Layernorm (LN) is applied before
every block, and residual connections after every block."  That is the
pre-norm arrangement Eqs. (2) and (3) show: LN inside, the residual
added outside.

MSA is Appendix A, p. 13, Eqs. (5)-(8): per head, [q, k, v] = z U_qkv
with U_qkv in R^{D x 3 D_h}; A = softmax(q k^T / sqrt(D_h));
SA(z) = A v; and MSA(z) = [SA_1(z); ...; SA_k(z)] U_msa with U_msa in
R^{(k . D_h) x D}.  D_h is "typically set to D/k", which this module
requires: num_heads must divide embed_dim.

The MLP hidden width is 4D, the ratio in Table 1, p. 5 (768/3072,
1024/4096, 1280/5120).

y is the image representation.  Section 3.1, p. 3: the state of the
class token at the output of the encoder, z_L^0, "serves as the image
representation y (Eq. 4)".  Attaching a head to y is vitfsv.

This assembles vitptm (Eq. 1 left), vitcls (Eq. 1 right), vitatt
(Eqs. 6-7) and vitmlp (the MLP of Eq. 3) rather than restating them.
All parameters -- E, x_class, E_pos, U_qkv and U_msa per head and layer,
W_1 and W_2 per layer -- are drawn in that order from the single shared
deterministic normal stream, so the Python and R arms hold identical
weights.  The network is untrained by construction; this is a reference
implementation of the architecture, not of a fitted model.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult
from .vitatt import vit_self_attention
from .vitcls import vit_cls_token
from .vitmlp import vit_mlp_block
from .vitptm import vit_patch_embed

__all__ = ["vit_forward"]


def vit_forward(x, patch_size, embed_dim, num_heads, num_layers, w_scale=1.0, mlp_ratio=4):
    """Run Equations (1)-(4) on one image and return y = LN(z_L^0).

    Parameters
    ----------
    x : array-like
        H-by-W matrix (C = 1), or a list of C such matrices.
    patch_size : int
        P.  Must divide H and W.
    embed_dim : int
        D.  Must be divisible by num_heads.
    num_heads : int
        k.  D_h = D/k.
    num_layers : int
        L.  L = 0 is allowed and returns LN(z_0^0), the degenerate case
        used as an anchor.
    w_scale : float
        Scales every parameter matrix.  w_scale = 0 makes MSA and MLP
        return zeros, so Eqs. (2)-(3) reduce to z_l = z_{l-1}.
    mlp_ratio : int
        Hidden width of the MLP as a multiple of D; 4 per Table 1, p. 5.

    Returns
    -------
    estimate : mean of y
    y        : the image representation, length D, Eq. (4)
    z0, zL   : (N+1)-by-D, Eqs. (1) and (3)
    attn     : the attention matrix of the last head of the last layer
    """
    d = int(embed_dim)
    k = int(num_heads)
    L = int(num_layers)
    if d < 1:
        raise ValueError("vit_forward: embed_dim must be a positive integer")
    if k < 1:
        raise ValueError("vit_forward: num_heads must be a positive integer")
    if L < 0:
        raise ValueError("vit_forward: num_layers must be non-negative")
    if d % k != 0:
        raise ValueError("vit_forward: num_heads must divide embed_dim (D_h = D/k)")
    dh = d // k
    ratio = int(mlp_ratio)
    if ratio < 1:
        raise ValueError("vit_forward: mlp_ratio must be a positive integer")
    hid = ratio * d

    pe = vit_patch_embed(x, patch_size, d, w_scale, 0)
    skip = pe["skip_used"]
    ct = vit_cls_token(pe["embeddings"], pe["n_patches"], w_scale, skip)
    skip = ct["skip_used"]
    z0 = ct["z0"]
    ns = len(z0)

    z = [row[:] for row in z0]
    attn = None
    for _ in range(L):
        zn = vc.layernorm_rows(z)
        heads = []
        for _h in range(k):
            Uq = vc.draw(d, dh, skip, w_scale)
            skip += d * dh
            Uk = vc.draw(d, dh, skip, w_scale)
            skip += d * dh
            Uv = vc.draw(d, dh, skip, w_scale)
            skip += d * dh
            sa = vit_self_attention(
                core.matmul(zn, Uq), core.matmul(zn, Uk), core.matmul(zn, Uv)
            )
            heads.append(sa["output"])
            attn = sa["attn"]
        cat = [[heads[hh][i][c] for hh in range(k) for c in range(dh)] for i in range(ns)]
        Umsa = vc.draw(k * dh, d, skip, w_scale)
        skip += k * dh * d
        msa = core.matmul(cat, Umsa)
        z = [[z[i][j] + msa[i][j] for j in range(d)] for i in range(ns)]
        zn2 = vc.layernorm_rows(z)
        mb = vit_mlp_block(zn2, hid, w_scale, skip)
        skip = mb["skip_used"]
        mo = mb["output"]
        z = [[z[i][j] + mo[i][j] for j in range(d)] for i in range(ns)]

    y = vc.layernorm(z[0])
    s = 0.0
    for t in y:
        s += t
    return RichResult(
        title="Vision Transformer forward pass",
        summary_lines=[
            ("patches", pe["n_patches"]),
            ("sequence length", ns),
            ("embed dim", d),
            ("heads", k),
            ("layers", L),
        ],
        payload={
            "estimate": s / d,
            "y": y,
            "z0": z0,
            "zL": z,
            "attn": attn,
            "patches": pe["patches"],
            "n_patches": pe["n_patches"],
            "seq_len": ns,
            "embed_dim": d,
            "d_head": dh,
            "num_heads": k,
            "num_layers": L,
            "hidden_dim": hid,
            "n": pe["n_patches"],
            "skip_used": skip,
            "method": "z_0 Eq.(1); z'_l = MSA(LN(z_{l-1})) + z_{l-1} Eq.(2); z_l = MLP(LN(z'_l)) + z'_l Eq.(3); y = LN(z_L^0) Eq.(4) (Dosovitskiy et al. 2021, p. 4)",
        },
    )


def cheatsheet():
    return "vitfwd: Vision Transformer forward pass, Eqs. (1)-(4) of Dosovitskiy et al. (2021)"


# compact alias per ledger/NAMING.md
vitforward = vit_forward
