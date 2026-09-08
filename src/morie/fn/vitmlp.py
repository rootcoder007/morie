# morie.fn -- slice s01 (rootcoder007/morie)
"""ViT MLP block: two layers with a GELU non-linearity.

SOURCE.  Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale", ICLR 2021;
arXiv:2010.11929v2.  Read from the PDF rendered as page images.

Page 4, the sentence immediately above Equation (1): "The MLP contains
two layers with a GELU non-linearity."  The block appears as MLP(.) in
Equation (3), p. 4:  z_l = MLP(LN(z'_l)) + z'_l,  l = 1 ... L.  The
residual addition and the LN belong to Eq. (3) and are applied by
vitfwd; this module is the MLP itself,

    MLP(x) = GELU(x W_1) W_2,   W_1 in R^{D x H},  W_2 in R^{H x D}.

The hidden width H is the "MLP size" column of Table 1, p. 5, which is
4D for every variant listed: ViT-Base D = 768, MLP 3072; ViT-Large
D = 1024, MLP 4096; ViT-Huge D = 1280, MLP 5120.  ``hidden_dim`` defaults
to 4D accordingly.

The paper names GELU but does not restate it; the non-linearity is
Hendrycks, D. and Gimpel, K. (2016), "Gaussian Error Linear Units
(GELUs)", arXiv:1606.08415, exactly GELU(x) = x Phi(x), taken from the
shared core.

The paper does not say whether the two layers carry bias terms.  They
are taken to be zero here.  That is this implementation's choice, stated
rather than attributed.

W_1 and W_2 are learned in the paper; here they come from the shared
deterministic normal stream so both language arms hold identical
numbers; see _vitcore.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult

__all__ = ["vit_mlp_block"]


def vit_mlp_block(x, hidden_dim=None, w_scale=1.0, skip=0):
    """MLP(x) = GELU(x W_1) W_2.

    Parameters
    ----------
    x : array-like
        N-by-D matrix, or a length-D vector treated as one row.
    hidden_dim : int or None
        H.  Defaults to 4D, the ratio in Table 1, p. 5.
    w_scale : float
        Scales W_1 and W_2.  w_scale = 0 makes the block return zeros.
    skip : int
        Offset into the shared deterministic stream.

    Returns
    -------
    estimate    : mean of the block output
    output      : MLP(x), N-by-D
    hidden_pre  : x W_1, N-by-H, before GELU
    hidden      : GELU(x W_1), N-by-H
    w1, w2      : the two weight matrices
    """
    X = core.mat(x)
    n = len(X)
    if n < 1:
        raise ValueError("vit_mlp_block: x must have at least one row")
    d = len(X[0])
    for r in X:
        if len(r) != d:
            raise ValueError("vit_mlp_block: rows of x have unequal length")
    h = 4 * d if hidden_dim is None else int(hidden_dim)
    if h < 1:
        raise ValueError("vit_mlp_block: hidden_dim must be a positive integer")
    W1 = vc.draw(d, h, skip, w_scale)
    W2 = vc.draw(h, d, int(skip) + d * h, w_scale)
    pre = core.matmul(X, W1)
    act = [[core.gelu(z) for z in r] for r in pre]
    out = core.matmul(act, W2)
    tot = 0.0
    for r in out:
        for z in r:
            tot += z
    return RichResult(
        title="ViT MLP block",
        summary_lines=[("rows", n), ("embed dim", d), ("hidden dim", h)],
        payload={
            "estimate": tot / (n * d),
            "output": out,
            "hidden_pre": pre,
            "hidden": act,
            "w1": W1,
            "w2": W2,
            "embed_dim": d,
            "hidden_dim": h,
            "n": n,
            "skip_used": int(skip) + 2 * d * h,
            "method": "MLP(x) = GELU(x W_1) W_2, two layers with a GELU non-linearity (Dosovitskiy et al. 2021, p. 4; hidden = 4D per Table 1 p. 5)",
        },
    )


def cheatsheet():
    return "vitmlp: ViT MLP block, GELU(x W1) W2 with hidden width 4D"


# compact alias per ledger/NAMING.md
vitmlpblock = vit_mlp_block
