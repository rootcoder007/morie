"""ViT MLP block (Eq. (3)); two layers with a GELU non-linearity."""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["vit_mlp_block"]


def _weights(r, c, base=2):
    z = core.normdraws(r * c, base)
    s = 1.0 / math.sqrt(r)
    return [[z[i * c + j] * s for j in range(c)] for i in range(r)]


def vit_mlp_block(x, hidden_dim, W1=None, b1=None, W2=None, b2=None):
    """MLP(x) = GELU(x W1 + b1) W2 + b2.

    Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words*, ICLR
    2021, arXiv:2010.11929v2, p. 4, immediately above Eq. (1): "The MLP
    contains two layers with a GELU non-linearity"; the block appears
    as MLP(LN(z')) in Eq. (3).  The paper gives no closed form for the
    non-linearity, so GELU is the exact one of its own source,
    Hendrycks and Gimpel (2016), *Gaussian Error Linear Units (GELUs)*,
    arXiv:1606.08415, GELU(u) = u Phi(u); this is ``core.gelu``, not
    the tanh approximation.

    Parameters
    ----------
    x : array-like
        N-by-D matrix of tokens (a length-D vector is one token).
    hidden_dim : int
        Width of the hidden layer.
    W1, b1, W2, b2 : array-like or None
        D-by-H, H, H-by-D and D parameters.  ``None`` uses
        deterministic weights and zero biases.

    Returns
    -------
    estimate : mean of the output
    output   : the N-by-D matrix MLP(x)
    hidden   : the N-by-H matrix GELU(x W1 + b1)
    """
    X = core.mat(x)
    N = len(X)
    if N == 0:
        raise ValueError("vit_mlp_block: x is empty")
    D = len(X[0])
    for r in X:
        if len(r) != D:
            raise ValueError("vit_mlp_block: x is ragged")
    Hd = int(hidden_dim)
    if Hd < 1:
        raise ValueError("vit_mlp_block: hidden_dim must be at least 1")
    A1 = _weights(D, Hd, 2) if W1 is None else core.mat(W1)
    A2 = _weights(Hd, D, 3) if W2 is None else core.mat(W2)
    c1 = [0.0] * Hd if b1 is None else [float(e) for e in core.vec(b1)]
    c2 = [0.0] * D if b2 is None else [float(e) for e in core.vec(b2)]
    if len(A1) != D or len(A1[0]) != Hd:
        raise ValueError("vit_mlp_block: W1 must be embed_dim-by-hidden_dim")
    if len(A2) != Hd or len(A2[0]) != D:
        raise ValueError("vit_mlp_block: W2 must be hidden_dim-by-embed_dim")
    if len(c1) != Hd or len(c2) != D:
        raise ValueError("vit_mlp_block: bias lengths do not match the layer widths")
    pre = core.matmul(X, A1)
    hid = [[core.gelu(pre[i][j] + c1[j]) for j in range(Hd)] for i in range(N)]
    out = core.matmul(hid, A2)
    tot = 0.0
    for i in range(N):
        for j in range(D):
            out[i][j] += c2[j]
            tot += out[i][j]
    return RichResult(payload={
        "estimate": tot / (N * D),
        "output": out,
        "hidden": hid,
        "n": N,
        "embed_dim": D,
        "hidden_dim": Hd,
        "method": "ViT MLP block",
    })


def cheatsheet():
    return "vitmlp: ViT MLP block"
