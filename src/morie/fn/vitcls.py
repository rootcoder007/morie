"""ViT [CLS] token and position embedding (Eq. (1))."""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["vit_cls_token"]


def _weights(r, c, base=2):
    z = core.normdraws(r * c, base)
    s = 1.0 / math.sqrt(r)
    return [[z[i * c + j] * s for j in range(c)] for i in range(r)]


def vit_cls_token(patches, n_patches, cls=None, pos=None):
    """Prepend the class token and add the 1D position embedding.

    Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words*, ICLR
    2021, arXiv:2010.11929v2, Eq. (1), p. 4:

        z_0 = [x_class; x_p^1 E; x_p^2 E; ... ; x_p^N E] + E_pos,
        E_pos in R^{(N+1) x D}

    Section 3.1, p. 3: the position embeddings are standard learnable
    1D ones, not 2D-aware; z_0^0 = x_class.

    Parameters
    ----------
    patches : array-like
        N-by-D matrix of patch embeddings (x_p E).
    n_patches : int
        N; must match the number of rows of ``patches``.
    cls : array-like or None
        Length-D class token.  ``None`` is the zero vector.
    pos : array-like or None
        (N+1)-by-D position embedding.  ``None`` uses a deterministic
        one; pass a zero matrix to switch it off.

    Returns
    -------
    estimate : N + 1, the sequence length the encoder sees
    tokens   : the (N+1)-by-D matrix z_0
    """
    Xp = core.mat(patches)
    N = len(Xp)
    if N == 0:
        raise ValueError("vit_cls_token: patches is empty")
    D = len(Xp[0])
    for r in Xp:
        if len(r) != D:
            raise ValueError("vit_cls_token: patches is ragged")
    if int(n_patches) != N:
        raise ValueError("vit_cls_token: n_patches does not match the number of rows of patches")
    c = [0.0] * D if cls is None else [float(e) for e in core.vec(cls)]
    if len(c) != D:
        raise ValueError("vit_cls_token: cls must have length embed_dim")
    E = _weights(N + 1, D) if pos is None else core.mat(pos)
    if len(E) != N + 1 or len(E[0]) != D:
        raise ValueError("vit_cls_token: pos must be (n_patches+1)-by-embed_dim")
    z = [[c[j] + E[0][j] for j in range(D)]]
    for i in range(N):
        z.append([Xp[i][j] + E[i + 1][j] for j in range(D)])
    return RichResult(payload={
        "estimate": float(N + 1),
        "tokens": z,
        "pos": E,
        "cls": c,
        "n_tokens": N + 1,
        "embed_dim": D,
        "n": N,
        "method": "ViT [CLS] token + position embedding",
    })


def cheatsheet():
    return "vitcls: ViT [CLS] token + position embedding"
