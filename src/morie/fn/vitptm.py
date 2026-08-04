"""ViT patch embedding (Eq. (1), the x_p E part)."""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["vit_patch_embed"]


def _weights(r, c, base=2):
    z = core.normdraws(r * c, base)
    s = 1.0 / math.sqrt(r)
    return [[z[i * c + j] * s for j in range(c)] for i in range(r)]


def vit_patch_embed(image, patch_size, embed_dim, E=None):
    """Cut an image into P-by-P patches, flatten them, project to D.

    Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words*, ICLR
    2021, arXiv:2010.11929v2, Section 3.1, p. 3 and Eq. (1), p. 4:
    the image x in R^{H x W x C} is reshaped into a sequence of
    flattened 2D patches x_p in R^{N x (P^2 . C)} with N = H W / P^2,
    and mapped to D dimensions by a trainable linear projection
    E in R^{(P^2 . C) x D}.  The paper calls this a linear projection;
    it is the stride-P convolution of the module name only in the sense
    that a stride-P conv with a P-by-P kernel *is* that projection.

    This arm takes a single-channel image (C = 1), so the patch
    dimension is P^2.  Patches are emitted in row-major order over the
    patch grid, and each patch is flattened row-major.

    Parameters
    ----------
    image : array-like
        H-by-W matrix, with P dividing both H and W.
    patch_size : int
        P.
    embed_dim : int
        D.
    E : array-like or None
        (P^2)-by-D projection.  ``None`` uses a deterministic one.

    Returns
    -------
    estimate   : N, the number of patches
    patches    : N-by-P^2 matrix of flattened patches, x_p
    embeddings : N-by-D matrix x_p E
    """
    X = core.mat(image)
    H = len(X)
    if H == 0:
        raise ValueError("vit_patch_embed: image is empty")
    W = len(X[0])
    for r in X:
        if len(r) != W:
            raise ValueError("vit_patch_embed: image is ragged")
    P = int(patch_size)
    D = int(embed_dim)
    if P < 1:
        raise ValueError("vit_patch_embed: patch_size must be at least 1")
    if D < 1:
        raise ValueError("vit_patch_embed: embed_dim must be at least 1")
    if H % P != 0 or W % P != 0:
        raise ValueError("vit_patch_embed: patch_size must divide both image dimensions")
    pd = P * P
    patches = []
    for gi in range(H // P):
        for gj in range(W // P):
            row = []
            for a in range(P):
                for b in range(P):
                    row.append(X[gi * P + a][gj * P + b])
            patches.append(row)
    N = len(patches)
    Emat = _weights(pd, D) if E is None else core.mat(E)
    if len(Emat) != pd or len(Emat[0]) != D:
        raise ValueError("vit_patch_embed: E must be (patch_size^2)-by-embed_dim")
    Z = core.matmul(patches, Emat)
    return RichResult(payload={
        "estimate": float(N),
        "patches": patches,
        "embeddings": Z,
        "n_patches": N,
        "patch_dim": pd,
        "embed_dim": D,
        "n": N,
        "method": "ViT patch embedding via 2D conv",
    })


def cheatsheet():
    return "vitptm: ViT patch embedding via 2D conv"
