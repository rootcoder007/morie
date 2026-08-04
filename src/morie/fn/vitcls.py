# morie.fn -- slice s01 (rootcoder007/morie)
"""ViT class token and position embedding: the rest of Equation (1).

SOURCE.  Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale", ICLR 2021;
arXiv:2010.11929v2.  Read from the PDF rendered as page images.

Section 3.1, p. 3: "Similar to BERT's [class] token, we prepend a
learnable embedding to the sequence of embedded patches
(z_0^0 = x_class), whose state at the output of the Transformer encoder
(z_L^0) serves as the image representation y (Eq. 4). ... Position
embeddings are added to the patch embeddings to retain positional
information.  We use standard learnable 1D position embeddings, since we
have not observed significant performance gains from using more advanced
2D-aware position embeddings."

Equation (1), p. 4:

    z_0 = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos,
    E in R^{(P^2 . C) x D},  E_pos in R^{(N+1) x D}.

This module takes the patch embeddings x_p^i E produced by vitptm and
returns z_0.  Note that E_pos has N+1 rows: the position embedding is
added to the class token as well, which the "+ E_pos" outside the
bracket in Eq. (1) makes explicit.

x_class and E_pos are learned in the paper.  Here they come from the
shared deterministic normal stream so both language arms hold identical
numbers; see _vitcore.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult

__all__ = ["vit_cls_token"]


def vit_cls_token(patches, n_patches=None, w_scale=1.0, skip=0):
    """Prepend the [class] token and add the 1D position embedding.

    Parameters
    ----------
    patches : array-like
        N-by-D matrix of patch embeddings x_p^i E.
    n_patches : int or None
        N.  Checked against the number of rows of ``patches`` if given.
    w_scale : float
        Scales x_class and E_pos.  w_scale = 0 makes both zero, so that
        z_0 is the patch embeddings with a zero row prepended.
    skip : int
        Offset into the shared deterministic stream.

    Returns
    -------
    estimate : N + 1, the sequence length the encoder sees
    z0       : (N+1)-by-D, Equation (1)
    cls      : x_class, length D
    pos      : E_pos, (N+1)-by-D
    """
    P = core.mat(patches)
    n = len(P)
    if n < 1:
        raise ValueError("vit_cls_token: need at least one patch embedding")
    d = len(P[0])
    for r in P:
        if len(r) != d:
            raise ValueError("vit_cls_token: patch embeddings have unequal length")
    if n_patches is not None and int(n_patches) != n:
        raise ValueError("vit_cls_token: n_patches does not match the number of rows of patches")
    cls = vc.draw(1, d, skip, w_scale)[0]
    pos = vc.draw(n + 1, d, int(skip) + d, w_scale)
    z0 = [[cls[j] + pos[0][j] for j in range(d)]]
    for i in range(n):
        z0.append([P[i][j] + pos[i + 1][j] for j in range(d)])
    return RichResult(
        title="ViT class token and position embedding",
        summary_lines=[("patches", n), ("sequence length", n + 1), ("embed dim", d)],
        payload={
            "estimate": float(n + 1),
            "z0": z0,
            "cls": cls,
            "pos": pos,
            "n_patches": n,
            "seq_len": n + 1,
            "embed_dim": d,
            "n": n,
            "skip_used": int(skip) + d + (n + 1) * d,
            "method": "z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos (Dosovitskiy et al. 2021, Eq. (1) p. 4)",
        },
    )


def cheatsheet():
    return "vitcls: ViT [class] token prepended plus learned 1D position embedding"


# compact alias per ledger/NAMING.md
vitclstoken = vit_cls_token
