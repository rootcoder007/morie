# morie.fn -- slice s01 (rootcoder007/morie)
"""ViT patch embedding: the flattened-patch reshape and its linear map.

SOURCE.  Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D.,
Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G.,
Gelly, S., Uszkoreit, J. and Houlsby, N. (2021), "An Image is Worth
16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021;
arXiv:2010.11929v2.  Read from the PDF rendered as page images, not from
the text layer.

Section 3.1, p. 3: "we reshape the image x in R^{H x W x C} into a
sequence of flattened 2D patches x_p in R^{N x (P^2 . C)}, where (H, W)
is the resolution of the original image, C is the number of channels,
(P, P) is the resolution of each image patch, and N = HW/P^2 is the
resulting number of patches, which also serves as the effective input
sequence length for the Transformer.  The Transformer uses constant
latent vector size D through all of its layers, so we flatten the
patches and map to D dimensions with a trainable linear projection
(Eq. 1)."

Equation (1), p. 4, gives that projection as E in R^{(P^2 . C) x D}, and
the patch embeddings are the products x_p^i E.  This module produces
x_p and x_p E; the class token and the position embeddings, which are
the rest of Eq. (1), are vitcls.

The paper does not fix an ordering inside a flattened patch, only its
length P^2 . C.  This implementation uses channel-major, then row, then
column, and raster order (top-to-bottom, left-to-right) over the patch
grid; that convention is this module's own and is stated here rather
than attributed.

E is not trained here.  It is drawn from the shared deterministic normal
stream so the Python and R arms hold identical numbers; see _vitcore.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult

__all__ = ["vit_patch_embed"]


def vit_patch_embed(image, patch_size, embed_dim, w_scale=1.0, skip=0):
    """Split an image into P-by-P patches and project them to D dimensions.

    Parameters
    ----------
    image : array-like
        H-by-W matrix (C = 1), or a list of C such matrices.
    patch_size : int
        P.  Must divide both H and W.
    embed_dim : int
        D, the constant latent vector size.
    w_scale : float
        Scales the deterministic projection E.  w_scale = 0 gives E = 0,
        the degenerate case used as an anchor.
    skip : int
        Offset into the shared deterministic stream.

    Returns
    -------
    estimate    : N = HW/P^2, the effective input sequence length
    patches     : x_p, N-by-(P^2 C)
    embeddings  : x_p E, N-by-D
    projection  : E, (P^2 C)-by-D
    n_patches, patch_dim, embed_dim, grid_rows, grid_cols, n_channels
    """
    ch = vc.channels(image)
    nc = len(ch)
    h = len(ch[0])
    w = len(ch[0][0])
    p = int(patch_size)
    d = int(embed_dim)
    if p < 1:
        raise ValueError("vit_patch_embed: patch_size must be a positive integer")
    if d < 1:
        raise ValueError("vit_patch_embed: embed_dim must be a positive integer")
    if h % p != 0 or w % p != 0:
        raise ValueError("vit_patch_embed: patch_size must divide both H and W")
    gr = h // p
    gc = w // p
    n = gr * gc
    pdim = p * p * nc
    patches = []
    for pr in range(gr):
        for pc in range(gc):
            row = []
            for c in range(nc):
                for r in range(p):
                    for s in range(p):
                        row.append(ch[c][pr * p + r][pc * p + s])
            patches.append(row)
    E = vc.draw(pdim, d, skip, w_scale)
    emb = core.matmul(patches, E)
    return RichResult(
        title="ViT patch embedding",
        summary_lines=[("patches", n), ("patch dim", pdim), ("embed dim", d)],
        payload={
            "estimate": float(n),
            "patches": patches,
            "embeddings": emb,
            "projection": E,
            "n_patches": n,
            "patch_dim": pdim,
            "embed_dim": d,
            "grid_rows": gr,
            "grid_cols": gc,
            "n_channels": nc,
            "n": n,
            "skip_used": int(skip) + pdim * d,
            "method": "x_p in R^{N x (P^2 C)}, N = HW/P^2, then x_p E (Dosovitskiy et al. 2021, Sec. 3.1 p. 3 and Eq. (1) p. 4)",
        },
    )


def cheatsheet():
    return "vitptm: ViT patch embedding via the flattened-patch linear projection"


# compact alias per ledger/NAMING.md
vitpatchembed = vit_patch_embed
