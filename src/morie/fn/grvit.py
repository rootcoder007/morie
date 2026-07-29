# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vision Transformer patch embedding."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_vit_patch_embedding"]

_METHOD = "ViT patch embedding"


def geron_vit_patch_embedding(image, patch_size, E, E_pos=None, cls_token=None):
    r"""Cut an image into patches, flatten, project, prepend CLS, add positions.

    .. math::
        \mathbf{z}_0 = [\,\mathbf{x}_{\text{class}};\;
            \mathbf{x}_p^{1}E;\; \dots;\; \mathbf{x}_p^{N}E\,] + E_{\text{pos}}

    A ViT has no convolution anywhere: the only spatial prior is the
    patch cut itself, and after flattening, position is carried purely by
    :math:`E_{\text{pos}}`.  Patches are taken in row-major order and each
    is flattened row-major over ``(row, col, channel)`` -- the ordering
    has to match whatever produced ``E``, so it is fixed and documented
    rather than inferred.  The image must divide evenly into patches;
    silently cropping the remainder would drop pixels.

    Parameters
    ----------
    image : array-like, shape (H, W) or (H, W, C)
    patch_size : int
        Must divide both H and W.
    E : array-like, shape (patch_size**2 * C, d_model)
    E_pos : array-like, shape (N + 1, d_model), optional
        Position embeddings, including the CLS slot.
    cls_token : array-like, shape (d_model,), optional
        Defaults to zeros.

    Returns
    -------
    RichResult
        Payload keys ``embeddings`` (N+1 x d_model), ``patches``
        (flattened), ``n_patches``, ``d_model``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 16, Vision Transformer (ViT) section.

    Examples
    --------
    A 2x2 image, patch size 1, identity-like projection to width 1:

    >>> img = [[1.0, 2.0], [3.0, 4.0]]
    >>> r = geron_vit_patch_embedding(img, 1, [[1.0]])
    >>> r["n_patches"]
    4
    >>> r["embeddings"]
    [[0.0], [1.0], [2.0], [3.0], [4.0]]

    One 2x2 patch instead, summed by a ones projection: ``1+2+3+4 = 10``.

    >>> r2 = geron_vit_patch_embedding(img, 2, [[1.0], [1.0], [1.0], [1.0]])
    >>> r2["embeddings"][1]
    [10.0]
    """
    A = np.asarray(image, dtype=float)
    if A.ndim == 2:
        A = A[:, :, None]
    if A.ndim != 3 or A.size == 0:
        raise ValueError(f"image must be (H, W) or (H, W, C), got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("image contains non-finite values.")
    p = int(patch_size)
    if p < 1:
        raise ValueError(f"patch_size must be at least 1, got {p}.")
    H, W, C = A.shape
    if H % p or W % p:
        raise ValueError(
            f"patch_size {p} does not divide the {H}x{W} image; cropping would drop pixels."
        )
    Em = np.atleast_2d(np.asarray(E, dtype=float))
    dim = p * p * C
    if Em.shape[0] != dim:
        raise ValueError(
            f"E must have {dim} rows (patch_size^2 * channels), got {Em.shape[0]}."
        )
    d_model = Em.shape[1]

    patches = []
    for i in range(0, H, p):
        for j in range(0, W, p):
            patches.append(A[i : i + p, j : j + p, :].reshape(-1))
    P = np.vstack(patches)
    Z = P @ Em

    if cls_token is None:
        cls = np.zeros(d_model)
    else:
        cls = np.asarray(cls_token, dtype=float).ravel()
        if cls.size != d_model:
            raise ValueError(f"cls_token must have {d_model} entries, got {cls.size}.")
    Z = np.vstack([cls[None, :], Z])

    if E_pos is not None:
        Ep = np.atleast_2d(np.asarray(E_pos, dtype=float))
        if Ep.shape != Z.shape:
            raise ValueError(
                f"E_pos must have shape {Z.shape} (patches + CLS), got {Ep.shape}."
            )
        Z = Z + Ep

    return RichResult(
        title="ViT patch embedding",
        summary_lines=[("Patches", int(P.shape[0])), ("d_model", int(d_model))],
        payload={
            "embeddings": Z.tolist(),
            "patches": P.tolist(),
            "n_patches": int(P.shape[0]),
            "d_model": int(d_model),
            "estimate": Z.tolist(),
            "n": int(Z.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grvit: row-major patches -> flatten -> @E -> prepend CLS -> +E_pos; size must divide exactly"
