# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ViT patch embedding (Dosovitskiy et al. 2021; Alammar Ch 9)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_vit_patch_embedding"]


def alammar_vit_patch_embedding(image, patch_size, E, cls_token=None,
                                E_pos=None):
    """z0 = [x_class; Flatten(p_1) E; ...; Flatten(p_N) E] + E_pos.

    The image must tile exactly: a remainder pixel row would silently
    shift every later patch, so it is refused.

    References: Alammar and Grootendorst, Ch 9; Dosovitskiy et al.
    (2021).
    """
    img = np.atleast_2d(np.asarray(image, dtype=float))
    P = int(patch_size)
    E = np.atleast_2d(np.asarray(E, dtype=float))
    if P < 1:
        raise ValueError("patch_size must be positive.")
    H, W = img.shape
    if H % P or W % P:
        raise ValueError(
            f"a {H} x {W} image does not tile into {P} x {P} patches; "
            "a remainder row would silently shift every later patch.")
    if E.shape[0] != P * P:
        raise ValueError(
            f"E must have {P * P} rows to accept a flattened patch; "
            f"got {E.shape[0]}.")
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patches.append(img[i:i + P, j:j + P].ravel())
    Z = np.asarray(patches) @ E
    if cls_token is not None:
        c = np.atleast_1d(np.asarray(cls_token, dtype=float))
        if len(c) != E.shape[1]:
            raise ValueError("cls token width must match E columns.")
        Z = np.vstack([c, Z])
    if E_pos is not None:
        Ep = np.atleast_2d(np.asarray(E_pos, dtype=float))
        if Ep.shape != Z.shape:
            raise ValueError(
                f"E_pos shape {Ep.shape} does not match sequence "
                f"{Z.shape}.")
        Z = Z + Ep
    return RichResult(payload={
        "sequence": [[float(v) for v in r] for r in Z],
        "n_patches": len(patches),
        "estimate": float(Z[0, 0]), "n": Z.shape[0],
        "method": "ViT patch embedding (Dosovitskiy et al. 2021)"})


def cheatsheet():
    return "alvit: flatten exact P x P tiles, project, prepend CLS, add positions"
