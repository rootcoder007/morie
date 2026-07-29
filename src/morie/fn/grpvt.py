# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pyramid Vision Transformer stage: spatial reduction before attention."""

import numpy as np

from ._richresult import RichResult
from .grsdpa import attend

__all__ = ["geron_pyramid_vit_stage"]

_METHOD = "Pyramid ViT spatial-reduction attention"


def geron_pyramid_vit_stage(X, WQ, WK, WV, reduction_ratio=2):
    r"""Attention with keys and values computed on a downsampled map.

    .. math::
        Q = X W_Q, \qquad K = \mathrm{SR}(X) W_K,
        \qquad V = \mathrm{SR}(X) W_V

    Swin shrinks the *query* side by windowing; PVT keeps full-resolution
    queries and shrinks only the keys and values.  With an ``R x R``
    reduction the attention matrix goes from :math:`HW \times HW` to
    :math:`HW \times HW/R^2`, so every query still sees the whole image,
    just at coarser granularity -- that is the trade PVT makes, and why
    it keeps dense-prediction resolution where Swin has to shift windows.
    ``SR`` here is average pooling over ``R x R`` blocks.

    Parameters
    ----------
    X : array-like, shape (H, W, d_model)
    WQ, WK : array-like, shape (d_model, d_k)
    WV : array-like, shape (d_model, d_v)
    reduction_ratio : int, optional
        ``R``; must divide H and W. ``R = 1`` is plain self-attention.

    Returns
    -------
    RichResult
        Payload keys ``output`` (HW x d_v), ``weights``,
        ``reduced_tokens``, ``compression``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 16, Pyramid Vision Transformer section.

    Examples
    --------
    A 2x2 map reduced by 2 gives a single key/value token, so every query
    attends to the average -- weights are all 1 and every output row is
    the mean, 2.5:

    >>> X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    >>> I = [[1.0]]
    >>> r = geron_pyramid_vit_stage(X, I, I, I, reduction_ratio=2)
    >>> r["reduced_tokens"]
    1
    >>> [row[0] for row in r["output"]]
    [2.5, 2.5, 2.5, 2.5]

    ``R = 1`` leaves all four key tokens:

    >>> geron_pyramid_vit_stage(X, I, I, I, reduction_ratio=1)["reduced_tokens"]
    4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 3 or A.size == 0:
        raise ValueError(f"X must be a non-empty (H, W, d_model) array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")
    H, W, d = A.shape
    R = int(reduction_ratio)
    if R < 1:
        raise ValueError(f"reduction_ratio must be at least 1, got {R}.")
    if H % R or W % R:
        raise ValueError(f"reduction_ratio {R} does not divide the {H}x{W} feature map.")
    mats = []
    for name, Wm in (("WQ", WQ), ("WK", WK), ("WV", WV)):
        Mm = np.atleast_2d(np.asarray(Wm, dtype=float))
        if Mm.shape[0] != d:
            raise ValueError(f"{name} must have {d} rows to match d_model, got {Mm.shape[0]}.")
        mats.append(Mm)
    Wq, Wk, Wv = mats
    if Wq.shape[1] != Wk.shape[1]:
        raise ValueError(f"WQ maps to d_k={Wq.shape[1]} but WK maps to {Wk.shape[1]}.")

    tokens = A.reshape(H * W, d)
    SR = A.reshape(H // R, R, W // R, R, d).mean(axis=(1, 3)).reshape(-1, d)
    out, w = attend(tokens @ Wq, SR @ Wk, SR @ Wv)

    return RichResult(
        title="Pyramid ViT stage",
        summary_lines=[("Query tokens", int(H * W)), ("Key tokens", int(SR.shape[0]))],
        payload={
            "output": out.tolist(),
            "weights": w.tolist(),
            "reduced_tokens": int(SR.shape[0]),
            "compression": float(H * W / SR.shape[0]),
            "reduced_map": SR.tolist(),
            "estimate": out.tolist(),
            "n": int(H * W),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpvt: Q full resolution, K/V from R x R average-pooled map; attention HW x HW/R^2"
