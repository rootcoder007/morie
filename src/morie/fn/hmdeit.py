# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-efficient Image Transformer (DeiT) with distillation token."""

from . import _array_core as np

from ._richresult import RichResult
from .grdeit import geron_deit_distillation_loss
from .hmdctr import block_params

__all__ = ["geron_deit"]


def geron_deit(
    image,
    patch_size=16,
    n_layers=12,
    teacher=None,
    d_model=384,
    n_heads=6,
    n_classes=1000,
    in_channels=3,
    logits_cls=None,
    logits_dist=None,
    y=None,
    alpha=0.5,
):
    """
    Data-efficient Image Transformer (DeiT) with distillation token.

    Formula: ViT + distillation token mimicking CNN teacher

    Two things, both real. The architecture is resolved against the
    concrete image in the ``hmalex`` manner: the patch grid, the token
    count and every parameter block are computed exactly, with block
    costs DELEGATED to :func:`morie.fn.hmdctr.block_params`. And when
    logits are supplied, the training objective is DELEGATED to
    :func:`morie.fn.grdeit.geron_deit_distillation_loss`.

    The distillation token is the entire architectural change from ViT,
    and it is a small one: one extra learned embedding of width
    ``d_model``, taking the sequence from ``n_patches + 1`` to
    ``n_patches + 2`` tokens, plus a second classifier head. That is
    reported as ``distillation_overhead``. Its output is trained against
    the *teacher's argmax* -- hard distillation -- which is what makes
    DeiT trainable on ImageNet alone, without the extra data ViT needed.

    The image side must be divisible by ``patch_size``; a remainder means
    patches would silently drop pixels at the edge.

    Parameters
    ----------
    image : array-like, shape (H, W) or (C, H, W)
    patch_size : int, default 16
    n_layers : int, default 12
    teacher : array-like or callable, optional
        Teacher logits per sample, or a callable applied to ``image``.
    d_model, n_heads, n_classes, in_channels : int
    logits_cls, logits_dist : array-like, shape (B, C), optional
        Student head logits; both required to compute the loss.
    y : array-like, optional
        Ground-truth labels.
    alpha : float, default 0.5

    Returns
    -------
    result : RichResult
        Keys: n_patches, n_tokens, total_params, patch_embed_params,
        block_params, distillation_overhead, loss, loss_cls, loss_dist,
        teacher_agreement, estimate, n, method.

    Examples
    --------
    A 32x32 image with 16x16 patches gives a 2x2 grid, so 4 patches plus
    the class and distillation tokens:

    >>> import numpy as np
    >>> img = np.zeros((3, 32, 32))
    >>> r = geron_deit(img, patch_size=16, n_layers=1, d_model=8, n_heads=2, n_classes=4)
    >>> r["n_patches"], r["n_tokens"]
    (4, 6)
    >>> r["patch_embed_params"]
    6152

    The distillation token costs one embedding plus one head:

    >>> r["distillation_overhead"]
    44

    With logits, the objective is computed: uniform logits on both heads
    give ``log 2`` per head:

    >>> import math
    >>> r2 = geron_deit(img, patch_size=16, n_layers=1, d_model=8, n_heads=2,
    ...                 n_classes=2, logits_cls=[[0.0, 0.0]], logits_dist=[[0.0, 0.0]],
    ...                 y=[0], teacher=[[1.0, 0.0]])
    >>> round(r2["loss"], 9) == round(math.log(2), 9)
    True
    >>> r2["teacher_agreement"]
    1.0

    A patch size that does not tile the image is rejected:

    >>> geron_deit(np.zeros((3, 30, 30)), patch_size=16)
    Traceback (most recent call last):
      ...
    ValueError: geron_deit: image side 30 is not divisible by patch_size 16

    References
    ----------
    Géron Ch 16
    """
    X = np.asarray(image, dtype=float)
    if X.ndim == 2:
        X = X[None, :, :]
    if X.ndim != 3 or X.size == 0:
        raise ValueError(f"geron_deit: image must be (H, W) or (C, H, W), got shape {X.shape}")
    C_in, H, W = X.shape
    P = int(patch_size)
    if P < 1:
        raise ValueError(f"geron_deit: patch_size must be >= 1, got {patch_size!r}")
    if H % P or W % P:
        raise ValueError(f"geron_deit: image side {H if H % P else W} is not divisible by patch_size {P}")
    L, d, Hh, K = int(n_layers), int(d_model), int(n_heads), int(n_classes)
    if L < 1 or d < 1 or Hh < 1 or K < 2:
        raise ValueError("geron_deit: n_layers, d_model, n_heads must be >= 1 and n_classes >= 2")
    if d % Hh:
        raise ValueError(f"geron_deit: d_model={d} is not divisible by n_heads={Hh}")

    grid_h, grid_w = H // P, W // P
    n_patches = int(grid_h * grid_w)
    n_tokens = n_patches + 2  # class token + distillation token
    patch_embed = int(P * P * C_in * d + d)
    pos = int(n_tokens * d)
    tokens = int(2 * d)
    per = block_params(d, cross_attention=False)
    heads = int(2 * (d * K + K))
    total = int(patch_embed + pos + tokens + L * per["total"] + 2 * d + heads)
    overhead = int(d + (d * K + K))  # the distillation token and its head

    loss = lcls = ldist = agree = None
    if logits_cls is not None or logits_dist is not None:
        if logits_cls is None or logits_dist is None or y is None or teacher is None:
            raise ValueError(
                "geron_deit: computing the loss needs logits_cls, logits_dist, y and teacher together"
            )
        t = teacher(X) if callable(teacher) else teacher
        base = geron_deit_distillation_loss(logits_cls, logits_dist, y, t, alpha=alpha)
        loss = float(base["loss"])
        lcls = float(base["loss_cls"])
        ldist = float(base["loss_dist"])
        agree = float(base["teacher_agreement"])

    return RichResult(
        title="DeiT",
        summary_lines=[("Patches", n_patches), ("Tokens", n_tokens), ("Parameters", total)],
        interpretation="The distillation token is a tiny architectural change; the data efficiency comes from the hard teacher labels.",
        payload={
            "n_patches": n_patches,
            "n_tokens": int(n_tokens),
            "patch_grid": (int(grid_h), int(grid_w)),
            "total_params": total,
            "patch_embed_params": patch_embed,
            "position_params": pos,
            "block_params": int(per["total"]),
            "head_params": heads,
            "distillation_overhead": overhead,
            "d_head": int(d // Hh),
            "loss": loss,
            "loss_cls": lcls,
            "loss_dist": ldist,
            "teacher_agreement": agree,
            "alpha": float(alpha),
            "estimate": float(total) if loss is None else loss,
            "n": int(n_tokens),
            "method": "DeiT architecture resolved concretely; blocks via hmdctr, distillation loss via grdeit",
        },
    )


def cheatsheet():
    return "hmdeit: Data-efficient Image Transformer (DeiT) with distillation token"


# compact alias per ledger/NAMING.md
gerondeit = geron_deit
