# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DETR: CNN-transformer hybrid for end-to-end object detection."""

import numpy as np

from ._richresult import RichResult
from .grdetr import geron_detr_hungarian_matching
from .hmdctr import block_params

__all__ = ["geron_detr"]


def geron_detr(
    image,
    n_queries=100,
    n_layers=6,
    d_model=256,
    n_heads=8,
    n_classes=91,
    backbone_stride=32,
    pred_boxes=None,
    pred_classes=None,
    gt_boxes=None,
    gt_classes=None,
):
    """
    DETR: CNN-transformer hybrid for end-to-end object detection.

    Formula: CNN features -> transformer encoder -> object queries ->
    bipartite matching

    The pipeline is resolved against a concrete image in the ``hmalex``
    manner: the backbone reduces the image by ``backbone_stride``, the
    resulting ``H/32 x W/32`` grid is flattened into that many encoder
    tokens, and the decoder attends from exactly ``n_queries`` learned
    embeddings. Transformer block costs are DELEGATED to
    :func:`morie.fn.hmdctr.block_params`.

    Two consequences follow from those numbers and are reported rather
    than asserted. Encoder self-attention is quadratic in the token count,
    so ``encoder_attention_cost`` grows as ``(HW/32^2)^2`` -- the reason
    DETR is slow on large images. And the number of queries is a hard
    ceiling: an image with more objects than queries cannot be fully
    detected, which ``max_detections`` states.

    When predictions and ground truth are supplied, the set-prediction
    loss and the bipartite matching are DELEGATED to
    :func:`morie.fn.grdetr.geron_detr_hungarian_matching` -- the piece
    that removes NMS by making every ground-truth box the responsibility
    of exactly one query.

    Parameters
    ----------
    image : array-like, shape (H, W) or (C, H, W)
    n_queries : int, default 100
    n_layers : int, default 6
    d_model, n_heads, n_classes : int
    backbone_stride : int, default 32
    pred_boxes, pred_classes, gt_boxes, gt_classes : array-like, optional
        Supply all four to run the matching and loss.

    Returns
    -------
    result : RichResult
        Keys: feature_shape, n_tokens, n_queries, total_params,
        encoder_attention_cost, max_detections, matching, loss,
        loss_bbox, loss_class, estimate, n, method.

    Examples
    --------
    A 224x224 image reduces to a 7x7 grid, i.e. 49 encoder tokens:

    >>> import numpy as np
    >>> r = geron_detr(np.zeros((3, 224, 224)), n_queries=10, n_layers=1,
    ...                d_model=8, n_heads=2, n_classes=3)
    >>> r["feature_shape"], r["n_tokens"]
    ((7, 7), 49)
    >>> r["max_detections"]
    10
    >>> r["encoder_attention_cost"]
    2401

    The matching is bipartite: the query whose box coincides with the
    ground truth wins, and the other query is left unmatched.

    >>> pb = [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]]
    >>> pc = [[10.0, 0.0], [0.0, 10.0]]
    >>> r2 = geron_detr(np.zeros((3, 224, 224)), n_queries=2, n_layers=1, d_model=8,
    ...                 n_heads=2, n_classes=2, pred_boxes=pb, pred_classes=pc,
    ...                 gt_boxes=[[0.0, 0.0, 1.0, 1.0]], gt_classes=[0])
    >>> r2["matching"]
    [(0, 0)]
    >>> r2["loss_bbox"]
    0.0

    More ground-truth objects than queries cannot be matched, and that is
    an error rather than a silent truncation:

    >>> geron_detr(np.zeros((3, 224, 224)), n_queries=1, pred_boxes=pb, pred_classes=pc,
    ...            gt_boxes=[[0.0, 0.0, 1.0, 1.0]], gt_classes=[0])
    Traceback (most recent call last):
      ...
    ValueError: geron_detr: 2 predictions supplied but n_queries is 1

    References
    ----------
    Géron Ch 16
    """
    X = np.asarray(image, dtype=float)
    if X.ndim == 2:
        X = X[None, :, :]
    if X.ndim != 3 or X.size == 0:
        raise ValueError(f"geron_detr: image must be (H, W) or (C, H, W), got shape {X.shape}")
    C_in, H, W = X.shape
    st = int(backbone_stride)
    if st < 1:
        raise ValueError(f"geron_detr: backbone_stride must be >= 1, got {backbone_stride!r}")
    Q, L, d, Hh, K = int(n_queries), int(n_layers), int(d_model), int(n_heads), int(n_classes)
    if Q < 1 or L < 1 or d < 1 or Hh < 1 or K < 1:
        raise ValueError("geron_detr: n_queries, n_layers, d_model, n_heads and n_classes must be >= 1")
    if d % Hh:
        raise ValueError(f"geron_detr: d_model={d} is not divisible by n_heads={Hh}")

    fh, fw = H // st, W // st
    if fh < 1 or fw < 1:
        raise ValueError(f"geron_detr: image {H}x{W} is smaller than the backbone stride {st}")
    tokens = int(fh * fw)

    enc = block_params(d, cross_attention=False)
    dec = block_params(d, cross_attention=True)
    proj = 2048 * d + d  # 1x1 conv from a ResNet-50 C5 map
    queries = Q * d
    heads = (d * (K + 1) + (K + 1)) + (3 * (d * d) + 3 * d + d * 4 + 4)  # class head + 3-layer box MLP
    total = int(proj + queries + L * enc["total"] + L * dec["total"] + heads)

    match = loss = lbox = lcls = None
    if any(v is not None for v in (pred_boxes, pred_classes, gt_boxes, gt_classes)):
        if any(v is None for v in (pred_boxes, pred_classes, gt_boxes, gt_classes)):
            raise ValueError("geron_detr: matching needs pred_boxes, pred_classes, gt_boxes and gt_classes together")
        P = np.atleast_2d(np.asarray(pred_boxes, dtype=float))
        if P.shape[0] > Q:
            raise ValueError(f"geron_detr: {P.shape[0]} predictions supplied but n_queries is {Q}")
        G = np.atleast_2d(np.asarray(gt_boxes, dtype=float))
        if G.shape[0] > Q:
            raise ValueError(f"geron_detr: {G.shape[0]} ground-truth objects exceed the {Q} available queries")
        base = geron_detr_hungarian_matching(pred_boxes, pred_classes, gt_boxes, gt_classes)
        match = list(base["matching"])
        loss = float(base["loss"])
        lbox = float(base["loss_bbox"])
        lcls = float(base["loss_class"])

    return RichResult(
        title="DETR",
        summary_lines=[("Feature grid", (fh, fw)), ("Queries", Q), ("Parameters", total)],
        interpretation="Bipartite matching makes each object the responsibility of one query, which is what removes NMS.",
        payload={
            "feature_shape": (int(fh), int(fw)),
            "n_tokens": tokens,
            "n_queries": Q,
            "total_params": total,
            "encoder_params": int(L * enc["total"]),
            "decoder_params": int(L * dec["total"]),
            "projection_params": int(proj),
            "query_params": int(queries),
            "head_params": int(heads),
            "encoder_attention_cost": int(tokens * tokens),
            "max_detections": Q,
            "matching": match,
            "loss": loss,
            "loss_bbox": lbox,
            "loss_class": lcls,
            "estimate": float(total) if loss is None else loss,
            "n": int(tokens),
            "method": "DETR pipeline resolved concretely; blocks via hmdctr, set matching via grdetr",
        },
    )


def cheatsheet():
    return "hmdetr: DETR: CNN-transformer hybrid for end-to-end object detection"
