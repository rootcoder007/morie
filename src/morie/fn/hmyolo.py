# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO: single-shot object detection via grid regression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_yolo", "box_iou"]


def box_iou(a, b):
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def geron_yolo(image, model, n_boxes=1, conf_threshold=0.5, iou_threshold=0.45):
    """
    YOLO: single-shot object detection via grid regression.

    Formula: each grid cell predicts (x,y,w,h,conf) and class probabilities

    Decodes a single-shot detection head and runs real non-max
    suppression. ``model(image)`` must return an ``(S, S, B*5 + C)``
    tensor: for each of B boxes per cell, ``(tx, ty, tw, th, conf)`` with
    the centre offsets ``tx, ty`` in [0, 1] *relative to the cell* and
    ``tw, th`` as fractions of the whole image, followed by C class
    probabilities shared by the cell (the YOLOv1 layout). Detection score
    is ``conf * class_prob``; overlapping detections of the same class are
    greedily suppressed at `iou_threshold` using :func:`box_iou`.

    Parameters
    ----------
    image : array-like
        Input image, passed to `model` unchanged.
    model : callable
        ``model(image) -> (S, S, B*5 + C)``.
    n_boxes : int, default 1
        Boxes per cell (B).
    conf_threshold : float, default 0.5
        Minimum ``conf * class_prob`` to keep a detection; in [0, 1].
    iou_threshold : float, default 0.45
        NMS overlap threshold; in [0, 1].

    Returns
    -------
    result : RichResult
        Keys: boxes, scores, classes, n_detections, n_candidates,
        suppressed, estimate, n, method.

    Examples
    --------
    Two detections in opposite corners of a 2x2 grid: they do not overlap,
    so both survive NMS and their coordinates are exact.

    >>> import numpy as np
    >>> def m(x):
    ...     p = np.zeros((2, 2, 7))
    ...     p[0, 0, :5] = [0.5, 0.5, 0.5, 0.5, 1.0]
    ...     p[0, 0, 5] = 1.0
    ...     p[1, 1, :5] = [0.5, 0.5, 0.5, 0.5, 0.8]
    ...     p[1, 1, 6] = 1.0
    ...     return p
    >>> r = geron_yolo(None, m)
    >>> int(r["n_detections"])
    2
    >>> [round(float(v), 6) for v in r["boxes"][0]]
    [0.0, 0.0, 0.5, 0.5]
    >>> [int(c) for c in r["classes"]]
    [0, 1]
    >>> round(float(r["scores"][1]), 6)
    0.8

    Overlapping same-class boxes: NMS keeps the more confident one.

    >>> def m2(x):
    ...     p = np.zeros((2, 2, 6))
    ...     p[0, 0, :5] = [0.5, 0.5, 0.6, 0.6, 0.9]
    ...     p[0, 0, 5] = 1.0
    ...     p[0, 1, :5] = [0.0, 0.5, 0.6, 0.6, 0.7]
    ...     p[0, 1, 5] = 1.0
    ...     return p
    >>> round(box_iou((-0.05, -0.05, 0.55, 0.55), (0.2, -0.05, 0.8, 0.55)), 6)
    0.411765
    >>> r2 = geron_yolo(None, m2, iou_threshold=0.4)
    >>> int(r2["n_candidates"]), int(r2["n_detections"]), int(r2["suppressed"])
    (2, 1, 1)
    >>> round(float(r2["scores"][0]), 6)
    0.9

    References
    ----------
    Géron Ch 12
    """
    if not callable(model):
        raise ValueError("geron_yolo: model must be a callable returning an (S, S, B*5 + C) prediction tensor")
    B = int(n_boxes)
    if B < 1:
        raise ValueError(f"geron_yolo: n_boxes must be >= 1, got {B}")
    ct, it = float(conf_threshold), float(iou_threshold)
    if not (0.0 <= ct <= 1.0):
        raise ValueError(f"geron_yolo: conf_threshold must lie in [0, 1], got {ct}")
    if not (0.0 <= it <= 1.0):
        raise ValueError(f"geron_yolo: iou_threshold must lie in [0, 1], got {it}")

    P = np.asarray(model(image), dtype=float)
    if P.ndim != 3 or P.shape[0] != P.shape[1]:
        raise ValueError(
            f"geron_yolo: model returned shape {P.shape}; a square (S, S, B*5 + C) grid is required"
        )
    if not np.all(np.isfinite(P)):
        raise ValueError("geron_yolo: model returned non-finite predictions")
    S = P.shape[0]
    C = P.shape[2] - 5 * B
    if C < 1:
        raise ValueError(
            f"geron_yolo: last axis has {P.shape[2]} entries, which leaves {C} class slots "
            f"after {B} boxes x 5; need at least one class"
        )

    cand = []
    for i in range(S):
        for j in range(S):
            cls = P[i, j, 5 * B :]
            k = int(np.argmax(cls))
            for b in range(B):
                tx, ty, tw, th, conf = P[i, j, 5 * b : 5 * b + 5]
                score = float(conf * cls[k])
                if score < ct or tw <= 0 or th <= 0:
                    continue
                cx = (j + tx) / S
                cy = (i + ty) / S
                cand.append((score, k, (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)))

    cand.sort(key=lambda z: -z[0])
    keep = []
    for score, k, box in cand:
        if all(not (k == kk and box_iou(box, bb) > it) for _, kk, bb in keep):
            keep.append((score, k, box))

    boxes = np.asarray([b for _, _, b in keep], dtype=float).reshape(-1, 4)
    scores = np.asarray([s for s, _, _ in keep], dtype=float)
    classes = np.asarray([k for _, k, _ in keep], dtype=int)
    order = np.lexsort((-scores, classes)) if scores.size else np.asarray([], dtype=int)
    boxes, scores, classes = boxes[order], scores[order], classes[order]

    return RichResult(
        title="YOLO detections",
        summary_lines=[
            ("Grid", f"{S}x{S}"),
            ("Classes", int(C)),
            ("Candidates", len(cand)),
            ("Detections", len(keep)),
        ],
        interpretation=(
            "One forward pass produces every box, which is why YOLO is fast; the price is that each "
            "cell can commit to only B boxes, so crowded scenes lose objects before NMS even runs."
        ),
        payload={
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "n_detections": int(len(keep)),
            "n_candidates": int(len(cand)),
            "suppressed": int(len(cand) - len(keep)),
            "grid": int(S),
            "n_classes": int(C),
            "estimate": float(scores.max()) if scores.size else 0.0,
            "n": int(S * S * B),
            "method": "Grid decode of (x, y, w, h, conf) + class scores, then greedy per-class NMS",
        },
    )


def cheatsheet():
    return "hmyolo: YOLO: single-shot object detection via grid regression"
