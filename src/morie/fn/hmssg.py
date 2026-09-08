# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semantic segmentation: per-pixel class labels."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_semantic_segmentation"]


def geron_semantic_segmentation(image, model, y_true=None):
    """
    Semantic segmentation: per-pixel class labels.

    Formula: y[i,j] in {1..K} for each pixel

    Orchestration around a caller-supplied segmentation head, with the
    contract enforced: ``model(image)`` must return per-pixel scores of
    shape (H, W, K) matching the image's spatial size. The label map is
    the per-pixel argmax; when ground truth is supplied the evaluation is
    the real one used for segmentation -- per-class intersection over
    union ``IoU_k = |P_k & G_k| / |P_k | G_k|``, its mean over the classes
    that actually occur, plus pixel accuracy and a confusion matrix.

    Parameters
    ----------
    image : array-like
        (H, W) or (H, W, C) input, passed to `model` unchanged.
    model : callable
        ``model(image) -> (H, W, K)`` scores or log-probabilities.
    y_true : array-like, optional
        (H, W) integer ground-truth labels in ``0..K-1``.

    Returns
    -------
    result : RichResult
        Keys: labels, scores, class_counts, iou, mean_iou, pixel_accuracy,
        confusion, estimate, n, method.

    Examples
    --------
    A model that splits the image down the middle, scored against a
    ground truth that is right about the left half only:

    >>> import numpy as np
    >>> img = [[0.0, 0.0], [0.0, 0.0]]
    >>> def m(x):
    ...     s = np.zeros((2, 2, 2))
    ...     s[:, 0, 0] = 1.0
    ...     s[:, 1, 1] = 1.0
    ...     return s
    >>> r = geron_semantic_segmentation(img, m, y_true=[[0, 0], [0, 0]])
    >>> r["labels"].tolist()
    [[0, 1], [0, 1]]
    >>> round(float(r["pixel_accuracy"]), 6)
    0.5
    >>> round(float(r["iou"][0]), 6)
    0.5

    References
    ----------
    Géron Ch 12
    """
    img = np.asarray(image, dtype=float)
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("geron_semantic_segmentation: image must be a non-empty (H, W) or (H, W, C) array")
    if not callable(model):
        raise ValueError("geron_semantic_segmentation: model must be a callable mapping the image to (H, W, K) scores")
    H, W = img.shape[0], img.shape[1]

    scores = np.asarray(model(image), dtype=float)
    if scores.ndim != 3:
        raise ValueError(
            f"geron_semantic_segmentation: model returned a {scores.ndim}-D array; (H, W, K) scores are required"
        )
    if scores.shape[0] != H or scores.shape[1] != W:
        raise ValueError(
            f"geron_semantic_segmentation: model returned a {scores.shape[0]}x{scores.shape[1]} map "
            f"but the image is {H}x{W}; segmentation output must stay registered with the input"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("geron_semantic_segmentation: model returned non-finite scores")
    K = scores.shape[2]
    if K < 2:
        raise ValueError(f"geron_semantic_segmentation: need at least 2 classes, model returned K={K}")

    labels = np.argmax(scores, axis=2).astype(int)
    counts = np.bincount(labels.ravel(), minlength=K)

    iou = None
    mean_iou = None
    acc = None
    conf = None
    if y_true is not None:
        G = np.asarray(y_true)
        if G.shape != (H, W):
            raise ValueError(f"geron_semantic_segmentation: y_true has shape {G.shape} but the image is {H}x{W}")
        G = G.astype(int)
        if G.min() < 0 or G.max() >= K:
            raise ValueError(
                f"geron_semantic_segmentation: y_true labels must lie in 0..{K - 1}, got {G.min()}..{G.max()}"
            )
        conf = np.zeros((K, K), dtype=int)
        for g, p in zip(G.ravel(), labels.ravel()):
            conf[g, p] += 1
        inter = np.diag(conf).astype(float)
        union = conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf)
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), np.nan)
        present = union > 0
        mean_iou = float(np.mean(iou[present]))
        acc = float(np.sum(np.diag(conf)) / conf.sum())

    return RichResult(
        title="Semantic segmentation",
        summary_lines=[
            ("Pixels", int(H * W)),
            ("Classes", int(K)),
            ("Mean IoU", mean_iou if mean_iou is not None else "n/a (no ground truth)"),
        ],
        interpretation=(
            "Pixel accuracy flatters segmenters on imbalanced scenes; mean IoU over the classes that "
            "actually occur is the honest summary."
        ),
        payload={
            "labels": labels,
            "scores": scores,
            "class_counts": counts,
            "iou": iou,
            "mean_iou": mean_iou,
            "pixel_accuracy": acc,
            "confusion": conf,
            "n_classes": int(K),
            "estimate": float(mean_iou) if mean_iou is not None else float(np.max(counts) / (H * W)),
            "n": int(H * W),
            "method": "Per-pixel argmax with per-class IoU / pixel accuracy against ground truth",
        },
    )


def cheatsheet():
    return "hmssg: Semantic segmentation: per-pixel class labels"
