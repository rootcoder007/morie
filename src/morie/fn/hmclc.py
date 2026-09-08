# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classification + localization: predict class and bounding box."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_classification_localization"]


def geron_classification_localization(image, model, n_classes=None, gt_class=None, gt_box=None, alpha=1.0):
    """
    Classification + localization: predict class and bounding box.

    Formula: [p_1,...,p_K, x, y, w, h] output

    The two-headed output in the formula line is parsed and scored for
    real. ``model`` is a callable returning one vector per image: ``K``
    class scores followed by exactly four box numbers ``(x, y, w, h)``
    in centre-size form. The contract is enforced -- a vector that is not
    ``K + 4`` long, or a non-positive width or height, raises rather than
    producing a nonsense box.

    The joint loss is the one the architecture implies: cross-entropy on
    the class head plus ``alpha`` times the MSE on the box head. Those
    two are not commensurable -- one is in nats, the other in squared
    pixels -- which is precisely why ``alpha`` exists and why both terms
    are reported separately as well as combined.

    Localisation quality is measured by IoU, computed from the corner
    form ``[x - w/2, y - h/2, x + w/2, y + h/2]``. IoU is the metric that
    matters at evaluation time and the MSE is what is actually
    differentiated, so both are returned; a box can have small MSE and
    poor IoU when the object is small.

    Parameters
    ----------
    image : array-like
        Input; passed to ``model``.
    model : callable
        ``model(image) -> array of length K + 4`` (or ``(B, K + 4)``).
    n_classes : int, optional
        ``K``; inferred as ``len(output) - 4`` if omitted.
    gt_class : array-like of int, optional
        True labels; enables the classification loss.
    gt_box : array-like, optional
        True boxes in ``(x, y, w, h)``; enables the box loss and IoU.
    alpha : float, default 1.0
        Weight on the localisation term.

    Returns
    -------
    result : RichResult
        Keys: class_probs, predicted_class, box, box_corners, iou, loss,
        loss_class, loss_box, n_classes, estimate, n, method.

    Examples
    --------
    Two classes and a box: the class head is softmaxed, the box head is
    read off directly.

    >>> import numpy as np
    >>> model = lambda img: np.array([0.0, 0.0, 5.0, 5.0, 2.0, 4.0])
    >>> r = geron_classification_localization(None, model)
    >>> r["class_probs"][0]
    [0.5, 0.5]
    >>> r["box"][0]
    [5.0, 5.0, 2.0, 4.0]
    >>> r["box_corners"][0]
    [4.0, 3.0, 6.0, 7.0]

    A perfect box scores IoU 1 and no localisation loss; the uniform
    class head still costs ``log 2``:

    >>> import math
    >>> r2 = geron_classification_localization(None, model, gt_class=[0],
    ...                                        gt_box=[[5.0, 5.0, 2.0, 4.0]])
    >>> round(r2["iou"][0], 12)
    1.0
    >>> r2["loss_box"]
    0.0
    >>> round(r2["loss"], 9) == round(math.log(2), 9)
    True

    A box shifted by its own width has no overlap at all:

    >>> r3 = geron_classification_localization(None, model, gt_class=[0],
    ...                                        gt_box=[[9.0, 5.0, 2.0, 4.0]])
    >>> r3["iou"][0]
    0.0
    >>> r3["loss_box"]
    16.0

    A degenerate box is an error, not a silent zero-area prediction:

    >>> bad = lambda img: np.array([0.0, 0.0, 5.0, 5.0, 0.0, 4.0])
    >>> geron_classification_localization(None, bad)
    Traceback (most recent call last):
      ...
    ValueError: geron_classification_localization: box 0 has non-positive width or height (0.0, 4.0)

    References
    ----------
    Géron Ch 12
    """
    if not callable(model):
        raise ValueError("geron_classification_localization: model must be a callable model(image) -> [p..., x, y, w, h]")
    out = np.asarray(model(image), dtype=float)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2 or out.size == 0:
        raise ValueError(f"geron_classification_localization: model must return a vector or (B, K+4) array, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("geron_classification_localization: model returned non-finite values")
    B, width = out.shape
    K = width - 4 if n_classes is None else int(n_classes)
    if K < 1:
        raise ValueError(
            f"geron_classification_localization: the output of width {width} leaves {K} class scores; "
            "it must contain at least one class plus the four box numbers"
        )
    if width != K + 4:
        raise ValueError(
            f"geron_classification_localization: model returned width {width} but n_classes={K} implies {K + 4}"
        )

    scores = out[:, :K]
    box = out[:, K:]
    for i in range(B):
        if box[i, 2] <= 0 or box[i, 3] <= 0:
            raise ValueError(
                f"geron_classification_localization: box {i} has non-positive width or height "
                f"({box[i, 2]}, {box[i, 3]})"
            )

    shift = scores - scores.max(axis=1, keepdims=True)
    logp = shift - np.log(np.exp(shift).sum(axis=1, keepdims=True))
    P = np.exp(logp)

    def corners(b):
        return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)

    C = corners(box)

    iou = loss_cls = loss_box = total = None
    if gt_box is not None:
        G = np.atleast_2d(np.asarray(gt_box, dtype=float))
        if G.shape != box.shape:
            raise ValueError(f"geron_classification_localization: gt_box must have shape {box.shape}, got {G.shape}")
        if np.any(G[:, 2:] <= 0):
            raise ValueError("geron_classification_localization: a ground-truth box has non-positive width or height")
        GC = corners(G)
        x1 = np.maximum(C[:, 0], GC[:, 0])
        y1 = np.maximum(C[:, 1], GC[:, 1])
        x2 = np.minimum(C[:, 2], GC[:, 2])
        y2 = np.minimum(C[:, 3], GC[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_p = box[:, 2] * box[:, 3]
        area_g = G[:, 2] * G[:, 3]
        iou = (inter / (area_p + area_g - inter)).tolist()
        loss_box = float(np.mean(np.sum((box - G) ** 2, axis=1)))
    if gt_class is not None:
        y = np.asarray(gt_class).ravel().astype(int)
        if y.size != B:
            raise ValueError(f"geron_classification_localization: gt_class has {y.size} entries but there are {B} images")
        if y.min() < 0 or y.max() >= K:
            raise ValueError(f"geron_classification_localization: a label lies outside 0..{K - 1}")
        loss_cls = float(-np.mean(logp[np.arange(B), y]))
    if loss_cls is not None or loss_box is not None:
        total = (loss_cls or 0.0) + float(alpha) * (loss_box or 0.0)

    return RichResult(
        title="Classification + localization",
        summary_lines=[("Images", B), ("Classes", K), ("Loss", total)],
        interpretation="One head is scored in nats and the other in squared pixels; alpha is what makes them comparable.",
        payload={
            "class_probs": P.tolist(),
            "log_probs": logp.tolist(),
            "predicted_class": P.argmax(axis=1).astype(int).tolist(),
            "box": box.tolist(),
            "box_corners": C.tolist(),
            "iou": iou,
            "loss": total,
            "loss_class": loss_cls,
            "loss_box": loss_box,
            "alpha": float(alpha),
            "n_classes": int(K),
            "estimate": float(P.max()) if total is None else total,
            "n": int(B),
            "method": "two-headed classification+localization output with cross-entropy, box MSE and IoU",
        },
    )


def cheatsheet():
    return "hmclc: Classification + localization: predict class and bounding box"
