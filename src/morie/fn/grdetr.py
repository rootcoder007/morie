# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DETR set-prediction loss: Hungarian matching + classification + bounding box."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_detr_hungarian_matching"]

_METHOD = "DETR Hungarian matching and set-prediction loss"


def _linear_sum_assignment(cost):
    """Minimum-cost assignment (Jonker-Volgenant shortest augmenting path).

    Returns ``(rows, cols)`` for the optimal assignment of the smaller
    side onto the larger. Pure numpy, O(n^2 m).
    """
    C = np.asarray(cost, dtype=float)
    if C.ndim != 2 or C.size == 0:
        raise ValueError(f"cost must be a non-empty 2-D matrix, got shape {C.shape}.")
    if not np.all(np.isfinite(C)):
        raise ValueError("cost matrix contains non-finite values.")
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)   # p[j] = row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    order = np.argsort(rows)
    rows = np.asarray(rows, dtype=int)[order]
    cols = np.asarray(cols, dtype=int)[order]
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols


def _to_xyxy(B, fmt):
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if B.ndim != 2 or B.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {B.shape}.")
    if not np.all(np.isfinite(B)):
        raise ValueError("boxes contain non-finite values.")
    if fmt == "xyxy":
        out = B.copy()
    elif fmt == "cxcywh":
        cx, cy, w, h = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
        if np.any(w < 0) or np.any(h < 0):
            raise ValueError("cxcywh boxes must have non-negative width and height.")
        out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    else:
        raise ValueError(f"box_format must be 'xyxy' or 'cxcywh', got {fmt!r}.")
    if np.any(out[:, 2] < out[:, 0]) or np.any(out[:, 3] < out[:, 1]):
        raise ValueError("boxes have x2 < x1 or y2 < y1 after conversion.")
    return out


def _giou(A, B):
    """Generalised IoU between every row of A and every row of B."""
    ax1, ay1, ax2, ay2 = (A[:, k][:, None] for k in range(4))
    bx1, by1, bx2, by2 = (B[:, k][None, :] for k in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    cw = np.maximum(ax2, bx2) - np.minimum(ax1, bx1)
    ch = np.maximum(ay2, by2) - np.minimum(ay1, by1)
    carea = cw * ch
    return np.where(carea > 0, iou - (carea - union) / np.where(carea > 0, carea, 1.0), iou)


def _log_softmax_rows(Z):
    M = Z.max(axis=1, keepdims=True)
    Z = Z - M
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def geron_detr_hungarian_matching(pred_boxes, pred_classes, gt_boxes, gt_classes,
                                  lam_bbox=5.0, lam_giou=2.0, box_format="xyxy",
                                  no_object_class=None, eos_coef=0.1,
                                  class_is_logits=True):
    r"""Match predictions to ground truth one-to-one, then score the match.

    The pairwise cost is DETR's

    .. math::
        \mathcal{C}(i, j) = -\hat p_i(c_j)
        + \lambda_{\text{bbox}}\|b_i - b_j\|_1
        - \lambda_{\text{giou}}\,\mathrm{GIoU}(b_i, b_j)

    and the assignment minimising its total is found exactly by the
    Hungarian algorithm.  Doing this instead of a per-anchor heuristic is
    what removes the need for NMS: a one-to-one matching means duplicate
    boxes are never both rewarded, so the model stops producing them.

    The reported loss is, over matched pairs,

    .. math::
        L = \sum \bigl[\mathrm{CE}(\hat p_i, c_j)
        + \lambda_{\text{bbox}}\|b_i - b_j\|_1
        + \lambda_{\text{giou}}(1 - \mathrm{GIoU})\bigr],

    plus, when ``no_object_class`` is given, ``eos_coef`` times the
    no-object cross-entropy on the unmatched queries.

    Parameters
    ----------
    pred_boxes : array-like, shape (N, 4)
    pred_classes : array-like, shape (N, C)
        Logits (default) or probabilities -- see ``class_is_logits``.
    gt_boxes : array-like, shape (M, 4)
    gt_classes : array-like, shape (M,)
        Ground-truth class indices.
    lam_bbox, lam_giou : float, optional
        Non-negative cost weights (DETR uses 5 and 2).
    box_format : {"xyxy", "cxcywh"}, optional
    no_object_class : int, optional
        Class index of the "no object" slot.
    eos_coef : float, optional
        Down-weight on the no-object term, DETR's 0.1.
    class_is_logits : bool, optional
        Whether ``pred_classes`` needs a softmax.

    Returns
    -------
    RichResult
        Payload keys ``matching`` (list of ``(pred, gt)``),
        ``cost_matrix``, ``total_cost``, ``loss``, ``loss_class``,
        ``loss_bbox``, ``loss_giou``, ``loss_no_object``,
        ``matched_giou``, ``unmatched_predictions``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 16, DETR section.

    Examples
    --------
    Two queries, one object.  The query whose box coincides with the
    ground truth wins the match:

    >>> pb = [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]]
    >>> pc = [[10.0, 0.0], [0.0, 10.0]]
    >>> r = geron_detr_hungarian_matching(pb, pc, [[0.0, 0.0, 1.0, 1.0]], [0])
    >>> r["matching"]
    [(0, 0)]
    >>> r["loss_bbox"]
    0.0
    >>> round(r["matched_giou"][0], 6)
    1.0

    Swapping the class logits does not move the box, but it does raise
    the classification loss:

    >>> r2 = geron_detr_hungarian_matching(pb, [[0.0, 10.0], [10.0, 0.0]],
    ...                                    [[0.0, 0.0, 1.0, 1.0]], [0])
    >>> r2["matching"]
    [(0, 0)]
    >>> round(r2["loss_class"], 4)
    10.0
    """
    P = _to_xyxy(pred_boxes, box_format)
    G = _to_xyxy(gt_boxes, box_format)
    N, M = P.shape[0], G.shape[0]
    if N == 0 or M == 0:
        raise ValueError(f"need at least one prediction and one ground-truth box, got {N} and {M}.")
    Z = np.atleast_2d(np.asarray(pred_classes, dtype=float))
    if Z.shape[0] != N:
        raise ValueError(f"pred_classes has {Z.shape[0]} rows but there are {N} predicted boxes.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("pred_classes contains non-finite values.")
    C = Z.shape[1]
    gt = np.asarray(gt_classes).ravel().astype(int)
    if gt.size != M:
        raise ValueError(f"gt_classes has {gt.size} entries but there are {M} ground-truth boxes.")
    if gt.min() < 0 or gt.max() >= C:
        raise ValueError(f"gt_classes must lie in [0, {C - 1}].")
    lam_bbox = float(lam_bbox)
    lam_giou = float(lam_giou)
    if lam_bbox < 0 or lam_giou < 0:
        raise ValueError(f"lam_bbox and lam_giou must be non-negative, got {lam_bbox}, {lam_giou}.")
    if N < M:
        raise ValueError(
            f"DETR needs at least as many queries as objects; got {N} predictions "
            f"for {M} ground-truth boxes."
        )

    if class_is_logits:
        logp = _log_softmax_rows(Z)
    else:
        if np.any(Z < 0):
            raise ValueError("probabilities must be non-negative; pass class_is_logits=True for logits.")
        if not np.allclose(Z.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("probability rows must sum to 1; pass class_is_logits=True for logits.")
        with np.errstate(divide="ignore"):
            logp = np.log(np.maximum(Z, 1e-300))
    prob = np.exp(logp)

    l1 = np.sum(np.abs(P[:, None, :] - G[None, :, :]), axis=2)     # (N, M)
    giou = _giou(P, G)                                             # (N, M)
    cost = -prob[:, gt] + lam_bbox * l1 - lam_giou * giou

    rows, cols = _linear_sum_assignment(cost)
    matching = [(int(i), int(j)) for i, j in zip(rows, cols)]
    total_cost = float(cost[rows, cols].sum())

    loss_cls = float(-logp[rows, gt[cols]].sum())
    loss_bbox = float(lam_bbox * l1[rows, cols].sum())
    loss_giou = float(lam_giou * (1.0 - giou[rows, cols]).sum())

    unmatched = sorted(set(range(N)) - set(int(i) for i in rows))
    loss_noobj = 0.0
    if no_object_class is not None:
        noc = int(no_object_class)
        if not (0 <= noc < C):
            raise ValueError(f"no_object_class must lie in [0, {C - 1}], got {noc}.")
        eos_coef = float(eos_coef)
        if eos_coef < 0:
            raise ValueError(f"eos_coef must be non-negative, got {eos_coef}.")
        if unmatched:
            loss_noobj = float(-eos_coef * logp[unmatched, noc].sum())

    loss = loss_cls + loss_bbox + loss_giou + loss_noobj

    return RichResult(
        title="DETR Hungarian matching",
        summary_lines=[("Matched pairs", len(matching)), ("Loss", loss),
                       ("Assignment cost", total_cost)],
        payload={
            "matching": matching,
            "cost_matrix": cost.tolist(),
            "total_cost": total_cost,
            "loss": loss,
            "loss_class": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_no_object": loss_noobj,
            "matched_giou": giou[rows, cols].tolist(),
            "matched_l1": l1[rows, cols].tolist(),
            "unmatched_predictions": unmatched,
            "estimate": loss,
            "n": int(M),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdetr: DETR -- Hungarian one-to-one matching on (class, L1, GIoU) cost, then set loss"
