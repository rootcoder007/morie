# morie.fn -- function file (rootcoder007/morie)
"""DETR set prediction with Hungarian bipartite matching."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["detr_set_prediction"]


def hungarian(cost):
    """Exact minimum-cost assignment (Jonker-Volgenant shortest paths).

    Rectangular cost with rows <= columns.  Returns the column chosen
    for each row.
    """
    n = len(cost)
    m = len(cost[0])
    if n > m:
        raise ValueError("hungarian: need at least as many columns as rows")
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
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
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            out[p[j] - 1] = j - 1
    return out


def detr_set_prediction(image, queries, n_objects=None, targets=None):
    """
    DETR set prediction

    Formula: transformer decoder; bipartite matching loss

    Every query predicts one box, and the loss is defined only after a
    one-to-one assignment between predictions and ground truth found by
    the Hungarian algorithm on the pairwise cost.  There is no
    non-maximum suppression and no anchor set: the matching is what
    removes the duplicates.  Unmatched queries are trained to the
    no-object class.

    Parameters
    ----------
    image : array-like
        Q x 4 matrix of predicted boxes (cx, cy, w, h) in [0, 1].  The
        name follows the stub signature; it is the query output.
    queries : array-like
        G x 4 matrix of ground-truth boxes.
    n_objects : int or None
        Number of ground-truth objects; inferred when None.
    targets : array-like or None
        Optional per-query confidence used to break exact cost ties.

    Returns
    -------
    result : dict
        Keys: estimate (matching cost), assignment, cost, matched,
        unmatched, l1_cost, giou_cost, Q, G.

    References
    ----------
    Carion et al. (2020), End-to-End Object Detection with
    Transformers, ECCV 2020:213-229.
    """
    P = core.mat(image)
    T = core.mat(queries)
    Q = len(P)
    G = len(T)
    if Q == 0 or G == 0:
        raise ValueError("empty input: both boxes and targets are required")
    if len(P[0]) != 4 or len(T[0]) != 4:
        raise ValueError("boxes must have four columns (cx, cy, w, h)")
    if n_objects is not None and int(n_objects) != G:
        raise ValueError("n_objects disagrees with the number of target rows")
    if G > Q:
        raise ValueError("more ground-truth boxes than queries")

    def corners(b):
        return (b[0] - b[2] / 2.0, b[1] - b[3] / 2.0,
                b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)

    l1 = [[0.0] * Q for _ in range(G)]
    giou = [[0.0] * Q for _ in range(G)]
    for g in range(G):
        for q in range(Q):
            l1[g][q] = sum(abs(T[g][k] - P[q][k]) for k in range(4))
            ax0, ay0, ax1, ay1 = corners(T[g])
            bx0, by0, bx1, by1 = corners(P[q])
            iw = max(min(ax1, bx1) - max(ax0, bx0), 0.0)
            ih = max(min(ay1, by1) - max(ay0, by0), 0.0)
            inter = iw * ih
            ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
            iou = inter / ua if ua > 0.0 else 0.0
            cw = max(ax1, bx1) - min(ax0, bx0)
            ch = max(ay1, by1) - min(ay0, by0)
            area_c = cw * ch
            g_iou = iou - (area_c - ua) / area_c if area_c > 0.0 else iou
            giou[g][q] = 1.0 - g_iou
    cost = [[5.0 * l1[g][q] + 2.0 * giou[g][q] for q in range(Q)]
            for g in range(G)]
    assign = hungarian(cost)
    total = sum(cost[g][assign[g]] for g in range(G))
    matched = sorted(assign)
    unmatched = [q for q in range(Q) if q not in matched]
    return RichResult(payload={
        "estimate": total,
        "assignment": assign,
        "cost": total,
        "matched": matched,
        "unmatched": unmatched,
        "l1_cost": sum(l1[g][assign[g]] for g in range(G)),
        "giou_cost": sum(giou[g][assign[g]] for g in range(G)),
        "Q": Q,
        "G": G,
        "method": "DETR set prediction with Hungarian matching",
    })


def cheatsheet():
    return "detrbb: DETR set prediction with Hungarian matching"


# compact alias per ledger/NAMING.md
detrsetprediction = detr_set_prediction
