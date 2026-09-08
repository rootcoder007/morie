# morie.fn -- function file (rootcoder007/morie)
r"""YOLOX: anchor-free, decoupled head, dynamic label assignment.

Three advances had arrived in detection and had not reached the YOLO
family: anchor-free prediction, decoupled heads, and advanced label
assignment. YOLOX integrates all three.

**The coupled head is a measured harm, not a style preference.** One
branch predicting classification and localisation together is a
known conflict; the paper's own experiments show replacing the YOLO
head with a **lite decoupled head** -- a :math:`1\times1` convolution
to reduce channels, then two parallel branches of :math:`3\times3`
convolutions -- improves AP, at a cost of 1.1 ms (11.6 vs 10.5 ms).
That trade is stated so it can be judged.

**Anchor-free removes tuned priors.** Instead of matching to
pre-clustered boxes, each location predicts four offsets
:math:`(l, t, r, b)` from itself, decoded with the feature stride.
``decode_box`` inverts ``encode_box`` exactly, which is the property
that makes the parametrisation usable at all.

**Center sampling fixes a starvation problem.** Assigning only the
single center location as positive discards high-quality predictions
whose gradients would help and worsens the positive/negative
imbalance; the center :math:`3\times3` area is assigned positive
instead.

**SimOTA: dynamic top-k instead of optimal transport.** Label
assignment is posed globally as an optimal-transport problem in OTA,
which costs 25% extra training time -- for 300 epochs, expensive. YOLOX
approximates it: compute a cost per (ground truth, prediction) pair,
give each ground truth a **dynamic** :math:`k` from the sum of its top
IoUs, and take its :math:`k` cheapest predictions. No Sinkhorn-Knopp,
no extra solver hyperparameters, and 45.0 to 47.3 AP.

References
----------
Ge, Z., Liu, S., Wang, F., Li, Z. & Sun, J. (2021) "YOLOX: Exceeding
YOLO Series in 2021", arXiv:2107.08430. Sec. 2: the switch to an
anchor-free manner together with a decoupled head and the leading
label assignment strategy SimOTA; that the conflict between
classification and regression tasks is well known and the coupled
detection head may harm performance, with the lite decoupled head
built from a 1x1 convolution reducing channels followed by two
parallel branches of 3x3 convolutions and adding 1.1 ms (11.6 vs 10.5
ms); that the anchor-free version selects only ONE positive (the
center) per object and ignores other high-quality predictions, fixed
by assigning the center 3x3 area as positives ("center sampling");
and that OTA formulates assignment as an Optimal Transport problem but
costs 25% extra training time, so it is simplified to a dynamic top-k
strategy, SimOTA, which avoids the Sinkhorn-Knopp solver's
hyperparameters and raises AP from 45.0% to 47.3%.

Ge, Z., Liu, S., Li, Z., Yoshie, O. & Sun, J. (2021) "OTA: Optimal
Transport Assignment for Object Detection", *CVPR 2021*, 303-312,
arXiv:2103.14259. The assignment being approximated.

Tian, Z., Shen, C., Chen, H. & He, T. (2019) "FCOS: Fully
Convolutional One-Stage Object Detection", *ICCV 2019*, 9627-9636,
arXiv:1904.01355. Center sampling and the anchor-free parametrisation.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["decoupled_head", "encode_box", "decode_box",
           "center_sampling", "simota_assign", "box_iou"]

_EPS = 1e-12


def decoupled_head(channels, reduced=256, n_classes=80):
    r"""1x1 to reduce, then TWO parallel 3x3 branches.

    Classification and localisation stop sharing a final layer, which
    is the conflict the coupled head suffers from.
    """
    c, r = int(channels), int(reduced)
    if c < 1 or r < 1:
        raise ValueError("yolovx: the channel counts must be "
                         "positive")
    n = int(n_classes)
    reduce_p = c * r
    cls_p = r * r * 9 + r * n
    reg_p = r * r * 9 + r * (4 + 1)
    coupled_p = c * (n + 5) * 9
    return {"reduce_params": reduce_p, "cls_params": cls_p,
            "reg_params": reg_p,
            "total": reduce_p + cls_p + reg_p,
            "coupled_total": coupled_p,
            "branches": ["classification", "regression+objectness"],
            "extra_latency_ms": 1.1,
            "note": "measured cost 11.6 ms against 10.5 ms coupled, "
                    "for a stated AP gain"}


def encode_box(box, cx, cy, stride=1.0):
    r"""Four distances from a location to the box sides."""
    x0, y0, x1, y1 = [float(v) for v in box]
    s = float(stride)
    if s <= 0.0:
        raise ValueError("yolovx: the stride must be positive")
    px, py = (float(cx) + 0.5) * s, (float(cy) + 0.5) * s
    if not (x0 <= px <= x1 and y0 <= py <= y1):
        raise ValueError("yolovx: the location is outside the box, "
                         "so it cannot be a positive sample")
    return {"ltrb": [(px - x0) / s, (py - y0) / s,
                     (x1 - px) / s, (y1 - py) / s],
            "center": (px, py), "stride": s}


def decode_box(ltrb, cx, cy, stride=1.0):
    r"""Back to corners. Inverts :func:`encode_box` exactly."""
    l, t, r, b = [float(v) for v in k.vec(ltrb)]
    s = float(stride)
    px, py = (float(cx) + 0.5) * s, (float(cy) + 0.5) * s
    if l < 0 or t < 0 or r < 0 or b < 0:
        raise ValueError("yolovx: the distances cannot be negative")
    return [px - l * s, py - t * s, px + r * s, py + b * s]


def box_iou(a, b):
    r"""Intersection over union of two corner boxes."""
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > _EPS else 0.0


def center_sampling(box, grid_w, grid_h, stride=1.0, radius=1.5):
    r"""The center 3x3 area is positive, not only the center cell.

    One positive per object throws away high-quality predictions whose
    gradients help, and worsens the positive/negative imbalance.
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    s = float(stride)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    inside, center = [], []
    for j in range(int(grid_h)):
        for i in range(int(grid_w)):
            px, py = (i + 0.5) * s, (j + 0.5) * s
            if x0 <= px <= x1 and y0 <= py <= y1:
                inside.append((i, j))
            if (abs(px - cx) <= float(radius) * s
                    and abs(py - cy) <= float(radius) * s):
                center.append((i, j))
    cand = sorted(set(inside) | set(center))
    return {"in_box": inside, "in_center": center,
            "candidates": cand, "n_candidates": len(cand),
            "single_center": 1,
            "note": "one positive per object starves the model of "
                    "useful gradients"}


def simota_assign(costs, ious, top_q=10, max_k=None):
    r"""Dynamic top-:math:`k`, an approximation to optimal transport.

    :math:`k_g` is the rounded sum of the :math:`q` largest IoUs for
    that ground truth, so a well-covered object gets more positives
    than a poorly covered one -- and there is no Sinkhorn solver and no
    extra hyperparameter.
    """
    C = [[float(v) for v in r] for r in k.mat(costs)]
    I = [[float(v) for v in r] for r in k.mat(ious)]
    G, P = len(C), len(C[0])
    if len(I) != G or len(I[0]) != P:
        raise ValueError("yolovx: the cost and IoU matrices differ "
                         "in shape")
    q = min(int(top_q), P)
    assign, ks = {}, []
    for g in range(G):
        top = sorted(I[g], reverse=True)[:q]
        kg = max(1, int(round(sum(top))))
        if max_k is not None:
            kg = min(kg, int(max_k))
        ks.append(kg)
        order = sorted(range(P), key=lambda p: C[g][p])
        assign[g] = sorted(order[:kg])
    owner = {}
    for g in range(G):
        for p in assign[g]:
            if p in owner:
                a, b = owner[p], g
                owner[p] = a if C[a][p] <= C[b][p] else b
            else:
                owner[p] = g
    final = {g: sorted(p for p in assign[g] if owner[p] == g)
             for g in range(G)}
    return RichResult(payload={
        "estimate": final, "assignment": final, "dynamic_k": ks,
        "n_positives": sum(len(v) for v in final.values()),
        "contested": sum(1 for p in owner
                         if sum(1 for g in range(G)
                                if p in assign[g]) > 1),
        "method": "SimOTA dynamic top-k; Ge et al. (2021)",
        "note": "k is DYNAMIC per ground truth, from the sum of its "
                "top IoUs -- no Sinkhorn, no extra hyperparameter",
    })


def cheatsheet():
    return ("yolovx: fold three advances into YOLO. DECOUPLED HEAD -- "
            "classification and localisation conflict in one branch; "
            "1x1 reduce then two parallel 3x3 branches, +1.1 ms "
            "(11.6 vs 10.5) for an AP gain. ANCHOR-FREE -- each "
            "location predicts (l,t,r,b) with the stride, so "
            "decode inverts encode exactly and no clustered priors are "
            "tuned. CENTER SAMPLING -- one positive per object starves "
            "the model, so the center 3x3 is positive. SIMOTA -- OTA's "
            "optimal transport costs 25% extra training time, so "
            "approximate it with DYNAMIC top-k from the sum of top "
            "IoUs: 45.0 to 47.3 AP with no solver hyperparameters.")


# compact alias per ledger/NAMING.md
yoloxhead = simota_assign

# public names resolved by fn/_lazy_map.json
yolo_decoupled_head = simota_assign
