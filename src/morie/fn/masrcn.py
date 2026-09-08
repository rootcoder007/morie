# morie.fn -- function file (rootcoder007/morie)
r"""Mask R-CNN: instance segmentation by adding a mask branch.

Faster R-CNN detects; FCN segments semantically. Mask R-CNN adds, to
Faster R-CNN, a third branch predicting a binary mask for each region
of interest, in parallel with classification and box regression. The
extension is conceptually simple; two details make it work.

**RoIAlign, because RoIPool quantises.** RoIPool rounds the region's
continuous coordinates to the feature-map grid, twice -- once for the
region boundary and once for the bins. For a box regression that is
tolerable; for pixel-to-pixel mask prediction it is a misalignment of
several pixels at stride 16 or 32. RoIAlign removes both roundings and
samples with **bilinear interpolation** at exact locations. The paper
describes it as a seemingly minor change with a large impact, and the
anchor measures the misalignment RoIPool introduces rather than
repeating the claim.

**Decoupling mask and class.** The mask branch predicts :math:`K`
binary masks, one per class, and applies a **per-pixel sigmoid** with
the loss taken only on the mask of the ground-truth class. The
alternative -- a per-pixel softmax over classes -- makes the masks
compete, so a pixel assigned to one class is evidence against another.
Decoupling them is what lets the classification branch decide the
category and the mask branch decide only the shape, and the paper
reports the softmax variant is substantially worse.

**The multi-task loss** is :math:`L = L_{cls} + L_{box} + L_{mask}`,
summed rather than traded off with weights, which is possible precisely
*because* the mask loss is decoupled from the class.

References
----------
He, K., Gkioxari, G., Dollar, P. & Girshick, R. (2017) "Mask R-CNN",
*Proceedings of the IEEE International Conference on Computer Vision
(ICCV 2017)*, 2980-2988, doi:10.1109/ICCV.2017.322,
arXiv:1703.06870. Sec. 1 and 3 (extending Faster R-CNN with a mask
branch in parallel with the existing classification and box branches;
that Faster R-CNN was not designed for pixel-to-pixel alignment
between inputs and outputs, most evident in RoIPool's coarse spatial
quantization; the quantization-free RoIAlign layer that faithfully
preserves exact spatial locations, described as a seemingly minor
change with a large impact; the multi-task loss
L = L_cls + L_box + L_mask; and the per-class binary masks with a
per-pixel sigmoid, decoupling mask and class prediction, against the
per-pixel softmax which makes classes compete).

Ren, S., He, K., Girshick, R. & Sun, J. (2015) "Faster R-CNN: Towards
Real-Time Object Detection with Region Proposal Networks", *NeurIPS
2015*, 91-99, arXiv:1506.01497. The detector being extended.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["roi_pool", "roi_align", "alignment_error", "mask_loss",
           "multitask_loss"]

_EPS = 1e-12


def _bilinear(F, y, x):
    h, w = len(F), len(F[0])
    y = min(max(float(y), 0.0), h - 1.0)
    x = min(max(float(x), 0.0), w - 1.0)
    y0, x0 = int(math.floor(y)), int(math.floor(x))
    y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
    dy, dx = y - y0, x - x0
    return (F[y0][x0] * (1 - dy) * (1 - dx)
            + F[y1][x0] * dy * (1 - dx)
            + F[y0][x1] * (1 - dy) * dx
            + F[y1][x1] * dy * dx)


def roi_pool(features, box, out_size=2, stride=1.0):
    r"""RoIPool: quantise the box and the bins, then max-pool.

    Both roundings are the point -- they are what RoIAlign removes.
    """
    F = [[float(v) for v in r] for r in k.mat(features)]
    y0, x0, y1, x1 = [float(v) / float(stride) for v in box]
    qy0, qx0 = int(math.floor(y0)), int(math.floor(x0))
    qy1, qx1 = int(math.floor(y1)), int(math.floor(x1))
    if qy1 <= qy0 or qx1 <= qx0:
        raise ValueError("masrcn: the box collapsed under "
                         "quantisation, which is itself the problem")
    n = int(out_size)
    bh = (qy1 - qy0) / float(n)
    bw = (qx1 - qx0) / float(n)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            a0 = qy0 + int(math.floor(i * bh))
            a1 = max(a0 + 1, qy0 + int(math.floor((i + 1) * bh)))
            b0 = qx0 + int(math.floor(j * bw))
            b1 = max(b0 + 1, qx0 + int(math.floor((j + 1) * bw)))
            vals = [F[a][b] for a in range(a0, min(a1, len(F)))
                    for b in range(b0, min(b1, len(F[0])))]
            row.append(max(vals) if vals else 0.0)
        out.append(row)
    return {"pooled": out, "quantised_box": (qy0, qx0, qy1, qx1),
            "quantisation_shift": (y0 - qy0, x0 - qx0),
            "caveat": "the box AND the bins are rounded to the "
                      "feature grid"}


def roi_align(features, box, out_size=2, stride=1.0, samples=2):
    r"""RoIAlign: no rounding, bilinear sampling at exact locations."""
    F = [[float(v) for v in r] for r in k.mat(features)]
    y0, x0, y1, x1 = [float(v) / float(stride) for v in box]
    if y1 <= y0 or x1 <= x0:
        raise ValueError("masrcn: the box has non-positive extent")
    n, s = int(out_size), int(samples)
    bh, bw = (y1 - y0) / n, (x1 - x0) / n
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            acc = []
            for a in range(s):
                for b in range(s):
                    yy = y0 + bh * (i + (a + 0.5) / s)
                    xx = x0 + bw * (j + (b + 0.5) / s)
                    acc.append(_bilinear(F, yy, xx))
            row.append(sum(acc) / len(acc))
        out.append(row)
    return {"pooled": out, "exact_box": (y0, x0, y1, x1),
            "samples_per_bin": s * s,
            "note": "no quantisation of the box or the bins"}


def alignment_error(features, box, out_size=2, stride=1.0):
    r"""How far RoIPool's grid sits from the true box.

    The shift is in *feature* pixels, so it multiplies by the stride
    in the input image -- which is why it matters for masks and not
    for boxes.
    """
    p = roi_pool(features, box, out_size, stride)
    dy, dx = p["quantisation_shift"]
    return {"feature_shift": (dy, dx),
            "input_pixel_shift": (dy * float(stride),
                                  dx * float(stride)),
            "stride": float(stride),
            "note": "a sub-pixel error on the feature map is a "
                    "several-pixel error in the image at stride 16 "
                    "or 32"}


def mask_loss(logits, target, decoupled=True):
    r"""Per-pixel sigmoid on the ground-truth class's mask.

    ``decoupled=False`` applies a softmax over classes instead, which
    makes them compete -- the variant the paper reports as
    substantially worse.
    """
    L = [[float(v) for v in r] for r in k.mat(logits)]
    T = [[float(v) for v in r] for r in k.mat(target)]
    if len(L) != len(T) or len(L[0]) != len(T[0]):
        raise ValueError("masrcn: the logits and target differ in "
                         "shape")
    tot, m = 0.0, 0
    if decoupled:
        for i in range(len(L)):
            for j in range(len(L[0])):
                p = 1.0 / (1.0 + math.exp(-L[i][j])) \
                    if L[i][j] > -700 else 0.0
                p = min(max(p, _EPS), 1.0 - _EPS)
                tot += -(T[i][j] * math.log(p)
                         + (1 - T[i][j]) * math.log(1 - p))
                m += 1
        return {"loss": tot / m, "kind": "per-pixel sigmoid",
                "note": "classes do not compete; the class branch "
                        "decides the category"}
    flat = [L[i][j] for i in range(len(L)) for j in range(len(L[0]))]
    mx = max(flat)
    z = sum(math.exp(v - mx) for v in flat)
    for i in range(len(L)):
        for j in range(len(L[0])):
            p = math.exp(L[i][j] - mx) / z
            tot += -T[i][j] * math.log(max(p, _EPS))
            m += 1
    return {"loss": tot / m, "kind": "per-pixel softmax",
            "caveat": "classes COMPETE, so a pixel assigned to one is "
                      "evidence against another"}


def multitask_loss(l_cls, l_box, l_mask):
    r""":math:`L = L_{cls} + L_{box} + L_{mask}`, summed unweighted.

    Possible only because the mask loss is decoupled from the class.
    """
    return {"total": float(l_cls) + float(l_box) + float(l_mask),
            "cls": float(l_cls), "box": float(l_box),
            "mask": float(l_mask),
            "note": "an unweighted sum, which the decoupling permits"}


def cheatsheet():
    return ("masrcn: Faster R-CNN plus a THIRD branch predicting a "
            "binary mask per RoI. Two details carry it. RoIPool "
            "QUANTISES twice -- box and bins -- which is fine for a "
            "box and a several-pixel misalignment for a mask at stride "
            "16 or 32; RoIAlign removes both roundings and samples "
            "bilinearly. And the mask is DECOUPLED from the class: K "
            "binary masks with a per-pixel SIGMOID, loss on the "
            "ground-truth class only, because a per-pixel softmax "
            "makes classes compete. The decoupling is what lets the "
            "losses simply add.")


# compact alias per ledger/NAMING.md
maskrcnn = roi_align

# public names resolved by fn/_lazy_map.json
mask_rcnn_segmentation = roi_align
