# morie.fn -- function file (rootcoder007/morie)
r"""ResNeXt: cardinality as a dimension in its own right.

VGG and ResNet stack blocks of the same shape, which keeps the
hyper-parameter space small and, the paper argues, reduces the risk of
over-adapting to one dataset. Inception blocks are more accurate but
carefully hand-designed per stage -- the split-transform-merge is
powerful, but the filter numbers and sizes are bespoke.

ResNeXt keeps the repeat-the-same-block discipline *and* the
split-transform-merge, by aggregating a set of transformations that all
have the **same topology**:

.. math:: y = x + \sum_{i=1}^{C} \mathcal{T}_i(x).

:math:`C` is the **cardinality**, and the claim is that it is a
dimension of the design space alongside depth and width. Increasing it
improves accuracy *while holding complexity fixed* -- and does so more
effectively than making the network deeper or wider. That is the
comparison to make, and the anchor holds parameter count fixed while
varying :math:`C`.

**Three equivalent forms, and the equivalence is the practical
point.** The block can be drawn as (a) :math:`C` separate paths each
ending in a full-width projection, (b) :math:`C` paths concatenated
then projected once, or (c) a single grouped convolution with
:math:`C` groups. They compute the same function, and (c) is what
makes the design fast on real hardware. ``block_equivalence`` checks
the identity numerically rather than taking the figure's word for it.

**Complexity accounting.** For a bottleneck of width :math:`d` per
path, the parameter count is roughly
:math:`C\cdot(256d + 3\cdot3\cdot d\cdot d + 256d)`; matching a
baseline means trading :math:`C` against :math:`d`, which is exactly
the constraint under which the cardinality claim is tested.

References
----------
Xie, S., Girshick, R., Dollar, P., Tu, Z. & He, K. (2017) "Aggregated
Residual Transformations for Deep Neural Networks", *Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition (CVPR
2017)*, 5987-5995, doi:10.1109/CVPR.2017.634, arXiv:1611.05431. The
abstract and Sec. 1 (repeating a building block that aggregates a set
of transformations with the SAME TOPOLOGY; "cardinality" as the size
of that set, exposed as an essential factor alongside depth and width;
that increasing cardinality improves accuracy even under fixed
complexity, and more effectively than going deeper or wider; the
VGG/ResNet discipline of stacking same-shape modules against
Inception's bespoke per-stage design; and second place in the ILSVRC
2016 classification task). Sec. 3 (the three equivalent formulations
of the block, the third being a grouped convolution).

He, K., Zhang, X., Ren, S. & Sun, J. (2016) "Deep Residual Learning
for Image Recognition", *CVPR 2016*, 770-778,
doi:10.1109/CVPR.2016.90, arXiv:1512.03385. The residual block being
widened.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D.,
Erhan, D., Vanhoucke, V. & Rabinovich, A. (2015) "Going deeper with
convolutions", *CVPR 2015*, 1-9, doi:10.1109/CVPR.2015.7298594. The
split-transform-merge whose hand-design ResNeXt avoids.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["aggregated_block", "grouped_block", "block_equivalence",
           "block_parameters", "match_complexity"]

_EPS = 1e-12


def _lin(W, x):
    return [sum(W[o][j] * x[j] for j in range(len(x)))
            for o in range(len(W))]


def _relu(v):
    return [max(0.0, q) for q in v]


def aggregated_block(x, Wins, Wmids, Wouts):
    r"""Form (a): :math:`y = x + \sum_i \mathcal{T}_i(x)`.

    Each path narrows, transforms, then projects back to full width.
    """
    xv = [float(v) for v in k.vec(x)]
    acc = [0.0] * len(xv)
    for i in range(len(Wins)):
        h = _relu(_lin(Wins[i], xv))
        h = _relu(_lin(Wmids[i], h))
        o = _lin(Wouts[i], h)
        acc = [acc[j] + o[j] for j in range(len(acc))]
    return [xv[j] + acc[j] for j in range(len(xv))]


def grouped_block(x, Wins, Wmids, Wout_concat):
    r"""Form (c): the paths concatenated and projected ONCE.

    Equivalent to form (a) when ``Wout_concat`` is the horizontal
    concatenation of the per-path output matrices -- which is why a
    grouped convolution can implement it.
    """
    xv = [float(v) for v in k.vec(x)]
    cat = []
    for i in range(len(Wins)):
        h = _relu(_lin(Wins[i], xv))
        cat.extend(_relu(_lin(Wmids[i], h)))
    o = _lin(Wout_concat, cat)
    return [xv[j] + o[j] for j in range(len(xv))]


def block_equivalence(x, Wins, Wmids, Wouts, tol=1e-9):
    r"""Check that the two forms compute the same function."""
    a = aggregated_block(x, Wins, Wmids, Wouts)
    cat = [sum((list(W[o]) for W in Wouts), [])
           for o in range(len(Wouts[0]))]
    c = grouped_block(x, Wins, Wmids, cat)
    dev = max(abs(a[j] - c[j]) for j in range(len(a)))
    return {"equivalent": dev < float(tol), "max_deviation": dev,
            "aggregated": a, "grouped": c,
            "note": "same function; the grouped form is what runs "
                    "fast"}


def block_parameters(width, cardinality, bottleneck):
    r"""Parameters in one bottleneck block."""
    W, C, d = int(width), int(cardinality), int(bottleneck)
    if min(W, C, d) < 1:
        raise ValueError("resnxt: width, cardinality and bottleneck "
                         "must all be at least 1")
    return {"parameters": C * (W * d + 9 * d * d + d * W),
            "cardinality": C, "bottleneck": d, "width": W}


def match_complexity(width, cardinality, target_parameters):
    r"""The bottleneck width :math:`d` that matches a budget.

    Trading :math:`C` against :math:`d` at fixed cost is the
    constraint under which the cardinality claim is tested.
    """
    W, C = int(width), int(cardinality)
    T = float(target_parameters)
    a = 9.0 * C
    b = 2.0 * C * W
    disc = b * b + 4.0 * a * T
    d = (-b + math.sqrt(disc)) / (2.0 * a)
    return {"bottleneck": d, "rounded": max(1, int(round(d))),
            "parameters": block_parameters(
                W, C, max(1, int(round(d))))["parameters"],
            "target": T, "cardinality": C}


def cheatsheet():
    return ("resnxt: y = x + sum_{i=1..C} T_i(x), every T_i with the "
            "SAME TOPOLOGY -- Inception's split-transform-merge "
            "without its per-stage hand design. C is CARDINALITY, a "
            "design dimension beside depth and width, and raising it "
            "beats going deeper or wider AT FIXED COMPLEXITY. Three "
            "equivalent block forms: C separate paths, concatenate-"
            "then-project, or one GROUPED CONVOLUTION -- same "
            "function, and the third is what runs fast.")


# compact alias per ledger/NAMING.md
resnext = aggregated_block

# public names resolved by fn/_lazy_map.json
resnext_block = aggregated_block
resnextblock = aggregated_block
