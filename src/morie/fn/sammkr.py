# morie.fn -- function file (rootcoder007/morie)
r"""Three masks, one gradient: SAM's answer to ambiguity.

**The failure being fixed is averaging.** With a single output, a
model given an ambiguous prompt minimises its expected loss by
predicting something between the valid masks -- a blur that is nobody's
answer. ``average_of_valid_masks`` produces exactly that, so the
problem is demonstrated rather than asserted.

**Three outputs, because nesting is about three deep.** The paper
finds three sufficient for most cases, and names the structure:
**whole, part, and subpart**. Not a tuned hyperparameter but an
observation about what ambiguity in segmentation looks like.

**Only the minimum-loss mask is trained.** During training the loss is
backpropagated through the *best* of the three predictions alone. That
single choice is what makes the outputs specialise instead of
converging: a mask that is never the best for any prompt receives no
gradient and drifts to a different interpretation. Averaging the three
losses would make all three the same mask again -- which the anchor
checks by arithmetic.

**Ranking needs a separate head.** With three masks and no ground
truth at inference, something must choose, so the model predicts its
own IoU with the true mask for each output. That prediction is a
*learned estimate*, so ``rank_masks`` reports the calibration error
too -- a confidently mis-ranked mask is the failure mode, and hiding
it behind a sorted list would be dishonest.

References
----------
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson,
L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P. &
Girshick, R. (2023) "Segment Anything", *ICCV 2023*, 4015-4026,
arXiv:2304.02643. Sec. 3, "Resolving ambiguity": that with one output
the model will AVERAGE multiple valid masks given an ambiguous
prompt; the modification to predict multiple output masks for a single
prompt; that 3 mask outputs is sufficient for most common cases since
nested masks are often at most three deep (whole, part and subpart);
that during training only the MINIMUM loss over masks is
backpropagated; and that the model predicts a confidence score
(estimated IoU) for each mask so they can be ranked.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["average_of_valid_masks", "iou", "min_loss_over_masks",
           "rank_masks", "whole_part_subpart"]

_EPS = 1e-12
_NESTING = ("whole", "part", "subpart")


def _flat(m):
    M = k.mat(m)
    return [float(v) for r in M for v in r]


def average_of_valid_masks(masks):
    r"""What ONE output is forced to produce: a blur of all of them.

    Not a strawman -- it is the loss-minimising single answer when
    several masks are valid, and it is valid for none of them.
    """
    F = [_flat(m) for m in masks]
    if not F:
        raise ValueError("sammkr: no masks given")
    n = len(F[0])
    if any(len(f) != n for f in F):
        raise ValueError("sammkr: the masks differ in size")
    avg = [sum(f[i] for f in F) / len(F) for i in range(n)]
    frac = sum(1 for v in avg if 0.05 < v < 0.95) / float(n)
    return {"mask": avg, "ambiguous_fraction": frac,
            "n_averaged": len(F),
            "note": "pixels strictly between 0 and 1 belong to no "
                    "single valid interpretation"}


def iou(a, b, threshold=0.5):
    r"""Intersection over union of two binary masks."""
    x = [1.0 if v > threshold else 0.0 for v in _flat(a)]
    y = [1.0 if v > threshold else 0.0 for v in _flat(b)]
    if len(x) != len(y):
        raise ValueError("sammkr: the masks differ in size")
    inter = sum(1 for i in range(len(x)) if x[i] > 0 and y[i] > 0)
    union = sum(1 for i in range(len(x)) if x[i] > 0 or y[i] > 0)
    return inter / float(union) if union else 1.0


def min_loss_over_masks(predictions, target, loss_fn):
    r"""Backpropagate through the BEST mask only.

    This is the mechanism that makes the three outputs specialise. The
    mean loss is returned alongside for comparison: it is the quantity
    that would collapse them into one.
    """
    if not predictions:
        raise ValueError("sammkr: no predictions given")
    losses = [float(loss_fn(p, target)) for p in predictions]
    j = min(range(len(losses)), key=lambda i: losses[i])
    mean = sum(losses) / len(losses)
    return {"loss": losses[j], "index": j, "losses": losses,
            "mean_loss": mean, "gap": mean - losses[j],
            "note": "only output %d receives gradient; the others "
                    "are free to specialise elsewhere" % j}


def whole_part_subpart(masks, target_hierarchy=None):
    r"""Name the three outputs by the nesting they were meant for."""
    if len(masks) != 3:
        raise ValueError("sammkr: the paper's argument is about "
                         "THREE outputs (whole, part, subpart), got "
                         "%d" % len(masks))
    sizes = [sum(1 for v in _flat(m) if v > 0.5) for m in masks]
    order = sorted(range(3), key=lambda i: -sizes[i])
    named = {}
    for rank, i in enumerate(order):
        named[_NESTING[rank]] = i
    nested = all(
        set(i for i, v in enumerate(_flat(masks[order[r + 1]]))
            if v > 0.5)
        <= set(i for i, v in enumerate(_flat(masks[order[r]]))
               if v > 0.5) for r in range(2))
    return {"assignment": named, "sizes": sizes, "nested": nested,
            "note": "nested masks are often at most three deep"}


def rank_masks(masks, predicted_iou, target=None):
    r"""Rank by the model's own IoU estimate, and check it.

    At inference there is no ground truth, so something has to choose;
    reporting the calibration error keeps a confidently wrong ranking
    visible instead of hidden inside a sorted list.
    """
    p = [float(v) for v in k.vec(predicted_iou)]
    if len(p) != len(masks):
        raise ValueError("sammkr: %d masks but %d predicted IoUs"
                         % (len(masks), len(p)))
    order = sorted(range(len(p)), key=lambda i: -p[i])
    out = {"order": order, "best": order[0], "predicted_iou": p}
    if target is not None:
        true = [iou(m, target) for m in masks]
        best_true = max(range(len(true)), key=lambda i: true[i])
        out.update({
            "true_iou": true, "best_true": best_true,
            "correct": order[0] == best_true,
            "calibration_error": sum(abs(p[i] - true[i])
                                     for i in range(len(p))) / len(p),
            "regret": true[best_true] - true[order[0]],
        })
    return RichResult(payload=dict(
        out, estimate=order[0],
        method="multi-mask output with IoU ranking; Kirillov et al. "
               "(2023)",
        note="the score is a LEARNED estimate, so its error is "
             "reported rather than assumed away"))


def cheatsheet():
    return ("sammkr: one output forces the model to AVERAGE the valid "
            "masks of an ambiguous prompt -- a blur that answers "
            "nobody. So predict THREE, because segmentation nesting is "
            "usually at most three deep: whole, part, subpart. During "
            "training backprop only the MINIMUM loss, which is what "
            "makes the three specialise instead of collapsing into one "
            "(the mean would collapse them). At inference there is no "
            "ground truth, so the model predicts its own IoU per mask "
            "to rank them -- a learned estimate, so report its "
            "calibration error.")


# compact alias per ledger/NAMING.md
sammultimask = rank_masks

# public names resolved by fn/_lazy_map.json
sam_multi_mask_rank = rank_masks
