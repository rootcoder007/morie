# morie.fn -- slice s04 (rootcoder007/morie)
"""Joint loss for multi-output DNN with mixed outcome types.

Book sections read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer.  Volume [Pages 477-532], Chapter 12, Section 12.4,
pp. 490-493: the multi-trait network is compiled with a per-trait loss
and per-trait loss_weights, "we need to specify a loss function and
metrics for each trait (outcome), that need to be in agreement with the
type of response of each trait", and the joint objective is the weighted
sum of them.  Section 12.4.5, p. 504, is the mixed-outcome case.  The
chapter also prescribes how to set the weights when the traits are on
different scales: "(1) first we calculated the median of each trait,
(2) then we calculated the 0.25 and 0.75 quantiles for each trait,
(3) then we calculated the maximum distance in terms of absolute value
between the median and both quantiles, (4) then we used as the weight
for the first trait its calculated distance, and (5) then we used as
weight for the second trait the value obtained by dividing the distance
of the first trait by the distance of the second trait", which is what
weights=None reproduces.

Volume [Pages 427-476], Chapter 11, Section 11.1.3, p. 428, fixes which
loss goes with which outcome: sum of squares for continuous, logistic
(cross-entropy) for binary, categorical cross-entropy for categorical
and ordinal, Poisson (or negative binomial) for counts.  Volume [Pages
379-425], Section 10.7.2, p. 401, writes the Poisson loss as
L = sum_ij (yhat_ij - y_ij log yhat_ij).
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["joint_loss_mixed_outcomes"]

_TYPES = ("cont", "binary", "count", "ordinal")


def _loss(kind, y, yh):
    n = len(y)
    s = 0.0
    if kind == "cont":
        for i in range(n):
            d = y[i] - yh[i]
            s += d * d
        return s / n
    if kind == "binary":
        for i in range(n):
            if y[i] != 0.0 and y[i] != 1.0:
                raise ValueError("joint_loss_mixed_outcomes: binary targets must be 0 or 1")
            p = min(max(yh[i], 1e-15), 1.0 - 1e-15)
            s -= y[i] * math.log(p) + (1.0 - y[i]) * math.log(1.0 - p)
        return s / n
    if kind == "count":
        for i in range(n):
            if yh[i] <= 0.0:
                raise ValueError("joint_loss_mixed_outcomes: count predictions must be positive")
            s += yh[i] - y[i] * math.log(yh[i])
        return s / n
    if kind == "ordinal":
        for i in range(n):
            p = min(max(yh[i], 1e-15), 1.0)
            s -= y[i] * math.log(p)
        return s / n
    raise ValueError("joint_loss_mixed_outcomes: unknown outcome type %r" % (kind,))


def joint_loss_mixed_outcomes(y_dict, y_hat_dict, weights=None):
    """Weighted sum of per-outcome losses, one term per trait.

    Parameters
    ----------
    y_dict : mapping
        outcome name -> (kind, observed vector), kind one of
        "cont", "binary", "count", "ordinal".
    y_hat_dict : mapping
        outcome name -> predicted vector.
    weights : mapping or sequence, optional
        The loss_weights.  When absent the Section 12.4 recipe is used.

    Returns
    -------
    estimate : the joint loss
    loss     : the same value
    parts    : the per-outcome losses
    weights  : the weights actually used
    """
    if not y_dict:
        raise ValueError("joint_loss_mixed_outcomes: no outcomes supplied")
    names = list(y_dict.keys())
    kinds = {}
    ys = {}
    for nm in names:
        kd, v = y_dict[nm]
        if kd not in _TYPES:
            raise ValueError("joint_loss_mixed_outcomes: unknown outcome type %r" % (kd,))
        kinds[nm] = kd
        ys[nm] = core.vec(v)
        if nm not in y_hat_dict:
            raise ValueError("joint_loss_mixed_outcomes: no prediction for outcome %r" % (nm,))
        if len(core.vec(y_hat_dict[nm])) != len(ys[nm]):
            raise ValueError("joint_loss_mixed_outcomes: outcome %r has mismatched lengths" % (nm,))
        if not ys[nm]:
            raise ValueError("joint_loss_mixed_outcomes: outcome %r is empty" % (nm,))
    if weights is None:
        # Section 12.4 recipe: max |median - quartile| per trait, then the
        # first trait's distance divided by each subsequent trait's.
        d = []
        for nm in names:
            med = core.median(ys[nm])
            q1 = core.quantile7(ys[nm], 0.25)
            q3 = core.quantile7(ys[nm], 0.75)
            d.append(max(abs(med - q1), abs(med - q3)))
        w = [d[0]]
        for j in range(1, len(names)):
            w.append(d[0] / d[j] if d[j] > 0.0 else 0.0)
        wt = dict(zip(names, w))
    elif isinstance(weights, dict):
        wt = {nm: float(weights[nm]) for nm in names}
    else:
        wv = core.vec(weights)
        if len(wv) != len(names):
            raise ValueError("joint_loss_mixed_outcomes: one weight per outcome is required")
        wt = dict(zip(names, wv))
    parts = {}
    tot = 0.0
    for nm in names:
        L = _loss(kinds[nm], ys[nm], core.vec(y_hat_dict[nm]))
        parts[nm] = L
        tot += wt[nm] * L
    return RichResult(
        title="Joint loss, mixed outcomes",
        summary_lines=[("outcomes", len(names))],
        payload={
            "estimate": tot,
            "loss": tot,
            "parts": parts,
            "weights": wt,
            "n": len(names),
            "method": "L = sum_t w_t L_t with the Chapter 11 Sect. 11.1.3 loss per outcome type, weights per Chapter 12 Sect. 12.4",
        },
    )


def cheatsheet():
    return "jntlO: Joint loss for multi-output DNN with mixed outcome types"
