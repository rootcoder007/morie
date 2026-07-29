# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log-loss (cross-entropy) cost for binary logistic regression."""

import numpy as np

from ._richresult import RichResult
from .hmlogp import geron_logistic_probability

__all__ = ["geron_logistic_cost"]

_METHOD = "Binary log loss (cross-entropy)"


def geron_logistic_cost(X, y, theta, add_bias=False):
    """
    Log-loss (cross-entropy) cost for binary logistic regression.

    Formula: J = -(1/m) sum_i [y_i log p_hat_i + (1-y_i) log(1-p_hat_i)]

    The probabilities are delegated to
    :func:`morie.fn.hmlogp.geron_logistic_probability`; only the loss is
    computed here.  It is evaluated in the numerically stable
    ``softplus`` form

    ``-y*z + log(1 + exp(z)) = -y*z + max(z,0) + log1p(exp(-|z|))``

    directly from the logits ``z``, so a confident wrong prediction
    costs a large finite number instead of ``log(0) = -inf``.  Clipping
    the probabilities instead would silently cap the penalty, which is
    the usual way this cost gets quietly wrong.

    The baseline cost of always predicting the class rate is returned
    for comparison: a model that cannot beat it has learned nothing.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix.
    y : array-like, shape (m,)
        Labels in {0, 1}.
    theta : array-like, shape (n,)
        Coefficients.
    add_bias : bool
        Prepend a ones column to ``X``.

    Returns
    -------
    result : RichResult
        Keys: cost, per_instance, p_hat, baseline_cost, estimate, n, method.

    Examples
    --------
    With ``theta = 0`` every probability is 0.5, so the cost is
    ``-log(0.5) = log 2`` whatever the labels are:

    >>> r = geron_logistic_cost([[1.0], [1.0]], [0, 1], [0.0])
    >>> round(r["cost"], 9)
    0.693147181

    A hand log-sum: logits 2 and -2 with labels 1 and 0 both cost
    ``log(1 + e^-2) = 0.126928011``:

    >>> r2 = geron_logistic_cost([[2.0], [-2.0]], [1, 0], [1.0])
    >>> round(r2["cost"], 9)
    0.126928011

    A confident wrong answer is finite, not infinite:

    >>> bool(np.isfinite(geron_logistic_cost([[1.0]], [0], [900.0])["cost"]))
    True
    >>> round(geron_logistic_cost([[1.0]], [0], [900.0])["cost"], 1)
    900.0

    References
    ----------
    Géron Ch 4
    """
    yy = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if yy.size == 0:
        raise ValueError("geron_logistic_cost: y is empty")
    if not np.all(np.isin(yy, (0.0, 1.0))):
        raise ValueError(
            f"geron_logistic_cost: y must contain only 0 and 1, got distinct values {np.unique(yy).tolist()}"
        )

    inner = geron_logistic_probability(X, theta, add_bias=add_bias)
    p = inner["p_hat"]
    z = inner["logits"]
    if p.size != yy.size:
        raise ValueError(f"geron_logistic_cost: X has {p.size} rows but y has {yy.size} entries")

    # Stable per-instance loss straight from the logits.
    per = -yy * z + np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    cost = float(np.mean(per))

    rate = float(np.mean(yy))
    if 0.0 < rate < 1.0:
        baseline = float(-(rate * np.log(rate) + (1 - rate) * np.log1p(-rate)))
    else:
        baseline = 0.0

    return RichResult(
        title="Logistic log loss",
        summary_lines=[("Cost", cost), ("Baseline (class rate)", baseline), ("Instances", int(yy.size))],
        interpretation=(
            "Convex in theta with a single global minimum, so gradient descent cannot get stuck; "
            "beating the baseline is the minimum bar."
        ),
        payload={
            "cost": cost,
            "per_instance": per,
            "p_hat": p,
            "logits": z,
            "baseline_cost": baseline,
            "estimate": cost,
            "n": int(yy.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlogcl: binary log loss computed stably from logits; probabilities delegated to hmlogp"
