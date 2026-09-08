# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdaBoost sample weight update rule."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_adaboost_weight_update"]

_METHOD = "AdaBoost sample-weight update"


def geron_adaboost_weight_update(y_true, y_pred, weights, alpha_t):
    r"""Re-weight the training set after fitting one AdaBoost stump.

    .. math::
        w_i \leftarrow w_i \exp\!\bigl(\alpha_t\,
        \mathbf{1}\{y_i \neq h_t(x_i)\}\bigr),
        \qquad w \leftarrow w / \textstyle\sum_j w_j

    Misclassified instances are boosted by :math:`e^{\alpha_t}`; correct
    ones keep their weight and then shrink relative to the rest through
    the normalisation.  The weighted error rate before the update is
    reported so the caller can check :math:`\alpha_t` against
    :math:`\log\frac{1-r}{r}`.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels of equal length. Any hashable label
        type works; only equality is used.
    weights : array-like
        Current non-negative sample weights, summing to a positive value.
    alpha_t : float
        Predictor weight of the round just fitted.

    Returns
    -------
    RichResult
        Payload keys ``weights_new``, ``misclassified``,
        ``weighted_error``, ``boost_factor``, ``estimate`` (weighted
        error rate before the update), ``n``, ``method``.

    References
    ----------
    Géron Ch 6, AdaBoost section.

    Examples
    --------
    >>> import numpy as np
    >>> r = geron_adaboost_weight_update([0, 0, 1, 1], [0, 1, 1, 1],
    ...                                  [0.25] * 4, np.log(3.0))
    >>> [round(w, 6) for w in r["weights_new"]]
    [0.166667, 0.5, 0.166667, 0.166667]
    >>> round(r["weighted_error"], 6)
    0.25
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError(
            f"y_true and y_pred must have equal length, got {y_true.size} and {y_pred.size}."
        )
    if y_true.size != w.size:
        raise ValueError(
            f"weights must have one entry per sample ({y_true.size}), got {w.size}."
        )
    if y_true.size == 0:
        raise ValueError("no samples supplied.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"weights must sum to a positive finite value, got {total}.")
    alpha_t = float(alpha_t)
    if not np.isfinite(alpha_t):
        raise ValueError(f"alpha_t must be finite, got {alpha_t}.")

    wrong = (y_true != y_pred).astype(float)
    weighted_error = float((w * wrong).sum() / total)
    w_new = w * np.exp(alpha_t * wrong)
    new_total = float(w_new.sum())
    if not np.isfinite(new_total) or new_total <= 0:
        raise ValueError(
            f"weight update overflowed (sum={new_total}); alpha_t={alpha_t} is too large."
        )
    w_new = w_new / new_total

    return RichResult(
        title="AdaBoost weight update",
        summary_lines=[
            ("Weighted error", weighted_error),
            ("Boost factor", float(np.exp(alpha_t))),
        ],
        payload={
            "weights_new": w_new.tolist(),
            "misclassified": wrong.astype(bool).tolist(),
            "weighted_error": weighted_error,
            "boost_factor": float(np.exp(alpha_t)),
            "alpha_t": alpha_t,
            "estimate": weighted_error,
            "n": int(y_true.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "gradaw: AdaBoost -- w_i *= exp(alpha_t * 1{misclassified}), then renormalise"
