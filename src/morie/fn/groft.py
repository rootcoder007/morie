# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Overfitting gap: training accuracy minus validation accuracy."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_overfitting_gap"]

_METHOD = "Train-validation generalisation gap"


def geron_overfitting_gap(train_scores, val_scores):
    r"""Per-epoch generalisation gap and where it opens.

    .. math::
        \mathrm{gap}_t = \mathrm{acc}^{\text{train}}_t
                       - \mathrm{acc}^{\text{val}}_t

    A positive gap means the model does better on data it has seen.  The
    useful output is not the final number but the *curve*: the epoch where
    validation score peaks is the early-stopping point, and everything
    after it is memorisation.  Both are reported.

    Parameters
    ----------
    train_scores, val_scores : array-like
        Per-epoch scores, same length. Higher = better.

    Returns
    -------
    RichResult
        Payload keys ``gap`` (per epoch), ``final_gap``, ``max_gap``,
        ``max_gap_epoch``, ``best_val_epoch``, ``overfitting_epochs``,
        ``estimate`` (final gap), ``n``, ``method``.

    References
    ----------
    Géron Ch 1, Overfitting section.

    Examples
    --------
    Validation peaks at epoch 1 and then falls while training keeps
    climbing -- textbook overfitting:

    >>> r = geron_overfitting_gap([0.80, 0.90, 0.97], [0.78, 0.86, 0.80])
    >>> [round(g, 10) for g in r["gap"]]
    [0.02, 0.04, 0.17]
    >>> r["best_val_epoch"]
    1
    >>> r["max_gap_epoch"]
    2
    """
    tr = np.asarray(train_scores, dtype=float).ravel()
    va = np.asarray(val_scores, dtype=float).ravel()
    if tr.size == 0:
        raise ValueError("train_scores is empty.")
    if tr.shape != va.shape:
        raise ValueError(
            f"train_scores has {tr.size} epochs but val_scores has {va.size}."
        )
    if not np.all(np.isfinite(tr)) or not np.all(np.isfinite(va)):
        raise ValueError("scores must be finite.")

    gap = tr - va
    best_val = int(np.argmax(va))
    return RichResult(
        title="Overfitting gap",
        summary_lines=[("Final gap", float(gap[-1])), ("Best val epoch", best_val)],
        payload={
            "gap": gap.tolist(),
            "final_gap": float(gap[-1]),
            "max_gap": float(gap.max()),
            "max_gap_epoch": int(np.argmax(gap)),
            "best_val_epoch": best_val,
            "overfitting_epochs": int((gap > 0).sum()),
            "estimate": float(gap[-1]),
            "n": int(tr.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "groft: gap_t = train_t - val_t per epoch; also best-val epoch (early-stopping point)"
