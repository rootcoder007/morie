# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Overfitting diagnostic from training and validation error."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_overfitting"]


def geron_overfitting(train_err, val_err, tol=0.0):
    """
    Overfitting: training error much lower than validation error.

    Formula: E_train << E_val; gap = E_val - E_train

    Scalars give the gap and its ratio. Whole learning CURVES (one entry
    per epoch) give more: the epoch of minimum validation error is the
    early-stopping point, and the gap's trend after it separates the two
    failure modes Geron distinguishes -- a gap that grows while training
    error keeps falling is overfitting, while both curves stalling high
    is underfitting, which more data will not fix.

    Parameters
    ----------
    train_err, val_err : float or array-like
        Errors (lower is better), scalar or one per epoch.
    tol : float, default 0.0
        Gap below which the fit is called neither over- nor underfit.

    Returns
    -------
    result : RichResult
        Keys: gap, ratio, overfitting, best_epoch, best_val, verdict,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_overfitting(0.1, 0.5)
    >>> float(r["gap"]), float(r["ratio"])
    (0.4, 5.0)
    >>> bool(r["overfitting"])
    True

    On a curve the best epoch is the validation minimum, not the last:

    >>> c = geron_overfitting([1.0, 0.5, 0.2, 0.1], [1.0, 0.6, 0.5, 0.7])
    >>> int(c["best_epoch"]), float(c["best_val"])
    (2, 0.5)
    >>> round(float(c["gap"]), 6)
    0.6

    References
    ----------
    Geron Ch 1
    """
    tr = np.atleast_1d(np.asarray(train_err, dtype=float)).ravel()
    va = np.atleast_1d(np.asarray(val_err, dtype=float)).ravel()
    if tr.size == 0 or va.size == 0:
        raise ValueError("geron_overfitting: train_err and val_err must be non-empty")
    if tr.size != va.size:
        raise ValueError(f"geron_overfitting: train_err has {tr.size} entries but val_err has {va.size}")
    if not np.all(np.isfinite(tr)) or not np.all(np.isfinite(va)):
        raise ValueError("geron_overfitting: errors contain non-finite values")
    if np.any(tr < 0) or np.any(va < 0):
        raise ValueError("geron_overfitting: errors must be non-negative")
    t = float(tol)
    if t < 0:
        raise ValueError(f"geron_overfitting: tol must be non-negative, got {tol!r}")

    gaps = va - tr
    gap = float(gaps[-1])
    if tr[-1] == 0.0:
        ratio = float("inf") if va[-1] > 0 else 1.0
    else:
        ratio = float(va[-1] / tr[-1])
    k = int(np.argmin(va))
    over = bool(gap > t)
    if over:
        verdict = "overfitting: validation error exceeds training error by more than tol"
    elif va[-1] > 0 and tr[-1] > 0 and abs(gap) <= t and va[-1] >= float(np.min(va)) * 1.0 and va.size == 1:
        verdict = "no gap: train and validation agree at this tolerance"
    else:
        verdict = "no overfitting gap: errors agree, so look at their LEVEL for underfitting"
    return RichResult(
        title="Overfitting diagnostic",
        summary_lines=[("Train error", float(tr[-1])), ("Validation error", float(va[-1])), ("Gap", gap)],
        interpretation=verdict,
        payload={
            "gap": gap,
            "gaps": gaps,
            "ratio": ratio,
            "overfitting": over,
            "best_epoch": k,
            "best_val": float(va[k]),
            "epochs_past_best": int(va.size - 1 - k),
            "verdict": verdict,
            "estimate": gap,
            "n": int(tr.size),
            "method": "Generalisation gap E_val - E_train with early-stopping point",
        },
    )


def cheatsheet():
    return "hmovf: Overfitting diagnostic from train/validation error"
