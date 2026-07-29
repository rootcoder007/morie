# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Underfitting: high training error because the model is too simple."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_underfitting"]


def geron_underfitting(train_err, threshold=None, val_err=None, baseline=None, tol=0.05):
    """
    Underfitting: high training error because the model is too simple.

    Formula: E_train high; model bias > variance

    Underfitting is not "big error", it is "big error *and* a small
    train-validation gap": the model is failing on data it has already
    seen, so more data will not help. The diagnosis therefore uses both
    numbers when they are available:

    * ``train_err > threshold`` and gap ``<= tol`` -> underfitting (bias);
    * gap ``> tol`` -> overfitting (variance) regardless of train error;
    * otherwise -> fit is adequate against the threshold.

    Learning curves may be passed instead of scalars; the final epoch is
    the verdict, and the plateau slope over the last third of the curve
    tells you whether training had actually converged (a curve still
    falling steeply is not yet evidence of underfitting).

    Parameters
    ----------
    train_err : float or array-like
        Training error, or a training-error curve.
    threshold : float, optional
        Acceptable training error. Defaults to `baseline` when given,
        otherwise it must be supplied.
    val_err : float or array-like, optional
        Validation error or curve.
    baseline : float, optional
        Error of a trivial predictor, for context.
    tol : float, default 0.05
        Absolute train-validation gap above which the problem is variance
        rather than bias.

    Returns
    -------
    result : RichResult
        Keys: diagnosis, underfitting, train_error, val_error, gap,
        plateau_slope, estimate, n, method.

    Examples
    --------
    High training error with almost no gap: the model is too simple.

    >>> r = geron_underfitting(0.40, threshold=0.10, val_err=0.41)
    >>> r["diagnosis"]
    'underfitting'
    >>> bool(r["underfitting"])
    True
    >>> round(float(r["gap"]), 12)
    0.01

    Low training error with a large gap is the opposite problem:

    >>> geron_underfitting(0.01, threshold=0.10, val_err=0.40)["diagnosis"]
    'overfitting'
    >>> geron_underfitting(0.02, threshold=0.10, val_err=0.03)["diagnosis"]
    'adequate'

    References
    ----------
    Géron Ch 1
    """
    tr = np.atleast_1d(np.asarray(train_err, dtype=float)).ravel()
    if tr.size == 0:
        raise ValueError("geron_underfitting: train_err is empty")
    if not np.all(np.isfinite(tr)):
        raise ValueError("geron_underfitting: train_err contains non-finite values")
    if np.any(tr < 0):
        raise ValueError("geron_underfitting: an error cannot be negative")
    thr = threshold if threshold is not None else baseline
    if thr is None:
        raise ValueError("geron_underfitting: supply `threshold` (or `baseline`) to judge the training error against")
    thr = float(thr)
    if not np.isfinite(thr) or thr < 0:
        raise ValueError(f"geron_underfitting: threshold must be non-negative and finite, got {thr}")
    t = float(tol)
    if not np.isfinite(t) or t < 0:
        raise ValueError(f"geron_underfitting: tol must be non-negative and finite, got {t}")

    final_tr = float(tr[-1])
    va = None
    gap = None
    if val_err is not None:
        vv = np.atleast_1d(np.asarray(val_err, dtype=float)).ravel()
        if vv.size == 0 or not np.all(np.isfinite(vv)):
            raise ValueError("geron_underfitting: val_err must be non-empty and finite")
        va = float(vv[-1])
        gap = va - final_tr

    slope = None
    if tr.size >= 3:
        tail = tr[-max(2, tr.size // 3) :]
        idx = np.arange(tail.size, dtype=float)
        slope = float(np.polyfit(idx, tail, 1)[0])

    scale = max(final_tr, np.finfo(float).tiny)
    if gap is not None and gap > t:
        diagnosis = "overfitting"
    elif final_tr > thr:
        diagnosis = "underfitting"
    else:
        diagnosis = "adequate"

    return RichResult(
        title="Underfitting diagnosis",
        summary_lines=[
            ("Training error", final_tr),
            ("Validation error", va if va is not None else "n/a"),
            ("Threshold", thr),
            ("Diagnosis", diagnosis),
        ],
        interpretation=(
            "Underfitting is a bias problem: the training error itself is high, so more data will not "
            "help -- a richer model, better features or less regularisation will."
        ),
        payload={
            "diagnosis": diagnosis,
            "underfitting": diagnosis == "underfitting",
            "train_error": final_tr,
            "val_error": va,
            "gap": gap,
            "threshold": thr,
            "baseline": None if baseline is None else float(baseline),
            "plateau_slope": slope,
            "converged": None if slope is None else bool(abs(slope) <= 0.01 * scale),
            "estimate": final_tr,
            "n": int(tr.size),
            "method": "Bias/variance verdict from the training error against a threshold plus the train-validation gap",
        },
    )


def cheatsheet():
    return "hmuf: Underfitting: high training error because the model is too simple"
