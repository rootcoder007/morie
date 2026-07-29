# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Learning curves: train and validation RMSE against training-set size."""

import numpy as np

from ._richresult import RichResult
from .grmse import geron_linreg_mse_cost
from .grn005 import geron_ch4_normal_equation

__all__ = ["geron_learning_curves"]

_METHOD = "Learning curves (train / validation RMSE vs m)"


def geron_learning_curves(X, y, n_splits=10, val_fraction=0.2):
    r"""Refit on growing prefixes of the training set and score both sets.

    .. math::
        \mathrm{RMSE}_{\text{train}}(m),\quad
        \mathrm{RMSE}_{\text{val}}(m)
        \quad \text{over growing subsets of size } m

    The shape is the diagnosis.  Training error that starts at zero and
    climbs, meeting a validation curve that is only slightly higher, is
    underfitting -- more data will not help.  A wide, persistent gap is
    overfitting.  ``final_gap`` reports that gap at full size.

    Each model is the closed-form least-squares fit from
    :func:`morie.fn.grn005.geron_ch4_normal_equation`, scored with
    :func:`morie.fn.grmse.geron_linreg_mse_cost`.  Subsets are prefixes
    of the given order, so the caller controls shuffling and the result
    is reproducible.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Include the bias column if the model has an intercept.
    y : array-like, shape (m,)
    n_splits : int, optional
        Number of training sizes to try, at least 1. Default 10.
    val_fraction : float, optional
        Tail fraction held out for validation, in ``(0, 1)``. Default
        0.2.

    Returns
    -------
    RichResult
        Payload keys ``train_sizes``, ``train_rmse``, ``val_rmse``,
        ``final_gap``, ``val_size``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Learning Curves section.

    Examples
    --------
    Noise-free linear data: as soon as there are as many points as
    parameters the fit is exact, so both curves sit at zero.

    >>> X = [[1.0, float(i)] for i in range(10)]
    >>> y = [float(i) for i in range(10)]
    >>> r = geron_learning_curves(X, y, n_splits=3)
    >>> [round(v, 10) for v in r["train_rmse"]]
    [0.0, 0.0, 0.0]
    >>> [round(v, 10) for v in r["val_rmse"]]
    [0.0, 0.0, 0.0]
    >>> r["train_sizes"]
    [2, 5, 8]

    Break the linearity and the validation curve stays above the
    training curve -- the gap the plot exists to show:

    >>> y2 = [float(i * i) for i in range(10)]
    >>> r2 = geron_learning_curves(X, y2, n_splits=3)
    >>> r2["final_gap"] > 0
    True
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    y_arr = np.asarray(y, dtype=float).ravel()
    if A.ndim != 2:
        raise ValueError(f"X must be 2-D of shape (m, n), got shape {A.shape}.")
    m, n_feat = A.shape
    if y_arr.size != m:
        raise ValueError(f"y has {y_arr.size} entries but X has {m} rows.")
    val_fraction = float(val_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}.")
    n_val = int(round(m * val_fraction))
    if n_val < 1:
        raise ValueError(
            f"val_fraction {val_fraction} leaves no validation instances out of "
            f"{m} rows; use more data or a larger fraction."
        )
    n_train = m - n_val
    if n_train < n_feat:
        raise ValueError(
            f"only {n_train} training rows for {n_feat} parameters; the smallest "
            f"least-squares fit needs at least as many rows as columns."
        )
    n_splits = int(n_splits)
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}.")

    Xtr, ytr = A[:n_train], y_arr[:n_train]
    Xva, yva = A[n_train:], y_arr[n_train:]
    sizes = np.unique(np.linspace(n_feat, n_train, n_splits).astype(int))

    tr_rmse, va_rmse = [], []
    for s in sizes:
        theta = geron_ch4_normal_equation(Xtr[:s], ytr[:s])["theta"]
        tr_rmse.append(geron_linreg_mse_cost(Xtr[:s], ytr[:s], theta)["rmse"])
        va_rmse.append(geron_linreg_mse_cost(Xva, yva, theta)["rmse"])

    return RichResult(
        title="Learning curves",
        summary_lines=[("Sizes", sizes.tolist()),
                       ("Final train RMSE", tr_rmse[-1]),
                       ("Final val RMSE", va_rmse[-1])],
        payload={
            "train_sizes": sizes.tolist(),
            "train_rmse": tr_rmse,
            "val_rmse": va_rmse,
            "final_gap": float(va_rmse[-1] - tr_rmse[-1]),
            "val_size": int(n_val),
            "estimate": va_rmse,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlrnc: OLS refit on growing prefixes; train vs val RMSE, gap = overfitting"
