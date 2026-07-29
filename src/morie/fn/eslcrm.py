# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multiclass cross-entropy loss (ESL Ch 4.4 / 11.3)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_cross_entropy"]


def esl_cross_entropy(y, p):
    """
    Cross-entropy (multinomial deviance) loss.

    Formula: L = -sum_k y_k log p_k for a single observation, or the
    MEAN over rows for a matrix of observations. ``y`` may be
    one-hot rows or an integer class vector; both are accepted and
    the interpretation is reported. A predicted probability of zero
    on the true class gives infinite loss -- returned as inf rather
    than clipped, since clipping silently invents a finite number.

    Parameters
    ----------
    y : array-like
        One-hot matrix (n, K), or integer class labels of length n.
    p : array-like
        Predicted probabilities, shape (n, K) or (K,), rows summing
        to 1 within 1e-8.

    Returns
    -------
    result : dict
        Keys: estimate (mean loss), per_observation, n, K,
        label_form, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 4.4 and Eq. 10.22.

    Examples
    --------
    >>> import math
    >>> out = esl_cross_entropy([1, 0], [[0.25, 0.75], [0.9, 0.1]])
    >>> abs(out["per_observation"][0] - -math.log(0.75)) < 1e-15
    True
    >>> abs(out["estimate"] - (-math.log(0.75) - math.log(0.9)) / 2) < 1e-15
    True
    >>> esl_cross_entropy([[0.0, 1.0]], [[0.25, 0.75]])["label_form"]
    'one-hot'
    >>> esl_cross_entropy([0], [[0.0, 1.0]])["estimate"]
    inf
    >>> esl_cross_entropy([0], [[0.5, 0.6]])
    Traceback (most recent call last):
        ...
    ValueError: each row of p must sum to 1; row 0 sums to 1.1.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    y_arr = np.asarray(y)
    if np.any(p < 0):
        raise ValueError("probabilities cannot be negative.")
    for i, s in enumerate(p.sum(axis=1)):
        if abs(s - 1.0) > 1e-8:
            raise ValueError(f"each row of p must sum to 1; row {i} sums to {round(float(s), 12)}.")
    n, K = p.shape
    if y_arr.ndim == 2:
        Y = np.asarray(y_arr, dtype=float)
        form = "one-hot"
        if Y.shape != p.shape:
            raise ValueError(f"y {Y.shape} and p {p.shape} shapes differ.")
    else:
        idx = np.asarray(y_arr, dtype=int).ravel()
        if idx.size != n:
            raise ValueError(f"y has {idx.size} labels but p has {n} rows.")
        if np.any((idx < 0) | (idx >= K)):
            raise ValueError(f"class labels must lie in [0, {K - 1}].")
        Y = np.zeros_like(p)
        Y[np.arange(n), idx] = 1.0
        form = "class-index"
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(Y > 0, -Y * np.log(p), 0.0)
    per = [float(v) for v in terms.sum(axis=1)]
    return RichResult(payload={
        "estimate": float(np.mean(per)), "per_observation": per,
        "n": int(n), "K": int(K), "label_form": form,
        "method": "cross-entropy -sum y_k log p_k, mean over observations"})


def cheatsheet():
    return "eslcrm: -sum y log p; one-hot or class-index y; p=0 on truth -> inf"
