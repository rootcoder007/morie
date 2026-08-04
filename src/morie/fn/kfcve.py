# morie.fn -- slice s04 (rootcoder007/morie)
"""K-fold cross-validation prediction error.

Source consulted: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, Section 4.3.2 and equation (4.1).  The data set
is split into K complementary folds; the model is fitted K times, each
time holding one fold out; the testing mean square error of fold k is

    MSE_k = (1/T_k) sum_{i in fold k} (y_i - yhat_i)^2      (4.1)

and "the arithmetic mean of the k folds is obtained and reported as the
prediction performance", that is

    CV_K = (1/K) sum_k MSE_k.

Section 4.3.3 notes that the leave-one-out scheme is the K = n case of
the same construction, which is what the LOO anchor exercises.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["k_fold_cv_error"]


def _blocks(n, K):
    """The book's complementary folds, taken in order rather than at random.

    Fold k gets the contiguous block of ceil/floor size, so that K = n
    reproduces leave-one-out exactly and K = 1 reproduces the whole set.
    """
    out = []
    start = 0
    for j in range(K):
        m = n // K + (1 if j < n % K else 0)
        out.append(list(range(start, start + m)))
        start += m
    return out


def k_fold_cv_error(y, y_hat_folds, folds=None):
    """Average of the per-fold testing mean square errors.

    Parameters
    ----------
    y : array-like
        The n observed responses.
    y_hat_folds : sequence of sequences
        y_hat_folds[k] holds the predictions for the members of fold k,
        in the order the fold lists them.
    folds : sequence of sequences, optional
        0-based index sets.  Defaults to the contiguous complementary
        partition of 0..n-1 into K blocks of near-equal size.

    Returns
    -------
    estimate : CV_K
    cv_error : the same value
    mse_fold : the K per-fold mean square errors
    """
    yy = k.vec(y)
    n = len(yy)
    if n == 0:
        raise ValueError("k_fold_cv_error: y is empty")
    yh = [k.vec(f) for f in y_hat_folds]
    K = len(yh)
    if K == 0:
        raise ValueError("k_fold_cv_error: no folds supplied")
    idx = [[int(i) for i in f] for f in folds] if folds is not None else _blocks(n, K)
    if len(idx) != K:
        raise ValueError("k_fold_cv_error: folds and y_hat_folds have different lengths")
    mse = []
    for j in range(K):
        if len(idx[j]) != len(yh[j]):
            raise ValueError("k_fold_cv_error: fold %d has a prediction count that does not match it" % j)
        if not idx[j]:
            raise ValueError("k_fold_cv_error: fold %d is empty" % j)
        s = 0.0
        for a in range(len(idx[j])):
            i = idx[j][a]
            if i < 0 or i >= n:
                raise ValueError("k_fold_cv_error: fold index out of range")
            d = yy[i] - yh[j][a]
            s += d * d
        mse.append(s / len(idx[j]))
    cv = 0.0
    for v in mse:
        cv += v
    cv = cv / K
    return RichResult(
        title="K-fold cross-validation error",
        summary_lines=[("n", n), ("K", K)],
        payload={
            "estimate": cv,
            "cv_error": cv,
            "mse_fold": mse,
            "rmse": math.sqrt(cv),
            "n": n,
            "method": "CV_K = (1/K) sum_k MSE_k, Chapter 4 Sect. 4.3.2 with MSE from eq. (4.1)",
        },
    )


def cheatsheet():
    return "kfcve: K-fold cross-validation prediction error"


# compact alias per ledger/NAMING.md
kfoldcverror = k_fold_cv_error
