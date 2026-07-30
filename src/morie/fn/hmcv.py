# morie.fn -- function file (rootcoder007/morie)
"""K-fold cross-validation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["k_fold_cross_validation", "geron_cross_validation"]


def k_fold_cross_validation(y, fold_predictions, folds, task="regression"):
    r"""Honest generalisation estimate from per-fold predictions.

    .. math:: \widehat{\mathrm{err}}_{CV} = \frac{1}{K}\sum_{k=1}^{K}
              \frac{1}{|F_k|}\sum_{i\in F_k} L(y_i, \hat f^{-k}(x_i))

    The average is taken over FOLDS, not over rows, which matters when
    the folds are unequal: a row in a small fold then carries more
    weight than one in a large fold. ``row_weighted_error`` gives the
    other convention so the difference is visible instead of silently
    chosen.

    Two properties are worth reporting rather than assuming. The
    between-fold standard error, :math:`\mathrm{sd}(e_k)/\sqrt{K}`,
    UNDERSTATES the true uncertainty, because the training sets overlap
    heavily and the fold errors are positively correlated -- there is
    no unbiased estimator of the variance of K-fold CV (Bengio and
    Grandvalet 2004), and treating this SE as if there were is the
    common error. And CV estimates the error of a model trained on
    :math:`n(K-1)/K` rows, which is pessimistic for the model finally
    fitted on all :math:`n`; that gap shrinks as :math:`K` grows, which
    is the real argument for leave-one-out over 5-fold.

    Parameters
    ----------
    y : array-like, shape (n,)
    fold_predictions : array-like, shape (n,)
        Out-of-fold prediction for each row.
    folds : array-like of int, shape (n,)
        Fold index per row.
    task : {'regression', 'classification'}

    Returns
    -------
    RichResult
        ``cv_error``, ``fold_errors``, ``se``, ``row_weighted_error``,
        ``train_fraction``, ``se_note``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapters 2-3.
    Bengio and Grandvalet (2004), *JMLR* 5:1089-1105, on the absence of
    an unbiased variance estimator.

    Examples
    --------
    >>> out = k_fold_cross_validation([1.0, 2.0, 3.0, 4.0],
    ...                               [1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    >>> float(out["cv_error"])
    0.0
    """
    yv = np.asarray(y, dtype=float).ravel()
    pv = np.asarray(fold_predictions, dtype=float).ravel()
    fv = np.asarray(folds).ravel()
    n = yv.size
    if not (pv.size == fv.size == n):
        raise ValueError(
            "y, fold_predictions and folds must agree in length, got "
            "%d, %d and %d." % (n, pv.size, fv.size)
        )
    if task not in ("regression", "classification"):
        raise ValueError(
            "task must be 'regression' or 'classification', got %r." % task
        )
    keys = list(dict.fromkeys(fv.tolist()))
    K = len(keys)
    if K < 2:
        raise ValueError("need at least 2 folds, got %d." % K)

    def loss(a, b):
        return (a != b).astype(float) if task == "classification" \
            else (a - b) ** 2

    per = []
    sizes = []
    for k in keys:
        m = fv == k
        sizes.append(int(m.sum()))
        per.append(float(np.mean(loss(yv[m], pv[m]))))
    per = np.asarray(per)
    sizes = np.asarray(sizes)
    cv = float(np.mean(per))
    row = float(np.mean(loss(yv, pv)))
    se = float(np.std(per, ddof=1) / np.sqrt(K)) if K > 1 else np.nan
    return RichResult(
        payload={
            "estimate": cv,
            "cv_error": cv,
            "cv_score": (1.0 - cv if task == "classification"
                         else 1.0 - cv / float(np.var(yv))
                         if np.var(yv) > 0 else np.nan),
            "fold_errors": per,
            "fold_sizes": sizes,
            "se": se,
            "se_note": (
                "the between-fold SE understates the true uncertainty: the "
                "training sets overlap so the fold errors are positively "
                "correlated, and no unbiased variance estimator for K-fold "
                "CV exists"
            ),
            "row_weighted_error": row,
            "weighting_note": (
                "cv_error averages over FOLDS, row_weighted_error over ROWS; "
                "they differ whenever the folds are unequal"
            ),
            "train_fraction": float((K - 1) / K),
            "pessimism_note": (
                "CV estimates the error of a model trained on %d%% of the "
                "data, which is pessimistic for the model fitted on all of "
                "it" % round(100 * (K - 1) / K)
            ),
            "k": int(K),
            "n": int(n),
            "task": task,
            "method": "K-fold cross-validation",
        }
    )


def cheatsheet():
    return (
        "hmcv: K-fold CV error with both weighting conventions and the "
        "warning that its standard error is not trustworthy"
    )


#: Catalogue alias for :func:`k_fold_cross_validation`.
geron_cross_validation = k_fold_cross_validation
