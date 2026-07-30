# morie.fn -- function file (rootcoder007/morie)
"""K-fold cross-validation -- ESL Sec 7.10."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_cv_score"]


def esl_cv_score(X, y, model=None, k=5, loss="mse", stratify=False, seed=0):
    r"""Estimate prediction error by ``k``-fold cross-validation.

    .. math::
        \mathrm{CV} = \frac{1}{n}\sum_{i=1}^{n}
                      L\!\left(y_i,\; \hat f^{-\kappa(i)}(x_i)\right),

    where :math:`\kappa(i)` is the fold holding observation :math:`i`, so
    every prediction comes from a model that never saw that observation.

    ``model`` is any callable ``model(X_train, y_train, X_test) -> y_pred``,
    which keeps this independent of any particular estimator. The default is
    ordinary least squares with an intercept.

    The reported ``se`` is the standard error *across folds*, which is what
    ESL's one-standard-error rule uses. It is not a standard error for the
    population prediction error -- the folds share training data and so are
    positively correlated, which biases it downward. Use it to compare models,
    not as a confidence statement.

    Parameters
    ----------
    X : array-like
        Predictors ``(n, p)``.
    y : array-like
        Response ``(n,)``.
    model : callable, optional
        ``model(X_tr, y_tr, X_te) -> predictions``. Defaults to OLS.
    k : int
        Number of folds, from 2 to ``n``. ``k = n`` is leave-one-out.
    loss : {"mse", "mae", "01"}
        Loss function. ``"01"`` is misclassification rate.
    stratify : bool
        Preserve class proportions per fold. Requires a discrete ``y``.
    seed : int
        Seed for the fold shuffle.

    Returns
    -------
    RichResult
        ``cv`` (the estimate), ``se``, ``fold_scores``, ``predictions``,
        ``fold_id``, ``k``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    On data with real signal, CV error is well below the variance of ``y``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = X @ [1.0, -2.0, 0.5] + rng.normal(0, 0.3, 200)
    >>> r = esl_cv_score(X, y, k=5)
    >>> bool(r["cv"] < 0.2 * y.var())
    True

    Every observation is predicted exactly once, out of fold.

    >>> sorted(np.bincount(r["fold_id"]).tolist()) == sorted(np.bincount(r["fold_id"]).tolist())
    True
    >>> int(np.sum(np.isnan(r["predictions"])))
    0

    Pure noise gives CV error at least as large as the variance of ``y`` --
    cross-validation does not reward overfitting.

    >>> yn = rng.normal(size=200)
    >>> bool(esl_cv_score(X, yn, k=5)["cv"] > 0.9 * yn.var())
    True

    >>> esl_cv_score(X, y, k=1)
    Traceback (most recent call last):
        ...
    ValueError: k must be between 2 and n
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but y has {n}")
    k = int(k)
    if not 2 <= k <= n:
        raise ValueError("k must be between 2 and n")
    if model is None:
        model = _ols

    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=int)
    if stratify:
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            fold[idx] = np.arange(idx.size) % k
    else:
        idx = rng.permutation(n)
        fold[idx] = np.arange(n) % k

    losses = {"mse": lambda a, b: (a - b) ** 2,
              "mae": lambda a, b: np.abs(a - b),
              "01": lambda a, b: (a != b).astype(float)}
    if loss not in losses:
        raise ValueError(f'loss must be one of {sorted(losses)}, got {loss!r}')
    lf = losses[loss]

    pred = np.full(n, np.nan)
    scores = np.empty(k)
    for j in range(k):
        te = fold == j
        tr = ~te
        if not tr.any():
            raise ValueError(f"fold {j} leaves no training data")
        pred[te] = np.asarray(model(X[tr], y[tr], X[te]), dtype=float).ravel()
        scores[j] = float(np.mean(lf(y[te], pred[te])))

    return RichResult(
        title=f"{k}-fold cross-validation",
        summary_lines=[("n", n), ("k", k), ("loss", loss), ("CV", float(np.mean(lf(y, pred))))],
        payload={
            "cv": float(np.mean(lf(y, pred))),
            "se": float(np.std(scores, ddof=1) / np.sqrt(k)) if k > 1 else np.nan,
            "fold_scores": scores, "predictions": pred, "fold_id": fold,
            "k": k, "loss": loss, "n": int(n),
            "method": "esl_cv_score",
        },
    )


def _ols(X_tr, y_tr, X_te):
    A = np.column_stack([np.ones(len(y_tr)), X_tr])
    beta = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    return np.column_stack([np.ones(X_te.shape[0]), X_te]) @ beta


def cheatsheet():
    return "eslcvr: k-fold CV for any model(X_tr,y_tr,X_te); `se` is across folds, correlated and biased low"
