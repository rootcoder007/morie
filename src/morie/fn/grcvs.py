# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean cross-validation score across K folds."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_cross_validation_score"]

_METHOD = "K-fold cross-validation"


def _ols_fit(Xtr, ytr):
    theta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return theta


def _ols_predict(theta, Xte):
    return Xte @ theta


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        raise ValueError(
            "a validation fold has zero target variance, so R^2 is undefined; "
            "pass your own score callable or shuffle the data."
        )
    return 1.0 - ss_res / ss_tot


def geron_cross_validation_score(X, y, K, fit=None, predict=None, score=None,
                                 shuffle=False, random_state=None):
    r"""Average a model's score over ``K`` held-out folds.

    .. math::
        \text{CV} = \frac{1}{K}\sum_{k=1}^{K}
        \text{score}\bigl(\text{model}_k, \text{fold}_k\bigr)

    Each fold trains on ``K-1`` parts and is scored on the part it never
    saw.  The *spread* across folds matters as much as the mean: a
    respectable average hiding one catastrophic fold usually means the
    folds are not exchangeable -- ordered data, or a grouping the split
    ignored -- so the per-fold scores and their standard error are
    returned alongside.

    By default the model is ordinary least squares (via ``lstsq``) scored
    by :math:`R^2`; supply ``fit``/``predict``/``score`` to cross-validate
    anything else.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix -- include a bias column yourself if wanted.
    y : array-like, shape (m,)
        Targets.
    K : int
        Number of folds, ``2 <= K <= m``.
    fit : callable, optional
        ``fit(X_train, y_train) -> model``.
    predict : callable, optional
        ``predict(model, X_test) -> y_pred``. Required when ``fit`` is
        given.
    score : callable, optional
        ``score(y_true, y_pred) -> float``, higher is better.
    shuffle : bool, optional
        Permute rows before splitting.
    random_state : int, optional
        Seed for ``shuffle``.

    Returns
    -------
    RichResult
        Payload keys ``cv_score``, ``fold_scores``, ``fold_sizes``,
        ``se``, ``worst_fold``, ``spread``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 1 (intro), Ch 2 (use).

    Examples
    --------
    Noiseless ``y = 2x``: every fold recovers the slope, so every fold
    scores 1.

    >>> r = geron_cross_validation_score([[1.0], [2.0], [3.0], [4.0]],
    ...                                  [2.0, 4.0, 6.0, 8.0], K=2)
    >>> [round(s, 8) for s in r["fold_scores"]]
    [1.0, 1.0]
    >>> round(r["cv_score"], 8)
    1.0

    A custom scorer -- negative mean absolute error -- on the same data:

    >>> import numpy as np
    >>> mae = lambda yt, yp: -float(np.mean(np.abs(yt - yp)))
    >>> round(geron_cross_validation_score([[1.0], [2.0], [3.0], [4.0]],
    ...                                    [2.0, 4.0, 6.0, 8.0], K=2,
    ...                                    score=mae)["cv_score"], 8)
    0.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size} entries.")
    if X.size == 0:
        raise ValueError("X is empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite.")
    m = X.shape[0]
    K = int(K)
    if K < 2:
        raise ValueError(f"K must be at least 2 folds, got {K}.")
    if K > m:
        raise ValueError(f"K={K} exceeds the {m} available observations.")

    if fit is None and predict is not None:
        raise ValueError("predict was supplied without fit.")
    if fit is not None and predict is None:
        raise ValueError("fit was supplied without predict.")
    fit_fn = _ols_fit if fit is None else fit
    pred_fn = _ols_predict if predict is None else predict
    score_fn = _r2 if score is None else score
    for name, fn in (("fit", fit_fn), ("predict", pred_fn), ("score", score_fn)):
        if not callable(fn):
            raise ValueError(f"{name} must be callable, got {type(fn).__name__}.")

    idx = np.arange(m)
    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(m)
    folds = np.array_split(idx, K)

    scores = []
    sizes = []
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
        if train_idx.size == 0:
            raise ValueError(f"fold {k} leaves no training data.")
        model = fit_fn(X[train_idx], y[train_idx])
        yp = np.asarray(pred_fn(model, X[test_idx]), dtype=float).ravel()
        if yp.size != test_idx.size:
            raise ValueError(
                f"predict returned {yp.size} predictions for fold {k}, expected "
                f"{test_idx.size}."
            )
        s = float(score_fn(y[test_idx], yp))
        if not np.isfinite(s):
            raise ValueError(f"score for fold {k} is not finite ({s}).")
        scores.append(s)
        sizes.append(int(test_idx.size))

    arr = np.asarray(scores, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(K)) if K > 1 else float("nan")

    return RichResult(
        title=f"{K}-fold cross-validation",
        summary_lines=[("CV score", float(arr.mean())), ("SE across folds", se)],
        payload={
            "cv_score": float(arr.mean()),
            "fold_scores": arr.tolist(),
            "fold_sizes": sizes,
            "se": se,
            "worst_fold": int(arr.argmin()),
            "spread": float(arr.max() - arr.min()),
            "K": K,
            "estimate": float(arr.mean()),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grcvs: K-fold CV score = mean over folds, with per-fold spread (OLS/R^2 by default)"
