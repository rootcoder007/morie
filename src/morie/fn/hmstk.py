# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stacking (blending): meta-learner combines outputs of base learners."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_stacking"]


def geron_stacking(X, y, base_models, meta_model=None, k_folds=3):
    """
    Stacking (blending): meta-learner combines outputs of base learners.

    Formula: y_hat = meta(f_1(x), ..., f_M(x))

    The meta-learner is trained on **out-of-fold** base predictions, not
    in-sample ones. That is the whole difficulty of stacking: a base
    model that memorises the training set looks perfect in-sample, the
    meta-learner learns to trust it, and the ensemble collapses at test
    time. K-fold cross-prediction is what prevents that leak, so it is
    done here rather than skipped.

    Contract for every learner (base and meta): a callable
    ``f(X_train, y_train, X_test) -> predictions`` -- fit and predict in
    one call, which keeps the ensemble independent of any estimator API.
    The default meta-learner is least-squares on the base predictions.

    Parameters
    ----------
    X : array-like
        Design matrix (n, d).
    y : array-like
        Targets, length n.
    base_models : sequence of callables
        At least one base learner with the contract above.
    meta_model : callable, optional
        Blender; default ordinary least squares with intercept.
    k_folds : int, default 3
        Folds for the out-of-fold predictions (2 <= k <= n).

    Returns
    -------
    result : RichResult
        Keys: predicted, meta_features, oof_mse, stacked_mse,
        best_base_mse, gain, estimate, n, method.

    Examples
    --------
    One base model that always predicts the training mean and one that
    fits the (exactly linear) target: the blender learns to follow the
    second, so the stacked error is essentially zero while the mean
    model's out-of-fold error is not.

    >>> import numpy as np
    >>> X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    >>> y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    >>> mean_model = lambda Xtr, ytr, Xte: np.full(len(Xte), np.mean(ytr))
    >>> def ols(Xtr, ytr, Xte):
    ...     A = np.hstack([np.ones((len(Xtr), 1)), np.asarray(Xtr, dtype=float)])
    ...     B = np.hstack([np.ones((len(Xte), 1)), np.asarray(Xte, dtype=float)])
    ...     return B @ np.linalg.lstsq(A, np.asarray(ytr, dtype=float), rcond=None)[0]
    >>> r = geron_stacking(X, y, [mean_model, ols], k_folds=3)
    >>> bool(r["stacked_mse"] < 1e-16)
    True
    >>> bool(r["oof_mse"][1] < r["oof_mse"][0])
    True
    >>> r["meta_features"].shape
    (6, 2)

    References
    ----------
    Géron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_stacking: X must be a non-empty (n, d) design matrix")
    t = np.asarray(y, dtype=float).ravel()
    if t.size != A.shape[0]:
        raise ValueError(f"geron_stacking: X has {A.shape[0]} rows but y has {t.size} targets")
    bases = list(base_models)
    if not bases:
        raise ValueError("geron_stacking: no base models supplied")
    for i, m in enumerate(bases):
        if not callable(m):
            raise ValueError(f"geron_stacking: base model {i} is not callable f(X_train, y_train, X_test)")
    n = int(A.shape[0])
    K = int(k_folds)
    if not (2 <= K <= n):
        raise ValueError(f"geron_stacking: k_folds must lie in 2..{n}, got {K}")

    def _ols(Xtr, ytr, Xte):
        P = np.hstack([np.ones((len(Xtr), 1)), np.asarray(Xtr, dtype=float)])
        Q = np.hstack([np.ones((len(Xte), 1)), np.asarray(Xte, dtype=float)])
        return Q @ np.linalg.lstsq(P, np.asarray(ytr, dtype=float), rcond=None)[0]

    meta = _ols if meta_model is None else meta_model
    if not callable(meta):
        raise ValueError("geron_stacking: meta_model must be a callable f(X_train, y_train, X_test)")

    folds = [np.arange(i, n, K) for i in range(K)]
    if any(f.size == 0 for f in folds):
        raise ValueError(f"geron_stacking: k_folds={K} leaves an empty fold for n={n}")

    M = len(bases)
    Z = np.empty((n, M))
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        for j, m in enumerate(bases):
            p = np.asarray(m(A[mask], t[mask], A[f]), dtype=float).ravel()
            if p.size != f.size:
                raise ValueError(
                    f"geron_stacking: base model {j} returned {p.size} predictions for {f.size} held-out rows"
                )
            if not np.all(np.isfinite(p)):
                raise ValueError(f"geron_stacking: base model {j} returned non-finite predictions")
            Z[f, j] = p

    oof_mse = np.asarray([float(np.mean((Z[:, j] - t) ** 2)) for j in range(M)])
    stacked = np.asarray(meta(Z, t, Z), dtype=float).ravel()
    if stacked.size != n:
        raise ValueError(f"geron_stacking: meta_model returned {stacked.size} predictions for {n} rows")
    if not np.all(np.isfinite(stacked)):
        raise ValueError("geron_stacking: meta_model returned non-finite predictions")
    stacked_mse = float(np.mean((stacked - t) ** 2))
    best = float(np.min(oof_mse))

    return RichResult(
        title="Stacking ensemble",
        summary_lines=[
            ("Base models", M),
            ("Folds", K),
            ("Stacked MSE", stacked_mse),
            ("Best base out-of-fold MSE", best),
        ],
        interpretation=(
            "The blender is fitted on out-of-fold predictions, so a base learner that overfits looks "
            "as bad to the blender as it will at test time."
        ),
        payload={
            "predicted": stacked,
            "meta_features": Z,
            "oof_mse": oof_mse,
            "stacked_mse": stacked_mse,
            "best_base_mse": best,
            "gain": float(best - stacked_mse),
            "n_base": M,
            "estimate": stacked_mse,
            "n": n,
            "method": "Stacking with K-fold out-of-fold meta-features and a least-squares blender by default",
        },
    )


def cheatsheet():
    return "hmstk: Stacking (blending): meta-learner combines outputs of base learners"
