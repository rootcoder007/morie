# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdaBoost with decision stumps."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_boosting"]


def _best_stump(X, y, w):
    """Weighted-error-minimising axis-aligned stump; ties resolved
    by (feature, threshold, sign) order for determinism."""
    n, d = X.shape
    best = (float("inf"), 0, 0.0, 1)
    for j in range(d):
        for thr in np.unique(X[:, j]):
            for sign in (1, -1):
                pred = np.where(X[:, j] <= thr, sign, -sign)
                err = float(np.sum(w[pred != y]))
                if err < best[0] - 1e-15:
                    best = (err, j, float(thr), sign)
    return best


def wasserman_boosting(X, y, model, T):
    """
    AdaBoost.M1 with decision stumps.

    Formula: alpha_t = (1/2) log((1 - err_t) / err_t); weights
    update w_i <- w_i exp(-alpha_t y_i h_t(x_i)) and renormalise;
    the committee is sign(sum_t alpha_t h_t). ``model`` = None uses
    the built-in exhaustive stump learner (deterministic tie-breaks);
    a callable (X, y, w) -> (predict_fn) is accepted for custom weak
    learners. Rounds stop early when a stump is perfect (err = 0) or
    no better than chance (err >= 1/2).

    Parameters
    ----------
    X : array-like, shape (n, d)
        Features.
    y : array-like
        Labels in {-1, +1}.
    model : callable or None
        Weak learner factory; None = stumps.
    T : int
        Boosting rounds, >= 1.

    Returns
    -------
    result : dict
        Keys: estimate (training error rate), prediction (committee
        signs), alphas, rounds_used, n, method.

    References
    ----------
    Wasserman (2004), Ch 22 (boosting); Freund & Schapire (1997).

    Examples
    --------
    XOR needs multiple stumps; a threshold problem needs one:

    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [1, 1, -1, -1]
    >>> out = wasserman_boosting(X, y, None, 5)
    >>> out["estimate"]
    0.0
    >>> out["rounds_used"]
    1
    >>> out["prediction"]
    [1, 1, -1, -1]
    >>> wasserman_boosting(X, [1, 2, 3, 4], None, 3)
    Traceback (most recent call last):
        ...
    ValueError: labels must lie in {-1, +1}.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = X.shape[0]
    T = int(T)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} labels.")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("labels must lie in {-1, +1}.")
    if T < 1:
        raise ValueError(f"boosting needs T >= 1 rounds; got {T}.")
    w = np.full(n, 1.0 / n)
    F = np.zeros(n)
    alphas = []
    rounds = 0
    for _ in range(T):
        if model is None:
            err, j, thr, sign = _best_stump(X, y, w)
            pred = np.where(X[:, j] <= thr, sign, -sign).astype(float)
        else:
            predict = model(X, y, w)
            pred = np.asarray(predict(X), dtype=float)
            err = float(np.sum(w[pred != y]))
        if err >= 0.5:
            break
        rounds += 1
        if err == 0.0:
            alpha = 10.0  # capped: a perfect stump dominates the committee
        else:
            alpha = 0.5 * math.log((1.0 - err) / err)
        alphas.append(float(alpha))
        F += alpha * pred
        if err == 0.0:
            break
        w = w * np.exp(-alpha * y * pred)
        w /= np.sum(w)
    committee = np.where(F >= 0, 1, -1)
    train_err = float(np.mean(committee != y))
    return RichResult(payload={
        "estimate": train_err,
        "prediction": [int(v) for v in committee],
        "alphas": alphas, "rounds_used": rounds, "n": int(n),
        "method": "AdaBoost.M1, exhaustive stumps, deterministic ties; perfect-stump alpha capped at 10"})


def cheatsheet():
    return "wsmbst: alpha = .5 log((1-err)/err); stop at err=0 or err>=.5"
