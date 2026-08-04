# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdaBoost: train sequential weighted weak learners."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_adaboost"]


def _fit_stump(X, y, w):
    """Weighted 1-split decision stump; returns (err, feature, threshold, polarity)."""
    n, d = X.shape
    best = (np.inf, 0, -np.inf, 1)
    for j in range(d):
        vals = np.unique(X[:, j])
        thrs = (vals[:-1] + vals[1:]) / 2.0 if vals.size > 1 else vals
        for thr in thrs:
            left = X[:, j] <= thr
            for pol in (1, -1):
                pred = np.where(left, pol, -pol)
                err = float(np.sum(w[pred != y]))
                if err < best[0]:
                    best = (err, j, float(thr), pol)
    return best


def _stump_predictor(j, thr, pol):
    def predict(A, _j=j, _t=thr, _p=pol):
        B = np.atleast_2d(np.asarray(A, dtype=float))
        return np.where(B[:, _j] <= _t, _p, -_p).astype(float)

    return predict


def geron_adaboost(X, y, base_estimator=None, n_estimators=10, eps=1e-10):
    """
    AdaBoost: train sequential weighted weak learners.

    Formula: alpha_t = 0.5*log((1-err_t)/err_t);
    w_{i,t+1} = w_{i,t}*exp(-alpha_t y_i f_t(x_i))

    Discrete (binary) AdaBoost. Labels are mapped to {-1, +1}. The default
    weak learner is a weighted decision stump; supply your own as
    ``base_estimator(X, y, sample_weight) -> predict(X) -> {-1,+1}``.

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
        Two distinct labels; {0,1} and {-1,1} are both accepted.
    base_estimator : callable, optional
        Weak-learner factory as described above.
    n_estimators : int
        Number of boosting rounds (>= 1).
    eps : float
        Floor/ceiling on the weighted error, so alpha stays finite when a
        weak learner is perfect on the current weighting.

    Returns
    -------
    result : RichResult
        Keys: alphas, errors, train_errors, predict, decision, margin,
        estimate, n, method.

    Examples
    --------
    Parity data no stump can separate: the best weighted error is 1/4, so
    alpha_1 = 0.5 log(3):

    >>> r = geron_adaboost([[1.0], [2.0], [3.0], [4.0]], [1, -1, 1, -1], n_estimators=1)
    >>> round(float(r["errors"][0]), 12)
    0.25
    >>> round(float(r["alphas"][0]), 6)
    0.549306
    >>> float(r["train_errors"][-1])
    0.25

    Separable data is fit exactly in one round:

    >>> r2 = geron_adaboost([[1.0], [2.0], [3.0], [4.0]], [-1, -1, 1, 1], n_estimators=1)
    >>> float(r2["train_errors"][-1])
    0.0

    References
    ----------
    Géron Ch 6
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_adaboost: X must be 2-D, got ndim={A.ndim}")
    n = A.shape[0]
    if n == 0:
        raise ValueError("geron_adaboost: X has no rows")
    yy = np.asarray(y).ravel()
    if yy.size != n:
        raise ValueError(f"geron_adaboost: X has {n} rows but y has {yy.size} entries")
    classes = np.unique(yy)
    if classes.size != 2:
        raise ValueError(f"geron_adaboost: discrete AdaBoost needs exactly 2 classes, got {classes.size}")
    ys = np.where(yy == classes[1], 1.0, -1.0)
    M = int(n_estimators)
    if M < 1:
        raise ValueError("geron_adaboost: n_estimators must be >= 1")
    e = float(eps)
    if not (0.0 < e < 0.5):
        raise ValueError("geron_adaboost: eps must lie in (0, 0.5)")

    w = np.full(n, 1.0 / n)
    predictors, alphas, errs, train_errs = [], [], [], []
    F = np.zeros(n)

    for _ in range(M):
        if base_estimator is None:
            err, j, thr, pol = _fit_stump(A, ys, w)
            pred = np.where(A[:, j] <= thr, pol, -pol).astype(float)
            fitted = _stump_predictor(j, thr, pol)
        else:
            fitted = base_estimator(A, ys, w)
            if not callable(fitted):
                raise ValueError("geron_adaboost: base_estimator must return a callable predictor")
            pred = np.asarray(fitted(A), dtype=float).ravel()
            if pred.size != n:
                raise ValueError(f"geron_adaboost: weak learner returned {pred.size} predictions for {n} rows")
            if not np.all(np.isin(pred, (-1.0, 1.0))):
                raise ValueError("geron_adaboost: weak learner must return labels in {-1, +1}")
            err = float(np.sum(w[pred != ys]))

        err_c = min(max(err, e), 1.0 - e)
        alpha = 0.5 * np.log((1.0 - err_c) / err_c)
        w = w * np.exp(-alpha * ys * pred)
        w = w / float(np.sum(w))

        predictors.append(fitted)
        alphas.append(float(alpha))
        errs.append(float(err))
        F = F + alpha * pred
        train_errs.append(float(np.mean(np.sign(F) != ys)))

    def predict(Xnew, _ps=predictors, _al=alphas, _cls=classes):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        s = np.zeros(B.shape[0])
        for a, f in zip(_al, _ps):
            s = s + a * np.asarray(f(B), dtype=float).ravel()
        return np.where(s >= 0, _cls[1], _cls[0])

    return RichResult(
        title="AdaBoost (discrete)",
        summary_lines=[("Rounds", M), ("Final training error", train_errs[-1])],
        payload={
            "alphas": np.asarray(alphas),
            "errors": np.asarray(errs),
            "train_errors": np.asarray(train_errs),
            "predict": predict,
            "decision": F,
            "margin": ys * F / float(np.sum(np.abs(alphas))) if np.sum(np.abs(alphas)) > 0 else np.zeros(n),
            "weights": w,
            "classes": classes,
            "estimate": float(train_errs[-1]),
            "n": int(n),
            "method": "Discrete AdaBoost with reweighted weak learners",
        },
    )


def cheatsheet():
    return "hmadab: AdaBoost: train sequential weighted weak learners"


# compact alias per ledger/NAMING.md
geronadaboost = geron_adaboost
