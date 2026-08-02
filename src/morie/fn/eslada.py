# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AdaBoost.M1 (ESL Ch 10.1)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_adaboost", "esl_adaboost_predict"]


def _weighted_stump(X, y, w):
    """Weighted-error-minimising axis-aligned stump; deterministic ties."""
    n, p = X.shape
    best = (float("inf"), 0, 0.0, 1)
    for j in range(p):
        for thr in np.unique(X[:, j]):
            for sg in (1, -1):
                pred = np.where(X[:, j] <= thr, sg, -sg)
                err = float(np.sum(w[pred != y]))
                if err < best[0] - 1e-15:
                    best = (err, j, float(thr), sg)
    return best


def esl_adaboost(X, y, M=50):
    """
    AdaBoost.M1 with decision stumps.

    ESL Algorithm 10.1: weight every observation equally, fit a weak
    classifier, give it weight alpha_m = log((1 - err_m) / err_m),
    then multiply the weights of the misclassified points by
    exp(alpha_m) and renormalise. Points the committee keeps getting
    wrong therefore dominate later rounds.

    Note the alpha here is ESL's log((1-err)/err) — NOT the
    half-log used in some texts. The two differ by a factor of two,
    which cancels in the sign of the committee vote but not in the
    reported alphas, so it is stated rather than left ambiguous.

    Stops early when a stump is perfect (alpha would be infinite) or
    no better than chance, and reports how many rounds actually ran.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Features.
    y : array-like, shape (n,)
        Labels in {-1, +1}.
    M : int
        Maximum boosting rounds, >= 1.

    Returns
    -------
    result : dict
        Keys: estimate (training error rate), stumps, alphas,
        rounds_used, prediction, margin, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 10.1 (Alg. 10.1).

    Examples
    --------
    A threshold problem is solved by the first stump:

    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [1, 1, -1, -1]
    >>> out = esl_adaboost(X, y)
    >>> out["estimate"]
    0.0
    >>> out["rounds_used"]
    1
    >>> esl_adaboost_predict(out, [[0.5], [2.5]])
    [1, -1]

    XOR-like data needs several stumps and still gets there:

    >>> X2 = [[0.0], [1.0], [2.0], [3.0]]
    >>> y2 = [1, -1, 1, -1]
    >>> out2 = esl_adaboost(X2, y2, M=20)
    >>> out2["rounds_used"] > 1
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    M = int(M)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} labels.")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("labels must lie in {-1, +1}.")
    if M < 1:
        raise ValueError(f"M must be >= 1; got {M}.")
    w = np.full(n, 1.0 / n)
    stumps, alphas = [], []
    F = np.zeros(n)
    for _ in range(M):
        err, j, thr, sg = _weighted_stump(X, y, w)
        if err >= 0.5 - 1e-12:
            break
        perfect = err <= 1e-15
        alpha = 10.0 if perfect else float(np.log((1.0 - err) / err))
        pred = np.where(X[:, j] <= thr, sg, -sg).astype(float)
        stumps.append({"feature": int(j), "threshold": thr, "sign": int(sg)})
        alphas.append(alpha)
        F = F + alpha * pred
        if perfect:
            break
        w = w * np.exp(alpha * (pred != y))
        w = w / w.sum()
    committee = np.where(F >= 0, 1, -1)
    return RichResult(payload={
        "estimate": float(np.mean(committee != y)),
        "stumps": stumps, "alphas": alphas, "rounds_used": len(stumps),
        "prediction": [int(v) for v in committee],
        "margin": [float(v) for v in y * F],
        "n": int(n), "p": int(p),
        "method": "AdaBoost.M1 (Alg. 10.1), alpha = log((1-err)/err), stumps"})


def esl_adaboost_predict(model, X):
    """
    Classify new rows with a committee returned by [esl_adaboost].

    Parameters
    ----------
    model : dict
        Payload from esl_adaboost (needs stumps and alphas).
    X : array-like, shape (m, p)

    Returns
    -------
    list of int
        Labels in {-1, +1}.

    Examples
    --------
    >>> m = esl_adaboost([[0.0], [1.0]], [1, -1])
    >>> esl_adaboost_predict(m, [[0.0], [1.0]])
    [1, -1]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    F = np.zeros(X.shape[0])
    for s, a in zip(model["stumps"], model["alphas"]):
        pred = np.where(X[:, s["feature"]] <= s["threshold"], s["sign"], -s["sign"])
        F = F + a * pred
    return [int(v) for v in np.where(F >= 0, 1, -1)]


def cheatsheet():
    return "eslada: alpha = log((1-err)/err) (ESL form, not half-log); stumps + predict"
