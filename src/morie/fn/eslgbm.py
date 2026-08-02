# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient boosting for squared-error loss (ESL Ch 10.10)."""

from . import _array_core as np

from ._richresult import RichResult
from .esldct import esl_decision_tree, esl_tree_predict

__all__ = ["esl_gbm", "esl_gbm_predict"]


def esl_gbm(X, y, M=100, nu=0.1, max_depth=2, min_leaf=1):
    """
    Gradient boosting: f_m(x) = f_{m-1}(x) + nu * h_m(x).

    For squared-error loss the negative gradient IS the residual, so
    each stage fits a regression tree to the current residuals and
    adds a shrunken version of it — ESL Algorithm 10.3. The
    initialiser f_0 is the mean of y, which is the constant
    minimising squared error.

    Two knobs that trade off against each other, per ESL Ch 10.12.1:
    the learning rate nu and the number of trees M. Smaller nu almost
    always generalises better but needs proportionally larger M, so
    setting nu = 0.01 with M = 100 underfits badly. Tree depth
    controls interaction order: depth 1 (stumps) can only produce an
    additive model, depth 2 allows pairwise interactions.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Features.
    y : array-like, shape (n,)
        Numeric response.
    M : int
        Boosting stages, >= 1.
    nu : float
        Learning rate in (0, 1].
    max_depth, min_leaf
        Passed to each stage's tree.

    Returns
    -------
    result : dict
        Keys: estimate (final training RSS), f0, trees, nu, M,
        train_rss_path, fitted, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 10.10 (Alg. 10.3) and
    Ch 10.12.1 on shrinkage.

    Examples
    --------
    Training error falls monotonically for squared-error boosting:

    >>> X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    >>> y = [0.0, 1.0, 4.0, 9.0, 16.0]
    >>> out = esl_gbm(X, y, M=50, nu=0.3, max_depth=1)
    >>> path = out["train_rss_path"]
    >>> all(path[i + 1] <= path[i] + 1e-9 for i in range(len(path) - 1))
    True
    >>> out["estimate"] < 1.0
    True

    A smaller learning rate at the same M fits less far:

    >>> slow = esl_gbm(X, y, M=50, nu=0.02, max_depth=1)
    >>> slow["estimate"] > out["estimate"]
    True
    >>> [round(v, 4) for v in esl_gbm_predict(out, [[0.0], [4.0]])]
    [0.0071, 15.9907]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    M = int(M)
    nu = float(nu)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if M < 1:
        raise ValueError(f"M must be >= 1; got {M}.")
    if not 0 < nu <= 1:
        raise ValueError(f"the learning rate must lie in (0, 1]; got {nu}.")
    f0 = float(np.mean(y))
    F = np.full(n, f0)
    trees, path = [], []
    for _ in range(M):
        resid = y - F                                   # = negative gradient
        if float(np.ptp(resid)) == 0.0:
            break
        t = esl_decision_tree(X, resid, max_depth=max_depth, min_leaf=min_leaf)["tree"]
        trees.append(t)
        F = F + nu * np.asarray(esl_tree_predict(t, X), dtype=float)
        r = y - F
        path.append(float(r @ r))
    resid = y - F
    return RichResult(payload={
        "estimate": float(resid @ resid), "f0": f0, "trees": trees,
        "nu": nu, "M": len(trees), "train_rss_path": path,
        "fitted": [float(v) for v in F], "n": int(n), "p": int(p),
        "method": "gradient boosting, squared-error (residual = negative gradient), shrunk by nu"})


def esl_gbm_predict(model, X):
    """
    Score new rows through a model returned by [esl_gbm].

    Parameters
    ----------
    model : dict
        The payload from esl_gbm (needs f0, trees, nu).
    X : array-like, shape (m, p)
        Rows to predict.

    Returns
    -------
    list of float

    Examples
    --------
    >>> m = esl_gbm([[0.0], [1.0]], [0.0, 2.0], M=10, nu=0.5, max_depth=1)
    >>> len(esl_gbm_predict(m, [[0.0], [1.0]]))
    2
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    F = np.full(X.shape[0], float(model["f0"]))
    nu = float(model["nu"])
    for t in model["trees"]:
        F = F + nu * np.asarray(esl_tree_predict(t, X), dtype=float)
    return [float(v) for v in F]


def cheatsheet():
    return "eslgbm: residual-fitting trees shrunk by nu; small nu needs large M"
