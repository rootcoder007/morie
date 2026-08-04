# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""k-nearest-neighbour fit (ESL Ch 2.3.2 / 13.3)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_knn"]


def esl_knn(X, y, k, query=None):
    """
    k-nearest neighbours: f_hat(x) = (1/k) sum_{x_i in N_k(x)} y_i.

    Averages the k nearest responses under Euclidean distance. With
    0/1 labels the average IS the majority vote and also the class
    probability, so this one routine serves regression and
    classification -- ESL Ch 2 makes exactly that point. Ties in
    distance are broken by lower index, which keeps the answer
    deterministic; without a rule, two runs can disagree. Note kNN is
    scale-sensitive (a variable measured in metres dominates one in
    kilometres), so the payload reports whether the columns are
    standardised.

    ``query = None`` fits at the training points themselves, which is
    the in-sample fit and includes each point as its own neighbour --
    that is why k = 1 gives zero training error and tells you nothing.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Training features.
    y : array-like, shape (n,)
        Responses or 0/1 labels.
    k : int
        Neighbours, 1 <= k <= n.
    query : array-like, optional
        Points to predict at; None means the training points.

    Returns
    -------
    result : dict
        Keys: estimate (first prediction), prediction, neighbours
        (0-based indices, row-major), k, columns_standardised, n, p,
        method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 2.3.2 (Eq. 2.8).

    Examples
    --------
    k = 1 at the training points returns the data exactly:

    >>> X = [[0.0], [1.0], [2.0], [10.0]]
    >>> y = [0.0, 0.0, 1.0, 1.0]
    >>> esl_knn(X, y, 1)["prediction"]
    [0.0, 0.0, 1.0, 1.0]
    >>> esl_knn(X, y, 3)["prediction"]        # at x=10 the neighbours are 10, 2, 1
    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.6666666666666666]
    >>> esl_knn(X, y, 4)["prediction"][0]
    0.5
    >>> esl_knn(X, y, 5)
    Traceback (most recent call last):
        ...
    ValueError: k must lie in [1, 4]; got 5.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    k = int(k)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}]; got {k}.")
    Q = X if query is None else np.atleast_2d(np.asarray(query, dtype=float))
    if Q.shape[1] != p:
        raise ValueError(f"query has {Q.shape[1]} columns but X has {p}.")
    preds, nbrs = [], []
    for q in Q:
        d = np.sum((X - q) ** 2, axis=1)
        idx = np.argsort(d, kind="stable")[:k]      # stable => ties to lower index
        nbrs.append(idx)
        preds.append(float(np.mean(y[idx])))
    sd = X.std(axis=0)
    return RichResult(payload={
        "estimate": preds[0], "prediction": preds,
        "neighbours": [int(v) for row in nbrs for v in row],
        "k": k, "columns_standardised": bool(np.allclose(sd, 1.0, atol=1e-8)),
        "n": int(n), "p": int(p),
        "method": "kNN mean of k nearest responses; ties to lower index"})


def cheatsheet():
    return "eslknn: mean of k nearest y; = majority vote for 0/1; scale-sensitive"


# compact alias per ledger/NAMING.md
eslknn = esl_knn
