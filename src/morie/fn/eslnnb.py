# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian naive Bayes (ESL Ch 6.6.3)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_naive_bayes"]


def esl_naive_bayes(X, y, query=None, var_smoothing=1e-9):
    """
    Naive Bayes: f(x) = argmax_k pi_k prod_j p_kj(x_j).

    The "naive" assumption is that features are conditionally
    INDEPENDENT within a class, so the joint density factorises into
    a product of one-dimensional densities. That assumption is
    usually false, and the classifier usually works anyway — because
    the argmax only needs the ranking of the class scores to be
    right, not the probabilities themselves. Worth knowing: the
    returned posteriors are typically over-confident even when the
    predicted class is correct.

    Densities are Gaussian per feature per class, with a small
    variance floor so a constant feature within a class cannot
    produce a divide-by-zero. Scores are accumulated as LOG
    probabilities; multiplying many small densities underflows.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Features.
    y : array-like, shape (n,)
        Class labels.
    query : array-like, optional
        Points to classify; None means the training points.
    var_smoothing : float
        Added to every variance.

    Returns
    -------
    result : dict
        Keys: estimate (predicted class of the first query point),
        prediction, log_posterior (row-major, unnormalised), classes,
        priors, n, p, K, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 6.6.3.

    Examples
    --------
    >>> X = [[0.0, 0.0], [0.5, 0.2], [10.0, 9.0], [9.5, 10.0]]
    >>> y = ["a", "a", "b", "b"]
    >>> out = esl_naive_bayes(X, y)
    >>> out["prediction"]
    ['a', 'a', 'b', 'b']
    >>> out["classes"]
    ['a', 'b']
    >>> out["priors"]
    [0.5, 0.5]
    >>> esl_naive_bayes(X, y, query=[[0.2, 0.1]])["estimate"]
    'a'
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y_arr = np.asarray(y).ravel()
    n, p = X.shape
    if y_arr.size != n:
        raise ValueError(f"X has {n} rows but y has {y_arr.size} labels.")
    classes = sorted(set(y_arr.tolist()), key=repr)
    K = len(classes)
    if K < 2:
        raise ValueError("naive Bayes needs at least two classes.")
    mus, varis, pis = [], [], []
    for c in classes:
        Xi = X[y_arr == c]
        mus.append(Xi.mean(axis=0))
        varis.append(Xi.var(axis=0) + float(var_smoothing))
        pis.append(Xi.shape[0] / n)
    Q = X if query is None else np.atleast_2d(np.asarray(query, dtype=float))
    L = np.empty((Q.shape[0], K))
    for j in range(K):
        v = varis[j]
        ll = -0.5 * np.sum(np.log(2.0 * np.pi * v) + (Q - mus[j]) ** 2 / v, axis=1)
        L[:, j] = ll + np.log(pis[j])
    pred = [classes[i] for i in np.argmax(L, axis=1)]
    return RichResult(payload={
        "estimate": pred[0], "prediction": pred,
        "log_posterior": [float(v) for v in L.ravel()],
        "classes": [c if isinstance(c, (int, float, str)) else repr(c) for c in classes],
        "priors": [float(v) for v in pis],
        "n": int(n), "p": int(p), "K": int(K),
        "method": "Gaussian naive Bayes in log space; posteriors over-confident by design"})


def cheatsheet():
    return "eslnnb: log-space product of per-feature Gaussians; ranking right, probs not"


# compact alias per ledger/NAMING.md
eslnaivebayes = esl_naive_bayes
