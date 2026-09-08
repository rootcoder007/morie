# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quadratic discriminant analysis (ESL Ch 4.3)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_qda"]


def esl_qda(X, y, query=None):
    """
    QDA discriminant
    delta_k(x) = -(1/2) log|S_k| - (1/2)(x-mu_k)' S_k^-1 (x-mu_k) + log pi_k.

    QDA drops LDA's common-covariance assumption and fits a separate
    covariance per class, which is why the boundary becomes quadratic.
    The cost is parameters: each class needs its own p(p+1)/2
    covariance entries, so every class must have MORE observations
    than features or its covariance is singular. That is checked and
    refused explicitly rather than producing a silent inf.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Features.
    y : array-like, shape (n,)
        Class labels.
    query : array-like, optional
        Points to classify; None means the training points.

    Returns
    -------
    result : dict
        Keys: estimate (predicted class of the first query point),
        prediction, discriminants (row-major), classes, priors,
        log_dets, n, p, K, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 4.3 (Eq. 4.12).

    Examples
    --------
    Classes with different spreads — the tight one wins near its
    centre even though the wide one is not much further away:

    >>> X = [[0.0], [0.1], [-0.1], [5.0], [10.0], [-5.0]]
    >>> y = [0, 0, 0, 1, 1, 1]
    >>> out = esl_qda(X, y)
    >>> out["prediction"][:3]
    [0, 0, 0]
    >>> out["priors"]
    [0.5, 0.5]
    >>> esl_qda([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])["K"]
    2
    >>> esl_qda([[0.0, 1.0], [1.0, 2.0], [5.0, 5.0], [6.0, 6.0]], [0, 0, 1, 1])
    Traceback (most recent call last):
        ...
    ValueError: class 0 has 2 observations but 2 features; QDA needs more observations than features per class.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y_arr = np.asarray(y).ravel()
    n, p = X.shape
    if y_arr.size != n:
        raise ValueError(f"X has {n} rows but y has {y_arr.size} labels.")
    classes = sorted(set(y_arr.tolist()), key=repr)
    K = len(classes)
    if K < 2:
        raise ValueError("QDA needs at least two classes.")
    mus, pis, covs, logdets = [], [], [], []
    for c in classes:
        Xi = X[y_arr == c]
        nk = Xi.shape[0]
        if nk <= p:
            raise ValueError(f"class {c} has {nk} observations but {p} features; "
                             "QDA needs more observations than features per class.")
        mu = Xi.mean(axis=0)
        C = Xi - mu
        S = C.T @ C / (nk - 1)
        sign, ld = np.linalg.slogdet(S)
        if sign <= 0:
            raise ValueError(f"class {c} has a singular covariance; QDA cannot proceed.")
        mus.append(mu); pis.append(nk / n); covs.append(np.linalg.inv(S)); logdets.append(ld)
    Q = X if query is None else np.atleast_2d(np.asarray(query, dtype=float))
    D = np.empty((Q.shape[0], K))
    for j in range(K):
        d = Q - mus[j]
        maha = np.einsum("ij,jk,ik->i", d, covs[j], d)
        D[:, j] = -0.5 * logdets[j] - 0.5 * maha + np.log(pis[j])
    pred = [classes[i] for i in np.argmax(D, axis=1)]
    return RichResult(payload={
        "estimate": pred[0], "prediction": pred,
        "discriminants": [float(v) for v in D.ravel()],
        "classes": [c if isinstance(c, (int, float, str)) else repr(c) for c in classes],
        "priors": [float(v) for v in pis],
        "log_dets": [float(v) for v in logdets],
        "n": int(n), "p": int(p), "K": int(K),
        "method": "QDA, per-class covariance (n_k-1 divisor), quadratic boundary"})


def cheatsheet():
    return "eslqda: per-class covariance; needs n_k > p in EVERY class"


# compact alias per ledger/NAMING.md
eslqda = esl_qda
