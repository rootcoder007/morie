# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linear discriminant analysis (ESL Ch 4.3)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_lda_disc"]


def esl_lda_disc(X, y, query=None):
    """
    LDA discriminant
    delta_k(x) = x' S^-1 mu_k - (1/2) mu_k' S^-1 mu_k + log pi_k.

    LDA's defining assumption is a COMMON covariance across classes,
    which is what makes the discriminant linear in x -- the quadratic
    term cancels between classes. The pooled estimate uses the
    (n - K) divisor. If that assumption is wrong, the right tool is
    QDA (eslqda), which keeps the quadratic term; the payload reports
    the per-class covariance spread so the assumption is checkable
    rather than assumed.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Features.
    y : array-like, shape (n,)
        Class labels (any hashable; sorted for a stable order).
    query : array-like, optional
        Points to classify; None means the training points.

    Returns
    -------
    result : dict
        Keys: estimate (predicted class of the first query point),
        prediction, discriminants (row-major n x K), classes, priors,
        means (row-major K x p), pooled_covariance (row-major),
        n, p, K, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 4.3 (Eq. 4.10).

    Examples
    --------
    Two well-separated classes on a line:

    >>> X = [[0.0], [1.0], [10.0], [11.0]]
    >>> y = [0, 0, 1, 1]
    >>> out = esl_lda_disc(X, y)
    >>> out["prediction"]
    [0, 0, 1, 1]
    >>> out["priors"]
    [0.5, 0.5]
    >>> [round(m, 6) for m in out["means"]]
    [0.5, 10.5]
    >>> esl_lda_disc(X, y, query=[[5.0]])["prediction"]
    [0]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y_arr = np.asarray(y).ravel()
    n, p = X.shape
    if y_arr.size != n:
        raise ValueError(f"X has {n} rows but y has {y_arr.size} labels.")
    classes = sorted(set(y_arr.tolist()), key=repr)
    K = len(classes)
    if K < 2:
        raise ValueError("LDA needs at least two classes.")
    if n <= K:
        raise ValueError(f"pooled covariance needs n > K; got n={n}, K={K}.")
    means, priors, S = [], [], np.zeros((p, p))
    for c in classes:
        Xi = X[y_arr == c]
        mu = Xi.mean(axis=0)
        means.append(mu)
        priors.append(Xi.shape[0] / n)
        C = Xi - mu
        S += C.T @ C
    S /= (n - K)
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        raise ValueError("the pooled covariance is singular; reduce dimensions or use more data.")
    Q = X if query is None else np.atleast_2d(np.asarray(query, dtype=float))
    D = np.empty((Q.shape[0], K))
    for j, (mu, pi_) in enumerate(zip(means, priors)):
        D[:, j] = Q @ Sinv @ mu - 0.5 * float(mu @ Sinv @ mu) + np.log(pi_)
    pred = [classes[i] for i in np.argmax(D, axis=1)]
    spread = float(np.max([np.linalg.norm(np.cov((X[y_arr == c] - np.mean(X[y_arr == c], axis=0)).T,
                                                 bias=True))
                           for c in classes])) if p == 1 else float("nan")
    return RichResult(payload={
        "estimate": pred[0], "prediction": pred,
        "discriminants": [float(v) for v in D.ravel()],
        "classes": [c if isinstance(c, (int, float, str)) else repr(c) for c in classes],
        "priors": [float(v) for v in priors],
        "means": [float(v) for m in means for v in np.atleast_1d(m)],
        "pooled_covariance": [float(v) for v in S.ravel()],
        "n": int(n), "p": int(p), "K": int(K),
        "method": "LDA, common pooled covariance (n-K divisor), linear discriminant"})


def cheatsheet():
    return "esllsc: common covariance => linear; use eslqda when that fails"
