# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Elastic net, ESL/glmnet parameterisation (ESL Ch 3.4.3 / 18.4)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_elastic_net"]


def _enet(X, y, lam1, lam2, max_iter, tol):
    """Coordinate descent for (1/2)|y-Xb|^2 + lam1|b|_1 + (lam2/2)|b|^2.

    Constant columns are left unpenalised, as in eslrdg/esllso.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    colsq = np.sum(X ** 2, axis=0)
    if np.any(colsq == 0):
        raise ValueError("an all-zero column cannot be penalised meaningfully.")
    const = np.array([bool(np.ptp(X[:, j]) == 0) for j in range(p)])
    beta = np.zeros(p)
    r = y.copy()
    converged, it = False, 0
    for it in range(1, int(max_iter) + 1):
        delta = 0.0
        for j in range(p):
            old = beta[j]
            rho = X[:, j] @ r + colsq[j] * old
            l1 = 0.0 if const[j] else lam1
            l2 = 0.0 if const[j] else lam2
            new = np.sign(rho) * max(abs(rho) - l1, 0.0) / (colsq[j] + l2)
            if new != old:
                r += X[:, j] * (old - new)
                beta[j] = new
                delta = max(delta, abs(new - old))
        if delta < tol:
            converged = True
            break
    return X, y, beta, r, const, n, p, it, converged


def esl_elastic_net(X, y, lambda_, alpha, max_iter=10000, tol=1e-12):
    """
    Elastic net in the glmnet mixing parameterisation.

    Formula: argmin (1/2)|y - X b|^2 + lambda(alpha |b|_1 +
    ((1-alpha)/2) |b|_2^2). alpha = 1 is the lasso, alpha = 0 is
    ridge, and intermediate values keep the lasso's sparsity while
    the L2 part shares weight among correlated predictors instead of
    picking one arbitrarily -- the grouping effect ESL Ch 18.4 is
    about. Constant columns are unpenalised, matching eslrdg/esllso.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    lambda_ : float
        Overall penalty, >= 0.
    alpha : float
        L1 share in [0, 1].
    max_iter, tol
        Coordinate-descent controls.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, n_nonzero,
        active_set, objective, lambda, alpha, lambda1, lambda2,
        iterations, converged, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.4.3 and Ch 18.4;
    Zou & Hastie (2005).

    Examples
    --------
    Two identical predictors: the lasso keeps one, the elastic net
    splits the weight between them.

    >>> X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    >>> y = [2.0, 4.0, 6.0]
    >>> enet = esl_elastic_net(X, y, 1.0, 0.5)["beta"]
    >>> abs(enet[0] - enet[1]) < 1e-9      # coordinate descent, tol-level equality
    True
    >>> all(v > 0 for v in enet)           # both kept, weight shared
    True
    >>> esl_elastic_net(X, y, 0.0, 1.0)["n_nonzero"] >= 1
    True
    >>> esl_elastic_net(X, y, 1.0, 1.5)
    Traceback (most recent call last):
        ...
    ValueError: alpha must lie in [0, 1]; got 1.5.
    """
    lam = float(lambda_)
    alpha = float(alpha)
    if lam < 0:
        raise ValueError(f"the penalty must be non-negative; got {lam}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1]; got {alpha}.")
    lam1, lam2 = lam * alpha, lam * (1.0 - alpha)
    X, y, beta, r, const, n, p, it, conv = _enet(X, y, lam1, lam2, max_iter, tol)
    active = [int(j) for j in np.flatnonzero(beta != 0)]
    obj = (0.5 * float(r @ r) + lam1 * float(np.sum(np.abs(beta[~const])))
           + 0.5 * lam2 * float(np.sum(beta[~const] ** 2)))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "n_nonzero": len(active), "active_set": active, "objective": obj,
        "lambda": lam, "alpha": alpha, "lambda1": lam1, "lambda2": lam2,
        "iterations": int(it), "converged": bool(conv), "n": int(n), "p": int(p),
        "method": "elastic net, glmnet (lambda, alpha) parameterisation"})


def cheatsheet():
    return "eslnln: lambda(alpha L1 + (1-alpha)/2 L2); alpha=1 lasso, 0 ridge"


# compact alias per ledger/NAMING.md
eslelasticnet = esl_elastic_net
