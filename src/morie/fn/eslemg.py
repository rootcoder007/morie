# morie.fn -- function file (rootcoder007/morie)
"""EM for Gaussian mixtures -- ESL Sec 8.5."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_em_gmm"]


def esl_em_gmm(X, k=2, max_iter=200, tol=1e-6, reg=1e-6, seed=0):
    r"""Fit a ``k``-component Gaussian mixture by expectation-maximisation.

    E step -- responsibilities of component :math:`k` for observation
    :math:`i`:

    .. math::
        \gamma_{ik} = \frac{\pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}
                            {\sum_{l} \pi_l\, \mathcal{N}(x_i \mid \mu_l, \Sigma_l)} .

    M step -- weighted moments, with :math:`N_k = \sum_i \gamma_{ik}`:

    .. math::
        \pi_k = N_k/n, \qquad \mu_k = \tfrac{1}{N_k}\sum_i \gamma_{ik} x_i,
        \qquad
        \Sigma_k = \tfrac{1}{N_k}\sum_i \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top .

    The observed-data log-likelihood is non-decreasing across iterations, and
    this is checked at run time -- a decrease means a bug, not slow
    convergence.

    Mixture likelihoods are unbounded: a component collapsing onto a single
    point drives its determinant to zero and the likelihood to infinity. The
    ``reg`` ridge on each covariance diagonal is what keeps that from
    happening, so it is a modelling choice and not a numerical nicety.

    Parameters
    ----------
    X : array-like
        Data, shape ``(n, p)``. A 1-D input is treated as ``(n, 1)``.
    k : int
        Number of components; must be at least 1 and at most ``n``.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Stop when the log-likelihood improves by less than this.
    reg : float
        Ridge added to each covariance diagonal, guarding against collapse.
    seed : int
        Seed for the k-means++ style initialisation.

    Returns
    -------
    RichResult
        ``pi``, ``mu`` ``(k, p)``, ``sigma`` ``(k, p, p)``, ``resp``
        ``(n, k)``, ``labels``, ``loglik``, ``loglik_path``, ``n_iter``,
        ``converged``, ``aic``, ``bic``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Two well-separated clusters are recovered, and the means come back in
    the data's own units.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(-4, 0.5, 200), rng.normal(4, 0.5, 200)]
    >>> r = esl_em_gmm(X, k=2, seed=1)
    >>> mu = np.sort(r["mu"].ravel())
    >>> bool(abs(mu[0] + 4) < 0.3 and abs(mu[1] - 4) < 0.3)
    True
    >>> bool(abs(r["pi"][0] - 0.5) < 0.1)
    True

    The log-likelihood never decreases -- the defining property of EM.

    >>> path = np.asarray(r["loglik_path"])
    >>> bool(np.all(np.diff(path) > -1e-8))
    True

    >>> esl_em_gmm([[1.0], [2.0]], k=5)
    Traceback (most recent call last):
        ...
    ValueError: k=5 exceeds the number of observations (2)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > n:
        raise ValueError(f"k={k} exceeds the number of observations ({n})")

    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centres], axis=0)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centres.append(X[rng.choice(n, p=probs)])
    mu = np.array(centres, dtype=float)
    sigma = np.array([np.cov(X, rowvar=False).reshape(p, p) + reg * np.eye(p)] * k)
    pi = np.full(k, 1.0 / k)

    path, prev, converged = [], -np.inf, False
    for it in range(1, max_iter + 1):
        logp = np.empty((n, k))
        for j in range(k):
            logp[:, j] = np.log(pi[j] + 1e-300) + _log_mvn(X, mu[j], sigma[j])
        mx = logp.max(axis=1, keepdims=True)
        lse = mx.ravel() + np.log(np.exp(logp - mx).sum(axis=1))
        ll = float(lse.sum())
        resp = np.exp(logp - lse[:, None])
        path.append(ll)
        if ll + 1e-9 < prev:
            raise RuntimeError(
                f"EM log-likelihood decreased ({prev:.10g} -> {ll:.10g}); this is a bug"
            )
        if abs(ll - prev) < tol:
            converged = True
            prev = ll
            break
        prev = ll
        Nk = resp.sum(axis=0) + 1e-300
        pi = Nk / n
        mu = (resp.T @ X) / Nk[:, None]
        for j in range(k):
            d = X - mu[j]
            sigma[j] = (resp[:, j, None] * d).T @ d / Nk[j] + reg * np.eye(p)

    n_par = k - 1 + k * p + k * p * (p + 1) / 2
    return RichResult(
        title="Gaussian mixture (EM)",
        summary_lines=[("n", n), ("p", p), ("k", k), ("loglik", prev), ("iterations", it)],
        warnings=[] if converged else [f"did not converge in {max_iter} iterations"],
        payload={
            "pi": pi, "mu": mu, "sigma": sigma, "resp": resp,
            "labels": resp.argmax(axis=1),
            "loglik": float(prev), "loglik_path": np.asarray(path),
            "n_iter": int(it), "converged": bool(converged),
            "aic": float(2 * n_par - 2 * prev),
            "bic": float(n_par * np.log(n) - 2 * prev),
            "n": int(n), "k": int(k),
            "method": "esl_em_gmm",
        },
    )


def _log_mvn(X, mu, S):
    """Log density of N(mu, S) at each row of X, via Cholesky."""
    p = X.shape[1]
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (p * np.log(2 * np.pi) + (z**2).sum(0)) - np.log(np.diag(L)).sum()


def cheatsheet():
    return "eslemg: EM for GMM; loglik is asserted non-decreasing, and `reg` is what stops component collapse"


# compact alias per ledger/NAMING.md
eslemgmm = esl_em_gmm
