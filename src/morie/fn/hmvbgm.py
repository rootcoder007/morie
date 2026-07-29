# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayesian Gaussian mixture with variational inference (VBGMM)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_variational_bayes_gmm", "digamma"]


def digamma(x):
    """Digamma psi(x) for x > 0, by recurrence plus the asymptotic series.

    numpy has no psi and scipy is not a dependency here, so it is computed
    directly: shift x above 6 with ``psi(x) = psi(x+1) - 1/x``, then use
    ``psi(x) ~ ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8)``.

    >>> round(float(digamma(1.0)), 9)
    -0.577215665
    >>> round(float(digamma(2.0) - digamma(1.0)), 12)
    1.0
    """
    a = np.asarray(x, dtype=float)
    if np.any(a <= 0):
        raise ValueError("digamma: defined here for x > 0 only")
    r = np.zeros_like(a)
    a = a.copy()
    while np.any(a < 6.0):
        m = a < 6.0
        r[m] -= 1.0 / a[m]
        a[m] += 1.0
    f = 1.0 / (a * a)
    return r + np.log(a) - 0.5 / a + f * (-1.0 / 12 + f * (1.0 / 120 + f * (-1.0 / 252 + f * (1.0 / 240))))


def geron_variational_bayes_gmm(X, n_components=3, max_iter=100, alpha0=1e-2, tol=1e-6, var_floor=1e-6, seed=0):
    """
    Bayesian Gaussian mixture with variational inference (VBGMM).

    Formula: q(theta) minimizes KL(q || p(theta|X)) with Dirichlet prior on pi

    Mean-field variational Bayes over the mixing weights with a
    ``Dirichlet(alpha0)`` prior. The difference from plain EM is one line
    and it is the whole point: the responsibility uses the *expected log*
    weight under the variational posterior,

    ``E_q[log pi_k] = psi(alpha_k) - psi(sum_j alpha_j)``  (see :func:`digamma`),

    not ``log pi_k``. Because ``psi`` diverges to -inf as ``alpha_k -> 0``,
    a small ``alpha0`` actively drives unneeded components to zero weight
    instead of splitting the data between them -- automatic relevance
    determination, which maximum likelihood cannot do. The component means
    and (diagonal) variances are then updated from the responsibilities.

    Parameters
    ----------
    X : array-like
        Data (n, d).
    n_components : int, default 3
        Maximum number of components (1 <= K <= n).
    max_iter : int, default 100
        Variational sweeps (>= 1).
    alpha0 : float, default 1e-2
        Dirichlet concentration; small values prune components (> 0).
    tol : float, default 1e-6
        Convergence tolerance on the mean responsibility change.
    var_floor : float, default 1e-6
        Lower bound on component variances (> 0), which stops a component
        collapsing onto a single point and taking the likelihood to infinity.
    seed : int, default 0
        LCG seed for the initial means.

    Returns
    -------
    result : RichResult
        Keys: weights, means, variances, resp, labels, alpha, n_effective,
        n_iter, estimate, n, method.

    Examples
    --------
    Two well-separated groups, three components offered: the Dirichlet
    prior prunes the surplus component instead of splitting a real one.

    >>> X = [[0.0], [0.1], [0.2], [9.8], [9.9], [10.0]]
    >>> r = geron_variational_bayes_gmm(X, n_components=3, alpha0=1e-3, max_iter=200)
    >>> int(r["n_effective"])
    2
    >>> sorted(round(float(m), 1) for m in r["means"][r["weights"] > 0.01].ravel())
    [0.1, 9.9]
    >>> round(float(r["weights"].sum()), 9)
    1.0
    >>> len(set(int(v) for v in r["labels"]))
    2

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_variational_bayes_gmm: X must be a non-empty (n, d) matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_variational_bayes_gmm: X contains non-finite values")
    n, d = A.shape
    K = int(n_components)
    if not (1 <= K <= n):
        raise ValueError(f"geron_variational_bayes_gmm: n_components must lie in 1..{n}, got {K}")
    it_max = int(max_iter)
    if it_max < 1:
        raise ValueError(f"geron_variational_bayes_gmm: max_iter must be >= 1, got {it_max}")
    a0 = float(alpha0)
    if not np.isfinite(a0) or a0 <= 0:
        raise ValueError(f"geron_variational_bayes_gmm: alpha0 must be positive and finite, got {a0}")
    vf = float(var_floor)
    if not np.isfinite(vf) or vf <= 0:
        raise ValueError(f"geron_variational_bayes_gmm: var_floor must be positive and finite, got {vf}")

    s = int(seed) % 2**32
    idx = []
    for _ in range(K):
        s = (1664525 * s + 1013904223) % 2**32
        idx.append(int(((s + 0.5) / 2**32) * n) % n)
    order = np.argsort(A[:, 0], kind="mergesort")
    means = np.asarray([A[order[int(i * (n - 1) / max(1, K - 1))]] for i in range(K)], dtype=float)
    variances = np.tile(np.maximum(A.var(axis=0), vf), (K, 1))
    alpha = np.full(K, a0 + n / K)
    resp = np.full((n, K), 1.0 / K)

    n_iter = 0
    for n_iter in range(1, it_max + 1):
        # -- variational E step: expected log weight, not log weight ------
        elog_pi = digamma(alpha) - digamma(np.sum(alpha))
        log_norm = -0.5 * np.sum(np.log(2 * np.pi * variances), axis=1)
        quad = np.empty((n, K))
        for k in range(K):
            quad[:, k] = -0.5 * np.sum((A - means[k]) ** 2 / variances[k], axis=1)
        log_r = elog_pi + log_norm + quad
        log_r -= log_r.max(axis=1, keepdims=True)
        new_resp = np.exp(log_r)
        new_resp /= new_resp.sum(axis=1, keepdims=True)
        delta = float(np.max(np.abs(new_resp - resp)))
        resp = new_resp

        # -- M step: Dirichlet posterior on pi, MAP Gaussians -------------
        Nk = resp.sum(axis=0)
        alpha = a0 + Nk
        for k in range(K):
            if Nk[k] > 1e-12:
                means[k] = (resp[:, k] @ A) / Nk[k]
                variances[k] = np.maximum((resp[:, k] @ (A - means[k]) ** 2) / Nk[k], vf)
        if delta < tol:
            break

    weights = alpha / np.sum(alpha)
    labels = np.argmax(resp, axis=1)
    n_eff = int(np.sum(weights > 1.0 / (10.0 * K)))

    return RichResult(
        title="Variational Bayesian Gaussian mixture",
        summary_lines=[
            ("Components offered", K),
            ("Components used", n_eff),
            ("Sweeps", n_iter),
            ("alpha0", a0),
        ],
        interpretation=(
            "The Dirichlet prior is what lets the model switch components off: with a small alpha0, "
            "psi(alpha_k) collapses for an unused component and its responsibility goes to zero."
        ),
        payload={
            "weights": weights,
            "means": means,
            "variances": variances,
            "resp": resp,
            "labels": labels,
            "alpha": alpha,
            "n_effective": n_eff,
            "n_iter": int(n_iter),
            "estimate": float(n_eff),
            "n": int(n),
            "method": "Mean-field VB with a Dirichlet prior on the weights (E[log pi] via digamma) and MAP diagonal Gaussians",
        },
    )


def cheatsheet():
    return "hmvbgm: Bayesian Gaussian mixture with variational inference (VBGMM)"
